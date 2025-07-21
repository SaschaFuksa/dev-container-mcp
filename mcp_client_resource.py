"""Langchain MCP Resource Client."""

import asyncio
import logging
import sys

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def setup_logging() -> logging.Logger:
    """Set up logging for the client."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


LOGGER = setup_logging()


class MCPResourceTool(BaseTool):
    """Tool for accessing MCP resources such as files, data, or context using a MultiServerMCPClient."""

    name: str = "mcp_resource_access"
    description: str = "Access MCP resources like files, data, or context"
    mcp_client: MultiServerMCPClient

    class MCPResourceInput(BaseModel):
        """Input schema for specifying the URI of the resource to access."""

        resource_uri: str = Field(description="URI of the resource to access")

    args_schema: type[BaseModel] = MCPResourceInput

    def _run(self, resource_uri: str, run_manager=None, *args, **kwargs) -> str:
        # Sync wrapper for async MCP call
        return asyncio.run(self._arun(resource_uri, *args, **kwargs))

    async def _arun(self, resource_uri: str, *args, **kwargs) -> str:
        try:
            content = await self.mcp_client.read_resource(resource_uri)
            return content.contents[0].text
        except Exception as e:
            return f"Error accessing resource {resource_uri}: {e!s}"


class MCPListResourcesTool(BaseTool):
    """Tool for listing all available MCP resources using a MultiServerMCPClient."""

    name: str = "list_mcp_resources"
    description: str = "List all available MCP resources"
    mcp_client: MultiServerMCPClient

    class EmptyInput(BaseModel):
        """Empty input schema for tools that require no arguments."""

    args_schema: type[BaseModel] = EmptyInput

    def _run(self, *args: object, **kwargs: object) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    async def _arun(self, *args: object, **kwargs: object) -> str:
        try:
            resources = await self.mcp_client.list_resources()
            resource_list = [
                f"- {resource.name}: {resource.uri}" for resource in resources.resources
            ]
            return "\n".join(resource_list)
        except Exception as e:
            return f"Error listing resources: {e!s}"


async def create_mcp_agent(
    mcp_server_url: str = "http://127.0.0.1:8000",
) -> tuple[AgentExecutor, MultiServerMCPClient]:
    """
    Create and configure an MCP agent executor and MCP client.

    Returns
    -------
    tuple[AgentExecutor, MultiServerMCPClient]
        The agent executor and the initialized MCP client.

    """
    # Initialize MCP client
    server_config = {
        "default": {
            "url": f"{mcp_server_url}/sse",
            "transport": "sse",
        },
    }
    mcp_client = MultiServerMCPClient(server_config)

    # Initialize Ollama
    llm = ChatOllama(
        model="llama3.2",
        base_url="http://host.docker.internal:11434",
        temperature=0.7,
    )

    # Create MCP tools
    tools = [MCPListResourcesTool(mcp_client=mcp_client), MCPResourceTool(mcp_client=mcp_client)]

    # Agent prompt
    prompt = PromptTemplate.from_template("""
    You are an intelligent assistant with access to MCP resources.

    You have access to these tools:
    {tools}

    Tool Names: {tool_names}

    When a user asks about project information:
    1. First list available resources
    2. Identify which resources are relevant
    3. Access those resources
    4. Provide a comprehensive answer

    Question: {input}

    Thought: {agent_scratchpad}
    """)

    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor, mcp_client


async def main() -> None:
    """Initialize and run the MCP Agent demo queries."""
    LOGGER.info("Initializing MCP Agent...")
    agent_executor, mcp_client = await create_mcp_agent()

    try:
        # Test queries
        queries = [
            "What resources are available?",
            "Tell me about the users in the system",
            "What's the current conversation context?",
        ]

        for query in queries:
            LOGGER.info("\n%s", "=" * 50)
            LOGGER.info("Query: %s", query)
            LOGGER.info("=" * 50)

            result = agent_executor.invoke({"input": query})
            LOGGER.info("Answer: %s", result["output"])

    finally:
        # MultiServerMCPClient doesn't have a close method, so we'll remove this
        pass


if __name__ == "__main__":
    asyncio.run(main())

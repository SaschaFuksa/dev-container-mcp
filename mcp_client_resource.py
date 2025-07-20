"""Langchain MCP Resource Client."""

import asyncio
import logging
import sys

import httpx
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


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


SYSTEM_PROMPT = """You are an AI assistant that helps users interact with a resources.
        You can read data from a text file using the available tools.
        When reading data:
        1. Read the full document to understand the context
        2. Use the tools provided to answer questions
        3. Always provide clear, concise answers

        When greeting user:
        1. Use the calculator_greeting tool to greet users
        2. Personalize the greeting with the user's name

        Always:
        1. Think through each step carefully
        2. Verify actions were successful
        3. Provide clear summaries of what was done"""

TOOL_TEMPLATES = """"Answer the following questions as best you can. You have access to
the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: {tool_names}
Observation: the result of the action
Thought: I now know the final answer
Final Answer: [The formatted answer for get_usage or greeting message for calculator_greeting]

For example:
Question: greet John
Thought: I need to greet John
Action: calculator_greeting
Observation: Hello John!
Thought: I have successfully greeted John
Final Answer: Hello John!

Question: read instructions
Thought: I need to retrieve the instructions for using the calculator
Action: get_usage
Observation: [Formatted usage instructions]
Thought: I have retrieved the usage instructions
Final Answer: The final instructions are as follows:
[Formatted usage instructions]

Begin!

Question: {input}
{agent_scratchpad}"""


class LangchainMCPClient:
    """
    Client for interacting with the Langchain MCP server using Langchain agents and tools.

    This class initializes the connection to the MCP server, sets up the language model,
    manages available tools, and provides methods for interactive chat and message processing.
    """

    def __init__(self, mcp_server_url: str = "http://127.0.0.1:8000") -> None:
        """
        Initialize the LangchainMCPClient.

        Args:
            mcp_server_url (str): The URL of the MCP server to connect to.

        """
        LOGGER.info("Initializing LangchainMCPClient...")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,  # Disable streaming for better compatibility
            base_url="http://host.docker.internal:11434",
        )
        server_config = {
            "default": {
                "url": f"{mcp_server_url}/sse",
                "transport": "sse",
            },
        }
        LOGGER.info("Connecting to MCP server at %s...", mcp_server_url)
        self.mcp_client = MultiServerMCPClient(server_config)
        self.chat_history = []

        # Initialize agent and agent_executor attributes to None
        self.agent = None
        self.agent_executor = None

    async def check_server_connection(self) -> bool | None:
        """Check if the MCP server is accessible."""
        base_url = self.mcp_client.connections["default"]["url"].replace("/sse", "")
        http_status_promt = 200
        try:
            LOGGER.info("Testing connection to %s...", base_url)
            async with httpx.AsyncClient(timeout=5.0) as client:  # Shorter timeout
                # Try the SSE endpoint directly
                sse_url = f"{base_url}/sse"
                LOGGER.info("Checking SSE endpoint at %s...", sse_url)
                response = await client.get(sse_url, timeout=5.0)
                LOGGER.info("Got response: %s", response.status_code)
                if response.status_code == http_status_promt:
                    LOGGER.info("SSE endpoint is accessible!")
                    return True

                LOGGER.info("Server responded with status code: %s", response.status_code)
                return False

        except httpx.ConnectError:
            LOGGER.info("Could not connect to server at %s", base_url)
            LOGGER.info("Please ensure the server is running and the port is correct")
            return False
        except httpx.ReadTimeout:
            LOGGER.info("Connection established but timed out while reading")
            LOGGER.info("This is normal for SSE connections - proceeding...")
            return True
        except httpx.HTTPError as e:
            LOGGER.info("HTTP error connecting to MCP server: %s - %s", type(e).__name__, e)
            return False
        except OSError as e:
            LOGGER.info(
                "OS error connecting to MCP server: %s - %s",
                type(e).__name__,
                e,
            )
            return False

    async def interactive_chat(self) -> None:
        """Start an interactive chat session."""
        LOGGER.info("Chat session started. Type 'exit' to quit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                LOGGER.info("Ending chat session...")
                break

            response = await self._process_message(user_input)
            LOGGER.info("\nAgent: %s", response)

    async def _process_message(self, user_input: str) -> str:
        """
        Process a user input message, invoke the agent, and return the agent's response.

        Args:
            user_input (str): The user's input message.

        Returns:
            str: The agent's response to the input message.

        """
        LOGGER.info("\nProcessing message: %s", user_input)
        try:
            LOGGER.info("\nProcessing message: %s", user_input)
            # Execute the agent
            response = await self.agent_executor.ainvoke(
                {"input": user_input, "chat_history": self.chat_history},
            )

            LOGGER.info("\nRaw response: %s", response)
            final_result = None

            # Get the result from intermediate steps
            if isinstance(response, dict) and "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                if steps and isinstance(steps[-1], tuple):
                    action, observation = steps[-1]

                    # Handle calculator_greeting response
                    if "calculator_greeting" in str(action):
                        final_result = str(action.tool_input)

                    # Handle get_usage response
                    elif "get_usage" in str(action):
                        if isinstance(observation, str) and "Showing" in observation:
                            final_result = observation
                        else:
                            final_result = str(observation)

                    # Use raw observation if no specific handling
                    if final_result is None:
                        final_result = str(observation)

                    # Update response output and chat history
                    response["output"] = final_result
                    self.chat_history.extend(
                        [
                            HumanMessage(content=user_input),
                            AIMessage(content=final_result),
                        ],
                    )

                    LOGGER.info("\nFinal result: %s", final_result)
                    return final_result

            return "Could not process the request. Please try again."

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            error_msg = (
                f"Error processing message: {type(e).__name__} - {e}\n"
                "Please try rephrasing your request."
            )
            LOGGER.info("\nError processing message: %s - %s", type(e).__name__, e)
            LOGGER.info("Full error: %s", getattr(e, "__dict__", {}))
            return error_msg

    async def initialize_agent(self) -> None:
        """Initialize the agent with resource and prompt template."""
        LOGGER.info("\nInitializing agent...")
        if not await self.check_server_connection():
            msg = "Cannot connect to MCP server. Please ensure the server is running."
            raise ConnectionError(
                msg,
            )

        try:
            LOGGER.info("Getting available resources...")
            mcp_resources = await self.mcp_client.get_resources(server_name="default")

            # Verify resources are properly initialized
            LOGGER.info("Verifying resources...")
            for i, resource in enumerate(mcp_resources):
                LOGGER.info("\nResource %s:", i)
                LOGGER.info("  Name: %s", resource.name if hasattr(resource, "name") else "No name")
                LOGGER.info(
                    "  Version: %s",
                    resource.version if hasattr(resource, "version") else "No version",
                )
                LOGGER.info(
                    "  Description: %s",
                    resource.description if hasattr(resource, "description") else "No description",
                )
                LOGGER.info("  Type: %s", type(resource))
                LOGGER.info("  Callable: %s", callable(resource))
                LOGGER.info(
                    "  Methods: %s",
                    [method for method in dir(resource) if not method.startswith("_")],
                )
                LOGGER.info("  Full resource: %s", resource.__dict__)

                # Test call
                try:
                    LOGGER.info("  Testing resource call...")
                    test_query = "John" if i == 0 else "What's the usage?"
                    result = await resource.ainvoke({"query": test_query})
                    LOGGER.info("  Test result: %s", result)
                except Exception as e:
                    LOGGER.info("  Test error: %s - %s", type(e).__name__, e)

            expected_resource_count = 2

            if len(mcp_resources) < expected_resource_count:
                msg = f"Expected {expected_resource_count} resources, got {len(mcp_resources)}"
                raise ValueError(msg)

            # Create async wrapper functions with better error handling
            async def calculator_greeting_wrapper(query: str) -> str | None:
                try:
                    resource = mcp_resources[0]  # calculator resource
                    if not resource:
                        LOGGER.info("Resource 0 (calculator) not properly initialized")
                        return "Error: Calculator resource not properly initialized"
                    LOGGER.info("Executing calculator with query: %s", query)
                    # Clean up the query
                    name = query.strip().replace("\n", " ").replace("  ", " ")
                    # Call the tool using the async method
                    result = await resource.ainvoke({"name": name})
                    LOGGER.info("Calculator result: %s", result)
                    if result:
                        return "Data added successfully"  # Clear success message
                    return "Failed to add data"  # Clear failure message
                except (TypeError, ValueError, AttributeError) as e:
                    LOGGER.info(
                        "Error in calculator_greeting_wrapper: %s - %s",
                        type(e).__name__,
                        e,
                    )
                    return f"Error adding data: {e}"

            async def get_usage_wrapper():
                try:
                    tool = mcp_resources[1]  # get_usage tool
                    if not tool:
                        LOGGER.info("Tool 1 (get_usage) not properly initialized")
                        return "Error: Get usage tool not properly initialized"
                    LOGGER.info("Executing get_usage")
                    # Call the tool using the async method
                    result = await tool.ainvoke()
                    LOGGER.info("Get usage result: %s", result)
                    if not result:
                        return "Error reading file: No content found"

                    return "\n".join(result)
                except (TypeError, ValueError, AttributeError) as e:
                    LOGGER.info("Error in read_data_wrapper: %s - %s", type(e).__name__, e)
                    return "Error reading data: %s" % e

            # Create Langchain tools with async functions
            self.tools = [
                Tool(
                    name="calculator_greeting",
                    description="Get a personalized greeting for the calculator",
                    func=lambda x: "Use async version",
                    coroutine=calculator_greeting_wrapper,
                ),
                Tool(
                    name="get_usage",
                    description="Get usage information from the file and answer questions.",
                    func=lambda x: "Use async version",
                    coroutine=get_usage_wrapper,
                ),
            ]

            LOGGER.info("Found %d tools", len(self.tools))

            # Create the prompt template with system message
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            human_message = HumanMessagePromptTemplate.from_template(TOOL_TEMPLATES)
            prompt = ChatPromptTemplate.from_messages([system_message, human_message]).partial(
                tool_names="calculator_greeting or get_usage",
            )

            # Create the agent with simpler configuration
            self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

            # Create the executor with better configuration
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=1,  # Only try once
                early_stopping_method="force",  # Stop after max_iterations
                return_intermediate_steps=True,  # Ensure we get the steps
            )

            LOGGER.info("\nAvailable tools:")
            for tool in self.tools:
                LOGGER.info("- %s: %s", tool.name, tool.description)

        except Exception as e:
            LOGGER.info("\nError initializing agent: %s", e)
            raise


async def main() -> None:
    """
    Start the Langchain MCP Client, initialize the agent, and.

    launch the interactive chat session.
    """
    try:
        LOGGER.info("Starting Langchain MCP Client...")
        client = LangchainMCPClient()

        LOGGER.info("\nInitializing agent...")
        await client.initialize_agent()

        LOGGER.info("\nStarting interactive chat...")
        await client.interactive_chat()

    except ConnectionError as e:
        LOGGER.info("\nConnection Error: %s", e)
        LOGGER.info("Please check that:")
        LOGGER.info("1. The MCP server is running (python server.py --server_type=sse)")
        LOGGER.info("2. The server URL is correct (http://127.0.0.1:8000)")
        LOGGER.info("3. The server is accessible from your machine")
    except (RuntimeError, ValueError, TypeError) as e:
        LOGGER.info("\nUnexpected error: %s - %s", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())

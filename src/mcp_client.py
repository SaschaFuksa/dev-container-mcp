"""Langchain MCP Client."""

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


REACT_TEMPLATE = """Answer the following questions as best you can. You have access to
the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: {tool_names}
Action Input: the SQL query to execute
Observation: the result of the action
Thought: I now know the final answer
Final Answer: [The formatted table for read_data or success message for add_data]

For example:
Question: add John Doe 30 year old Engineer
Thought: I need to add a new person to the database
Action: add_data
Action Input: INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')
Observation: Data added successfully
Thought: I have successfully added the person
Final Answer: Successfully added John Doe (age: 30, profession: Engineer) to the database

Question: show all records
Thought: I need to retrieve all records from the database
Action: read_data
Action Input: SELECT * FROM people
Observation: [Formatted table with records]
Thought: I have retrieved all records
Final Answer: [The formatted table showing all records]

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
        Initialize the LangchainMCPClient with the specified MCP server URL.

        Args:
            mcp_server_url (str): The URL of the MCP server to connect to.

        """
        logger.info("Initializing LangchainMCPClient...")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,  # Disable streaming for better compatibility
            base_url="http://host.docker.internal:11434",
        )
        # Updated server configuration with shorter timeouts
        server_config = {
            "default": {
                "url": f"{mcp_server_url}/sse",
                "transport": "sse",
            },
        }
        logger.info("Connecting to MCP server at %s...", mcp_server_url)
        self.mcp_client = MultiServerMCPClient(server_config)
        self.chat_history = []

        # Initialize agent and agent_executor attributes to None
        self.agent = None
        self.agent_executor = None

        # System prompt for the agent
        self.system_promt = """You are an AI assistant that helps users interact with a database.
        You can add and read data from the database using the available tools.
        When adding data:
        1. Format the SQL query correctly: INSERT INTO people (name, age, profession) VALUES
        ('Name', Age, 'Profession')
        2. Make sure to use single quotes around text values
        3. Don't use quotes around numeric values

        When reading data:
        1. Use SELECT * FROM people for all records
        2. Use WHERE clause for filtering: SELECT * FROM people WHERE condition
        3. Present results in a clear, formatted way

        Always:
        1. Think through each step carefully
        2. Verify actions were successful
        3. Provide clear summaries of what was done"""

    async def check_server_connection(self) -> bool | None:
        """Check if the MCP server is accessible."""
        base_url = self.mcp_client.connections["default"]["url"].replace("/sse", "")
        http_status_promt = 200
        try:
            logger.info("Testing connection to %s...", base_url)
            async with httpx.AsyncClient(timeout=5.0) as client:  # Shorter timeout
                # Try the SSE endpoint directly
                sse_url = f"{base_url}/sse"
                logger.info("Checking SSE endpoint at %s...", sse_url)
                response = await client.get(sse_url, timeout=5.0)
                logger.info("Got response: %s", response.status_code)
                if response.status_code == http_status_promt:
                    logger.info("SSE endpoint is accessible!")
                    return True

                logger.info("Server responded with status code: %s", response.status_code)
                return False

        except httpx.ConnectError:
            logger.info("Could not connect to server at %s", base_url)
            logger.info("Please ensure the server is running and the port is correct")
            return False
        except httpx.ReadTimeout:
            logger.info("Connection established but timed out while reading")
            logger.info("This is normal for SSE connections - proceeding...")
            return True
        except httpx.HTTPError as e:
            logger.info("HTTP error connecting to MCP server: %s - %s", type(e).__name__, e)
            return False
        except Exception as e:  # noqa: BLE001
            logger.info("Unexpected error connecting to MCP server: %s - %s", type(e).__name__, e)
            return False

    async def initialize_agent(self) -> None:
        """Initialize the agent with tools and prompt template"""
        logger.info("\nInitializing agent...")
        if not await self.check_server_connection():
            raise ConnectionError(
                "Cannot connect to MCP server. Please ensure the server is running.",
            )

        try:
            logger.info("Getting available tools...")
            mcp_tools = await self.mcp_client.get_tools()

            # Verify tools are properly initialized
            logger.info("Verifying tools...")
            for i, tool in enumerate(mcp_tools):
                logger.info("\nTool %s:", i)
                logger.info("  Name: %s", tool.name if hasattr(tool, "name") else "No name")
                logger.info(
                    "  Version: %s",
                    tool.version if hasattr(tool, "version") else "No version",
                )
                logger.info(
                    "  Description: %s",
                    tool.description if hasattr(tool, "description") else "No description",
                )
                logger.info("  Type: %s", type(tool))
                logger.info("  Callable: %s", callable(tool))
                logger.info(
                    "  Methods: %s",
                    [method for method in dir(tool) if not method.startswith("_")],
                )
                logger.info("  Full tool: %s", tool.__dict__)

                # Test call
                try:
                    logger.info("  Testing tool call...")
                    if i == 0:
                        test_query = (
                            "INSERT INTO people (name, age, profession) VALUES ('Test', 30, 'Test')"
                        )
                    else:
                        test_query = "SELECT * FROM people"
                    result = await tool.ainvoke({"query": test_query})
                    logger.info("  Test result: %s", result)
                except Exception as e:
                    logger.info("  Test error: %s - %s", type(e).__name__, e)

            if len(mcp_tools) < 2:
                raise ValueError("Expected 2 tools, got %d", len(mcp_tools))

            # Create async wrapper functions with better error handling
            async def add_data_wrapper(query: str):
                try:
                    tool = mcp_tools[0]  # add_data tool
                    if not tool:
                        logger.info("Tool 0 (add_data) not properly initialized")
                        return "Error: Add data tool not properly initialized"
                    logger.info("Executing add_data with query: %s", query)
                    # Clean up the query
                    query = query.strip().replace("\n", " ").replace("  ", " ")
                    # Fix common formatting issues
                    if "VALUES" in query:
                        parts = query.split("VALUES")
                        if len(parts) == 2:
                            values = parts[1].strip()
                            if values.startswith("(") and values.endswith(")"):
                                values = values[1:-1].split(",")
                                if len(values) == 3:
                                    name = values[0].strip().strip("'")
                                    age = values[1].strip()
                                    profession = values[2].strip().strip("'")
                                    query = "INSERT INTO people (name, age, profession) VALUES (%s, %s, %s)"
                                    params = (name, age, profession)
                    # Call the tool using the async method
                    result = await tool.ainvoke({"query": query})
                    logger.info("Add data result: %s", result)
                    if result:
                        return "Data added successfully"  # Clear success message
                    return "Failed to add data"  # Clear failure message
                except Exception as e:
                    logger.info("Error in add_data_wrapper: %s - %s", type(e).__name__, e)
                    return "Error adding data: %s" % e

            async def read_data_wrapper(query: str = "SELECT * FROM people"):
                try:
                    tool = mcp_tools[1]  # read_data tool
                    if not tool:
                        logger.info("Tool 1 (read_data) not properly initialized")
                        return "Error: Read data tool not properly initialized"
                    logger.info("Executing read_data with query: %s", query)
                    # Clean up the query
                    query = query.strip().replace("\n", " ").replace("  ", " ")
                    # Call the tool using the async method
                    result = await tool.ainvoke({"query": query})
                    logger.info("Read data result: %s", result)
                    if not result:
                        return "No records found"

                    # Format results in a table
                    records = []
                    for i in range(0, len(result), 4):
                        records.append(
                            {
                                "name": result[i + 1],
                                "age": result[i + 2],
                                "profession": result[i + 3],
                            },
                        )

                    # Create table header
                    output = [
                        f"Showing {len(records)} records:",
                        "",
                        "| Name          | Age | Profession       |",
                        "|---------------|-----|------------------|",
                    ]

                    # Add each record
                    for record in records:
                        name = record["name"].ljust(13)
                        age = str(record["age"]).ljust(5)
                        profession = record["profession"].ljust(16)
                        output.append(f"| {name} | {age} | {profession} |")

                    return "\n".join(output)
                except Exception as e:
                    logger.info("Error in read_data_wrapper: %s - %s", type(e).__name__, e)
                    return "Error reading data: %s" % e

            # Create Langchain tools with async functions
            self.tools = [
                Tool(
                    name="add_data",
                    description="Add a person to the database. Example: INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer'). If there are information missing, insert 99 for age and 'Pizzabär' for profession",
                    func=lambda x: "Use async version",
                    coroutine=add_data_wrapper,
                ),
                Tool(
                    name="read_data",
                    description="Read from the database. Example: SELECT * FROM people",
                    func=lambda x: "Use async version",
                    coroutine=read_data_wrapper,
                ),
            ]

            logger.info(f"Found {len(self.tools)} tools")

            # Create the prompt template with system message
            system_message = SystemMessage(content=self.system_promt)
            human_message = HumanMessagePromptTemplate.from_template(REACT_TEMPLATE)
            prompt = ChatPromptTemplate.from_messages([system_message, human_message]).partial(
                tool_names="add_data or read_data",
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

            logger.info("\nAvailable tools:")
            for tool in self.tools:
                logger.info("- %s: %s", tool.name, tool.description)

        except Exception as e:
            logger.info("\nError initializing agent: %s", e)
            raise

    async def process_message(self, user_input: str) -> str:
        """
        Process a user input message, invoke the agent, and return the agent's response.

        Args:
            user_input (str): The user's input message.

        Returns:
            str: The agent's response to the input message.

        """
        logger.info("\nProcessing message: %s", user_input)
        try:
            logger.info("\nProcessing message: %s", user_input)
            # Execute the agent
            response = await self.agent_executor.ainvoke(
                {"input": user_input, "chat_history": self.chat_history},
            )

            logger.info("\nRaw response: %s", response)
            final_result = None

            # Get the result from intermediate steps
            if isinstance(response, dict) and "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                if steps and isinstance(steps[-1], tuple):
                    action, observation = steps[-1]

                    # Handle add_data response
                    if "add_data" in str(action):
                        query = str(action.tool_input)
                        if "VALUES" in query:
                            values = query[query.find("VALUES") + 7 :].strip("() ")
                            name, age, profession = [
                                v.strip().strip("'") for v in values.split(",")
                            ]
                            final_result = (
                                f"Successfully added {name} (age: {age}, profession: {profession}) "
                                "to the database"
                            )

                    # Handle read_data response
                    elif "read_data" in str(action):
                        if isinstance(observation, str) and "Showing" in observation:
                            final_result = observation  # Use the formatted table
                        else:
                            final_result = str(observation)  # Use any other read response

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

                    logger.info("\nFinal result: %s", final_result)
                    return final_result

            return "Could not process the request. Please try again."

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            error_msg = (
                f"Error processing message: {type(e).__name__} - {e}\n"
                "Please try rephrasing your request."
            )
            logger.info("\nError processing message: %s - %s", type(e).__name__, e)
            logger.info("Full error: %s", getattr(e, "__dict__", {}))
            return error_msg

    async def interactive_chat(self) -> None:
        """Start an interactive chat session."""
        logger.info("Chat session started. Type 'exit' to quit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                logger.info("Ending chat session...")
                break

            response = await self.process_message(user_input)
            logger.info("\nAgent: %s", response)


async def main() -> None:
    """
    Start the Langchain MCP Client, initialize the agent, and.

    launch the interactive chat session.
    """
    try:
        logger.info("Starting Langchain MCP Client...")
        client = LangchainMCPClient()

        logger.info("\nInitializing agent...")
        await client.initialize_agent()

        logger.info("\nStarting interactive chat...")
        await client.interactive_chat()

    except ConnectionError as e:
        logger.info("\nConnection Error: %s", e)
        logger.info("Please check that:")
        logger.info("1. The MCP server is running (python server.py --server_type=sse)")
        logger.info("2. The server URL is correct (http://127.0.0.1:8000)")
        logger.info("3. The server is accessible from your machine")
    except (RuntimeError, ValueError, TypeError) as e:
        logger.info("\nUnexpected error: %s - %s", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())

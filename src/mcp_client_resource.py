"""
Langchain MCP Resource Client

Verwendet MCP-Ressourcen direkt als Kontext für den LLM Agent, ohne künstliche Tools.
"""

import asyncio
import logging
import sys

from fastmcp.client import Client
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama import ChatOllama

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

PROMPT_TEMPLATE = """Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: {tool_names}
Action Input: Input is described by the tool you want to use
Observation: detect the needed information from of the resource
Thought: I read the result and now know the final answer
Final Answer: [The formatted table for read_data or success message for add_data]

For example:
Question: Which roles our user have?
Thought: I need to detect the roles from the users
Action: get_user_data
Action Input: None
Observation: User Data content with roles
Thought: I have successfully detected the roles
Final Answer: Roles: desginer, developer

Begin!

Question: {input}
{agent_scratchpad}"""


SYSTEM_PROMPT = """You are an AI assistant that helps users with informations.
        You can aread data from the server using the available tools.

        Always:
        1. Think through each step carefully
        2. Verify actions were successful
        3. Provide clear summaries of what was done"""


class LangchainMCPResourceClient:
    """Langchain MCP Resource Client"""

    def __init__(self):
        """Initialisiere den MCP Resource Client."""
        self.llm: ChatOllama = None
        self.mcp_client: Client = None
        self.agent_executor: AgentExecutor = None

    async def initialize_llm(self) -> None:
        """Initialisiere die MCP-Client-Verbindung."""
        LOGGER.info("Initialisiere LLM...")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,
            base_url="http://host.docker.internal:11434",
        )
        LOGGER.info("LLM erfolgreich initialisiert")

    async def initialize_mcp_client(self) -> None:
        """Initialisiere die MCP-Client-Verbindung."""
        LOGGER.info("Initialisiere MCP-Client...")
        url = "http://127.0.0.1:8000/sse"
        self.mcp_client = Client(url)
        self.chat_history = []
        LOGGER.info("MCP-Client erfolgreich initialisiert")

    async def check_server_connection(self) -> bool:
        """Prüfe die Verbindung zum MCP-Server."""
        async with self.mcp_client:
            return await self.mcp_client.ping()

    async def initialize_agent(self) -> None:
        """Initialisiere den Agent mit dem Ressourcen-Kontext."""
        LOGGER.info("Initialisiere Agent...")
        async with self.mcp_client:
            tools = await self.mcp_client.list_tools()
            for i, tool in enumerate(tools):
                LOGGER.info(f"\nTool {i}:")
                LOGGER.info(f"  Name: {tool.name}")
                LOGGER.info(f"  Description: {tool.description}")

            if len(tools) < 3:
                raise ValueError(f"Expected 3 tools, got {len(tools)}")
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            human_message = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
            tool_names = [tool.name for tool in tools]
            prompt = ChatPromptTemplate.from_messages([system_message, human_message]).partial(
                tools=tool_names,
            )
            self.tools = []
            for tool in tools:
                self.tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description,
                        func=lambda x: "Use async version",
                        # coroutine=tool.coroutine,
                    ),
                )
            self.agent = create_react_agent(
                prompt=prompt,
                llm=self.llm,
                tools=self.tools,
            )
            LOGGER.info("Agent erfolgreich initialisiert")
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=2,  # Only try once
                early_stopping_method="force",  # Stop after max_iterations
                return_intermediate_steps=True,  # Ensure we get the steps
            )
            LOGGER.info("Agent Executor erfolgreich initialisiert")

    async def chat(self, user_input: str) -> str:
        """Verarbeite eine Benutzereingabe und gib eine Antwort zurück."""
        try:
            response = await self.agent_executor.ainvoke(
                {
                    "input": user_input,
                    "chat_history": self.chat_history,
                },
            )
            LOGGER.info(f"Agent Output: {response}")
            return response.get("output", "Keine Antwort erhalten")
        except Exception as e:
            LOGGER.error("Fehler bei der Chat-Verarbeitung: %s", e)
            return f"Entschuldigung, es gab einen Fehler: {e}"

    async def run_interactive_chat(self) -> None:
        """Starte eine interaktive Chat-Session."""
        LOGGER.info("\n🤖 MCP Resource Chat gestartet!")
        LOGGER.info("💡 Verfügbare Ressourcen wurden geladen und stehen als Kontext zur Verfügung.")
        LOGGER.info(
            "❓ Du kannst Fragen zu den Projektdaten, Benutzerdaten oder dem Gesprächskontext stellen.",
        )
        LOGGER.info("🚪 Schreibe 'quit' oder 'exit' zum Beenden.\n")

        while True:
            try:
                user_input = input("\n🙋 Du: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    LOGGER.info("\n👋 Auf Wiedersehen!")
                    break

                if not user_input:
                    continue

                LOGGER.info("\n🤔 Denke nach...")
                response = await self.chat(user_input)
                LOGGER.info(f"\n🤖 Assistent: {response}")

            except KeyboardInterrupt:
                LOGGER.info("\n\n👋 Session beendet.")
                break
            except Exception as e:
                LOGGER.info(f"\n❌ Fehler: {e}")


async def main() -> None:
    """Start the Langchain MCP Resource Client."""
    client = None
    try:
        # Erstelle und initialisiere Client
        client = LangchainMCPResourceClient()

        await client.initialize_llm()

        # Initialisiere MCP-Verbindung
        await client.initialize_mcp_client()

        if not await client.check_server_connection():
            raise ConnectionError(
                "Cannot connect to MCP server. Please ensure the server is running.",
            )

        # Initialisiere Agent
        await client.initialize_agent()

        # Starte interaktive Chat-Session
        await client.run_interactive_chat()

    except Exception as e:
        LOGGER.info("Fehler beim Starten des Clients: %s", e)
        LOGGER.info(f"❌ Fehler: {e}")
    finally:
        # Bereinige Verbindungen
        LOGGER.info("Ending chat session...")


if __name__ == "__main__":
    asyncio.run(main())

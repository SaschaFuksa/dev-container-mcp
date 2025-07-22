"""
Langchain MCP Resource Client
Verwendet MCP-Ressourcen direkt als Kontext für den LLM Agent, ohne künstliche Tools.
"""

import asyncio
import logging

import httpx
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

# Konfiguriere Logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LangchainMCPResourceClient:
    """
    Ein LangChain-basierter MCP Resource Client, der Ressourcen direkt
    als Kontext für den Agent verwendet.
    """

    def __init__(self, model_name: str = "llama3.1:8b"):
        """Initialisiere den MCP Resource Client."""
        self.logger = LOGGER
        self.model_name = model_name
        self.llm = None
        self.mcp_client: MultiServerMCPClient | None = None
        self.agent_executor: AgentExecutor | None = None
        self.resource_context = ""

    async def initialize_mcp_client(self) -> None:
        """Initialisiere die MCP-Client-Verbindung."""
        try:
            self.logger.info("Initialisiere MCP-Client...")

            # Konfiguration für MultiServerMCPClient
            server_config = {
                "default": {
                    "url": "http://127.0.0.1:8000/sse",
                    "transport": "sse",
                },
            }

            self.mcp_client = MultiServerMCPClient(server_config)
            self.logger.info("MCP-Client erfolgreich initialisiert")
        except Exception as e:
            self.logger.error("Fehler beim Initialisieren des MCP-Clients: %s", e)
            raise

    async def check_server_connection(self) -> bool:
        """Prüfe die Verbindung zum MCP-Server."""
        try:
            async with httpx.AsyncClient() as client:
                await client.get("http://127.0.0.1:8000", timeout=5.0)
                return True  # Server antwortet, auch wenn 404
        except Exception as e:
            self.logger.error("Server-Verbindung fehlgeschlagen: %s", e)
            return False

    async def load_resource_context(self) -> None:
        """Lade alle MCP-Ressourcen als Kontext für den Agent."""
        try:
            # Hole verfügbare Ressourcen über MultiServerMCPClient
            resources = await self.mcp_client.get_resources("default")

            self.logger.info("Verfügbare MCP-Ressourcen: %s", [blob.source for blob in resources])

            # Sammle alle Ressourceninhalte
            context_parts = ["=== VERFÜGBARE MCP-RESSOURCEN ===\n"]

            for blob in resources:
                try:
                    context_parts.append(f"RESSOURCE: {blob.source}")
                    context_parts.append(
                        f"BESCHREIBUNG: {blob.metadata.get('description', 'Keine Beschreibung')}",
                    )
                    context_parts.append(f"INHALT:\n{blob.as_string()}\n")
                    context_parts.append("-" * 50)
                except Exception as e:
                    context_parts.append(f"FEHLER bei {blob.source}: {e}\n")

            self.resource_context = "\n".join(context_parts)
            self.logger.info("Ressourcen-Kontext geladen (%d Zeichen)", len(self.resource_context))

        except Exception as e:
            self.logger.error("Fehler beim Laden der Ressourcen: %s", e)
            self.resource_context = "Keine MCP-Ressourcen verfügbar"

    async def initialize_agent(self) -> None:
        """Initialisiere den Agent mit dem Ressourcen-Kontext."""
        self.logger.info("Initialisiere Agent...")

        # Prüfe Server-Verbindung
        if not await self.check_server_connection():
            raise ConnectionError(
                "Kann nicht zum MCP-Server verbinden. Stelle sicher, dass der Server läuft.",
            )

        # Lade Ressourcen-Kontext
        await self.load_resource_context()

        # Initialisiere LLM
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,  # Disable streaming for better compatibility
            base_url="http://host.docker.internal:11434",
        )

        # Erstelle Prompt-Template mit Ressourcen-Kontext
        template = """Du bist ein hilfreicher Assistent mit Zugriff auf MCP-Ressourcen.

VERFÜGBARE RESSOURCEN:
{resource_context}

Du kannst die obigen Ressourcen nutzen, um Fragen zu beantworten. Du hast Zugriff auf:
- README-Informationen des Projekts
- Benutzerdaten 
- Gesprächskontext

WERKZEUGE:
Du hast Zugriff auf die folgenden Werkzeuge:

{tools}

Verwende das folgende Format:

Frage: Die Eingabefrage, die du beantworten musst
Gedanke: Du solltest immer über das nachdenken, was zu tun ist
Aktion: Die auszuführende Aktion, sollte eine von [{tool_names}] sein
Aktions-Input: Die Eingabe für die Aktion
Beobachtung: Das Ergebnis der Aktion
... (dieser Gedanke/Aktion/Aktions-Input/Beobachtung kann N-mal wiederholt werden)
Gedanke: Ich kenne jetzt die endgültige Antwort
Endgültige Antwort: Die endgültige Antwort auf die ursprüngliche Eingabefrage

Beginne!

Frage: {input}
Gedanke: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        # Setze den Ressourcen-Kontext
        prompt = prompt.partial(resource_context=self.resource_context)

        # Erstelle Agent
        agent = create_react_agent(self.llm, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

        self.logger.info("Agent erfolgreich initialisiert")

    async def chat(self, user_input: str) -> str:
        """Verarbeite eine Benutzereingabe und gib eine Antwort zurück."""
        try:
            result = await self.agent_executor.ainvoke({"input": user_input})
            return result.get("output", "Keine Antwort erhalten")
        except Exception as e:
            self.logger.error("Fehler bei der Chat-Verarbeitung: %s", e)
            return f"Entschuldigung, es gab einen Fehler: {e}"

    async def cleanup(self) -> None:
        """Bereinige die MCP-Client-Verbindung."""
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__
            except Exception as e:
                self.logger.error("Fehler beim Schließen der MCP-Verbindung: %s", e)

    async def run_interactive_chat(self) -> None:
        """Starte eine interaktive Chat-Session."""
        print("\n🤖 MCP Resource Chat gestartet!")
        print("💡 Verfügbare Ressourcen wurden geladen und stehen als Kontext zur Verfügung.")
        print(
            "❓ Du kannst Fragen zu den Projektdaten, Benutzerdaten oder dem Gesprächskontext stellen.",
        )
        print("🚪 Schreibe 'quit' oder 'exit' zum Beenden.\n")

        while True:
            try:
                user_input = input("\n🙋 Du: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Auf Wiedersehen!")
                    break

                if not user_input:
                    continue

                print("\n🤔 Denke nach...")
                response = await self.chat(user_input)
                print(f"\n🤖 Assistent: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Session beendet.")
                break
            except Exception as e:
                print(f"\n❌ Fehler: {e}")


async def main() -> None:
    """Start the Langchain MCP Resource Client."""
    client = None
    try:
        # Erstelle und initialisiere Client
        client = LangchainMCPResourceClient()

        # Initialisiere MCP-Verbindung
        await client.initialize_mcp_client()

        # Initialisiere Agent
        await client.initialize_agent()

        # Starte interaktive Chat-Session
        await client.run_interactive_chat()

    except Exception as e:
        LOGGER.info("Fehler beim Starten des Clients: %s", e)
        LOGGER.info(f"❌ Fehler: {e}")
    finally:
        # Bereinige Verbindungen
        if client:
            await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

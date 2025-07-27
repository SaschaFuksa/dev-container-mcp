"""
MCP Client Example.

This example demonstrates how to set up a simple MCP client that interacts with a math MCP server.
"""

import asyncio
import logging
import sys
from typing import Annotated

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mcp.client.session import ClientSession
from typing_extensions import TypedDict

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["src/math_mcp_server.py"],
            "transport": "stdio",
        },
        "bmi": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    },
)


async def create_graph(math_session: ClientSession, bmi_session: ClientSession) -> StateGraph:
    """
    Create and compile a StateGraph for the MCP client using the provided session.

    Returns
    -------
    StateGraph
        The compiled StateGraph ready for agent execution.

    """
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.6,
        streaming=False,
        base_url="http://host.docker.internal:11434",
    )

    math_tools = await load_mcp_tools(math_session)
    bmi_tools = await load_mcp_tools(bmi_session)
    tools = math_tools + bmi_tools
    llm_with_tool = llm.bind_tools(tools)

    system_prompt = await load_mcp_prompt(math_session, "system_prompt")
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt[0].content), MessagesPlaceholder("messages")],
    )
    chat_llm = prompt_template | llm_with_tool

    # State Management
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # Nodes
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Building the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {"tools": "tool_node", "__end__": END},
    )
    graph_builder.add_edge("tool_node", "chat_node")
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph


async def main() -> None:
    """
    Initialize the client session for the MCP client example.

    Initializes the client session, checks available tools, prompts, and resources,
    and runs an interactive loop to communicate with the MCP server.
    """
    config = {"configurable": {"thread_id": 1234}}
    async with client.session("math") as math_session, client.session("bmi") as bmi_session:
        agent = await create_graph(math_session, bmi_session)
        while True:
            message = input("User: ")
            response = await agent.ainvoke({"messages": message}, config=config)
            LOGGER.info("AI: %s", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

"""Example of a FastMCP server resource."""

import argparse
import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("resource-demo")


@mcp.tool(name="get_readme", title="README", description="Returns the Project README content")
async def get_readme() -> str:
    """Project README content."""
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "# Sample Project\nThis is a sample README file for MCP testing."


@mcp.tool(
    name="get_user_data",
    title="User Data",
    description="Returns user profiles and preferences.",
)
async def get_user_data() -> str:
    """User profiles and preferences."""
    data = {
        "users": [
            {"name": "Alice", "role": "developer", "skills": ["Python", "JavaScript"]},
            {"name": "Bob", "role": "designer", "skills": ["Figma", "Photoshop"]},
        ],
        "preferences": {"theme": "dark", "language": "en"},
    }
    return json.dumps(data, indent=2)


@mcp.tool(
    name="get_conversation_context",
    title="Conversation Context",
    description="Returns the current conversation context and history.",
)
async def get_conversation_context() -> str:
    """Return the current conversation context and history."""
    context = {
        "current_topic": "MCP implementation",
        "user_goal": "Learn FastMCP with Ollama",
        "tech_stack": ["Python", "Ollama", "VSCode", "Dev-Container"],
        "previous_questions": ["How to use @mcp.resource?", "Integration with LLM?"],
    }
    return json.dumps(context, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type",
        type=str,
        default="sse",
        choices=["sse", "stdio"],
    )
    logger.info("🚀Starting server... ")
    args = parser.parse_args()
    # python src/mcp_server_resource.py --server_type=sse
    mcp.run(args.server_type)

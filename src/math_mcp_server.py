"""
MCP Server Example.

This example demonstrates how to set up a simple MCP server with tools and resources.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

mcp = FastMCP("Math")


# Prompts
@mcp.prompt()
def example_prompt(question: str) -> str:
    """Return an example prompt description."""
    LOGGER.info("🚀Example prompt called.")
    return f"""
    You are a math assistant. Answer the question.
    Question: {question}
    """


@mcp.prompt()
def system_prompt() -> str:
    """System prompt description."""
    LOGGER.info("🚀System prompt called.")
    return """
    /no_think You are an AI assistant use the tools if needed.
    """


# Resources
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    LOGGER.info("🚀Fetching greeting for %s.", name)
    return f"Hello, {name}!"


@mcp.resource("config://app")
def get_config() -> str:
    """Return static configuration data."""
    LOGGER.info("🚀Fetching app configuration.")
    return "App configuration here"


# Tools
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    LOGGER.info("🚀Calc Add.")
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    LOGGER.info("🚀Calc Multiply.")
    return a * b


if __name__ == "__main__":
    # mcp.run(transport="streamable-http")  # Run server via streamable-http  # noqa: ERA001
    LOGGER.info("🚀Starting server... ")
    # mcp.run(transport="sse")  # noqa: ERA001
    mcp.run()

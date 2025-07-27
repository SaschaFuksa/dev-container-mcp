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

mcp = FastMCP("BMI")


# Tools
@mcp.tool()
def calculate_bmi(weight: int, height: int) -> str:
    """Calculate BMI."""
    return "BMI: " + str(weight / (height * height))


if __name__ == "__main__":
    # mcp.run(transport="streamable-http")  # noqa: ERA001
    mcp.run()

"""Example of a FastMCP server resource."""

import argparse
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("resource-demo")


@mcp.resource("calculator://greet/{name}")
def calculator_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}! Ready to calculate something today?"


@mcp.resource("usage://guide")
def get_usage() -> str:
    """Get usage instructions for the calculator resource."""
    with Path("docs/usage.txt").open(encoding="utf-8") as f:
        return f.read()


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
    # python mcp_server_resource.py --server_type=sse
    mcp.run(args.server_type)

"""
MCP Client Example.

This example demonstrates how to set up a simple MCP client that interacts with a math MCP server.
"""

import asyncio
import logging
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

# Math Server Parameters
server_params = StdioServerParameters(
    command="python",
    args=["src/math_mcp_server.py"],
    env=None,
)


async def main() -> None:
    """Run the MCP client."""
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        # List available prompts
        response = await session.list_prompts()
        LOGGER.info("\n/////////////////prompts//////////////////")
        for prompt in response.prompts:
            LOGGER.info(prompt)

        # List available resources
        response = await session.list_resources()
        LOGGER.info("\n/////////////////resources//////////////////")
        for resource in response.resources:
            LOGGER.info(resource)

        # List available resource templates
        response = await session.list_resource_templates()
        LOGGER.info("\n/////////////////resource_templates//////////////////")
        for resource_template in response.resourceTemplates:
            LOGGER.info(resource_template)

        # List available tools
        response = await session.list_tools()
        LOGGER.info("\n/////////////////tools//////////////////")
        for tool in response.tools:
            LOGGER.info(tool)

        # Get a prompt
        prompt = await session.get_prompt(
            "example_prompt",
            arguments={"question": "what is 2+2"},
        )
        LOGGER.info("\n/////////////////prompt//////////////////")
        LOGGER.info(prompt.messages[0].content.text)

        # Read a resource
        content, mime_type = await session.read_resource("greeting://Sascha")
        LOGGER.info("\n/////////////////content//////////////////")
        LOGGER.info(content)
        LOGGER.info(mime_type[1][0].text)

        # Call a tool
        result = await session.call_tool("add", arguments={"a": 2, "b": 2})
        LOGGER.info("\n/////////////////result//////////////////")
        LOGGER.info(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

"""MCP server for AgenticCyberSense tools."""

import functions.age as age_module
import functions.lastname_retrieval as lastname_module
import functions.reverse_name as reverse_name_module
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dummy-mcp-server")


@mcp.tool()
def lastname(last: str) -> str:
    """Return lastname tool."""
    return lastname_module.lastname(last)


@mcp.tool()
def age(age: int) -> int:
    """Return age tool."""
    return age_module.age(age)


@mcp.tool()
def reverse_name(name: str) -> str:
    """Return reversed name tool."""
    return reverse_name_module.reverse_name(name)
    """Returns lastname tool."""
    return None


if __name__ == "__main__":
    # Runs as an MCP stdio server (what LangChain MCP client expects)
    mcp.run()

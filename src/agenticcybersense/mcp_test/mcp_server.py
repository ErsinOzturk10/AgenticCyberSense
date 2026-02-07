"""MCP server for AgenticCyberSense tools."""

from agenticcybersense.mcp_test.functions import age as age_module
from agenticcybersense.mcp_test.functions import lastname_retrieval as lastname_module
from agenticcybersense.mcp_test.functions import reverse_name as reverse_name_module
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


if __name__ == "__main__":
    # Runs as an MCP stdio server (what LangChain MCP client expects)
    mcp.run()
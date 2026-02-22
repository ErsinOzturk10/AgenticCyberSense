"""MCP Server Tools."""

from mcp.server.fastmcp import FastMCP

# Assuming these modules exist in your project structure
from agenticcybersense.mcp.functions import age as age_module
from agenticcybersense.mcp.functions import lastname_retrieval as lastname_module
from agenticcybersense.mcp.functions import reverse_name as reverse_name_module

# Initialize FastMCP server
mcp = FastMCP("dummy-mcp-tools-server")


@mcp.tool()
def get_lastname(name: str) -> str:
    """Retrieve the last name for a given first name."""
    return lastname_module.lastname(name)


@mcp.tool()
def process_age(user_age: int) -> int:
    """Process or record the user's age."""
    return age_module.age(user_age)


@mcp.tool()
def reverse_name(name: str) -> str:
    """Reverse the characters of a given name string."""
    return reverse_name_module.reverse_name(name)

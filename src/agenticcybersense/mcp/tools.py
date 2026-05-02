"""MCP Server Tools."""

from mcp.server.fastmcp import FastMCP

from agenticcybersense.agents.telegram.telegram import telegram_search as _telegram_search

# Assuming these modules exist in your project structure
from agenticcybersense.rag.rag import rag_search as _rag_search

# Initialize FastMCP server
mcp = FastMCP("dummy-mcp-tools-server")


@mcp.tool()
def rag_search(user_input: str) -> str:
    """Mandatory first tool for every user request.

    This tool provides information from uploaded PDFs about Cyber Threat Intelligence.

    Tool orchestration rules:
    - rag_search must be the first tool call for every user request.
    - After rag_search returns, use its output as context for deciding what to do next.
    - After rag_search, decide whether any additional tool is required.
    - Do not produce the final answer until all necessary tool calls are completed.

    If relevant information is not found in the PDFs, return:
    "Insufficient information in the provided documents."
    """
    return _rag_search(query=user_input)


@mcp.tool()
async def telegram_search(user_input: str) -> str:
    """Search Telegram threat intelligence channels for cyber threats.

    Use this tool after rag_search when the user asks about:
    - Recent cyber attacks or threat actor activity on Telegram
    - CVEs or vulnerabilities mentioned in Telegram channels
    - Leaked data, breach news, malware campaigns, APT activity

    If no relevant data is found, return:
    "No relevant Telegram intelligence found."
    """
    return await _telegram_search(query=user_input)

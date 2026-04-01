"""MCP Server Tools."""

from mcp.server.fastmcp import FastMCP

# Assuming these modules exist in your project structure
from agenticcybersense.mcp.functions import age as age_module
from agenticcybersense.mcp.functions import lastname_retrieval as lastname_module
from agenticcybersense.mcp.functions import reverse_name as reverse_name_module
from agenticcybersense.rag.rag import rag_search as _rag_search
from agenticcybersense.agents.telegram import TelegramAgent
from agenticcybersense.schemas.messages import AgentRequest

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


@mcp.tool()
def rag_search(user_input: str) -> str:
    """Retrieve information ONLY from the uploaded Cyber Threat Intelligence PDFs.

    Use this tool when the user asks about:
    - Cyber Threat Intelligence (definition, characteristics, benefits)
    - Adversaries (cybercriminals, hacktivists, espionage actors)
    - Threat indicators, threat data feeds
    - Intelligence lifecycle (collection, analysis, dissemination)
    - Tactical, operational, strategic intelligence usage
    - Incident response, SOC, SIEM context
    - Intelligence program implementation
    - Intelligence partner selection
    - Risk prioritization, assets, threat actors

    DO NOT use this tool for:
    - General knowledge questions (e.g., weather, geography, politics)
    - Real-time information
    - Topics unrelated to cybersecurity or cyber threat intelligence

    If relevant information is not found in the PDFs, return:
    "Insufficient information in the provided documents."
    """
    return _rag_search(query=user_input)

@mcp.tool()
async def telegram_search(user_input: str) -> str:
    """Search Telegram threat intelligence channels for cyber threats.

    Use this tool when the user asks about:
    - Recent cyber attacks or threat actor activity on Telegram
    - CVEs or vulnerabilities mentioned in Telegram channels
    - Leaked data, breach news, malware campaigns, APT activity

    DO NOT use this tool for:
    - General knowledge questions
    - Topics unrelated to cybersecurity

    If no relevant data is found, return:
    "No relevant Telegram intelligence found."
    """

    agent = TelegramAgent()
    response = await agent.process(AgentRequest(query=user_input))
    return response.content

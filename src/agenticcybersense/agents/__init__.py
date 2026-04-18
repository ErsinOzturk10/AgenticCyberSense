"""Agent definitions for AgenticCyberSense."""

from agenticcybersense.agents.base import BaseAgent

# Import agents to trigger registration
from agenticcybersense.agents.documentation import DocumentationAgent
from agenticcybersense.agents.orchestrator import OrchestratorAgent
from agenticcybersense.agents.registry import AgentRegistry, get_registry, register_agent
from agenticcybersense.agents.web import WebAgent

try:
    from agenticcybersense.agents.telegram import TelegramAgent
except ImportError:  # telethon is optional
    TelegramAgent = None  # type: ignore[assignment,misc]

__all__ = [
    "AgentRegistry",
    "BaseAgent",
    "DocumentationAgent",
    "OrchestratorAgent",
    "TelegramAgent",
    "WebAgent",
    "get_registry",
    "register_agent",
]

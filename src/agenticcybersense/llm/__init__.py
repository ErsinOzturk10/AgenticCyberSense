"""LLM integration module."""

from agenticcybersense.llm.factory import create_llm, get_llm
from agenticcybersense.llm.prompts import PromptTemplates

__all__ = ["PromptTemplates", "create_llm", "get_llm"]

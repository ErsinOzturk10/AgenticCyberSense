"""Central LLM factory for AgenticCyberSense.

All runtime LLM selection must go through this module.
The provider is controlled only by settings.llm_provider / LLM_PROVIDER.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agenticcybersense.settings import settings

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_chat_llm(temperature: float | None = None) -> BaseChatModel:
    """Build the configured chat LLM.

    Provider is selected only from settings.llm_provider.
    No terminal prompt or runtime user selection is allowed here.
    """
    settings.validate_llm_settings()

    provider = settings.normalized_llm_provider()
    selected_temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "openai":
        logger.info("Using OpenAI model: %s", settings.openai_model)

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=selected_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "ollama":
        logger.info("Using Ollama model: %s", settings.ollama_model)

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=selected_temperature,
        )

    msg = f"Unsupported LLM_PROVIDER: {settings.llm_provider}"
    raise ValueError(msg)


def generate_text(prompt: str, temperature: float | None = None) -> str:
    """Generate plain text from the configured LLM.

    Useful for modules that currently expect a simple string response.
    """
    llm = build_chat_llm(temperature=temperature)
    response = llm.invoke(prompt)

    content: Any = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()

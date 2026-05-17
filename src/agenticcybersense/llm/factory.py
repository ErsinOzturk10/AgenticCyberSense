"""LLM factory for creating language model instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama

from agenticcybersense.logging_utils import get_logger
from agenticcybersense.settings import settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger("llm.factory")

# Global LLM instance cache
_llm_instance: BaseChatModel | None = None


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> BaseChatModel:
    """Create a new LLM instance.

    Args:
        provider: LLM provider ("ollama", "openai", etc.). Defaults to settings.
        model: Model name. Defaults to settings.
        **kwargs: Additional arguments passed to the LLM constructor.

    Returns:
        A LangChain chat model instance.

    """
    provider = provider or settings.llm_provider
    logger.info("Creating LLM with provider: %s", provider)

    if provider == "ollama":
        model = model or settings.ollama_model
        temperature_value = kwargs.pop("temperature", 0.7)
        temperature = float(temperature_value) if temperature_value is not None else None
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    if provider == "openai":
        msg = "OpenAI provider not yet implemented"
        raise NotImplementedError(msg)

    msg = f"Unknown LLM provider: {provider}"
    raise ValueError(msg)


def get_llm(**kwargs: Any) -> BaseChatModel:  # noqa: ANN401
    """Get or create the global LLM instance.

    Args:
        **kwargs: Arguments passed to create_llm if creating new instance.

    Returns:
        The global LLM instance.

    """
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is None:
        _llm_instance = create_llm(**kwargs)
    return _llm_instance


def reset_llm() -> None:
    """Reset the global LLM instance."""
    global _llm_instance  # noqa: PLW0603
    _llm_instance = None

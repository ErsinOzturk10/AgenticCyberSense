"""LLM factory for creating language model instances."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
        provider: LLM provider ("ollama" or "openai"). Defaults to settings.
        model: Model name. Defaults to the model configured for the selected provider.
        **kwargs: Additional arguments passed to the LLM constructor.

    Returns:
        A LangChain chat model instance.

    """
    provider = (provider or settings.normalized_llm_provider()).strip().lower()

    if provider == "ollama":
        effective_model = model or settings.ollama_model
    elif provider == "openai":
        effective_model = model or settings.openai_model
    else:
        effective_model = model or "unknown"

    logger.info(
        "LLM_FACTORY_CALLED create_llm | provider=%s | effective_model=%s",
        provider,
        effective_model,
    )

    temperature_value = kwargs.pop("temperature", settings.llm_temperature)
    temperature = float(temperature_value) if temperature_value is not None else None

    if provider == "ollama":
        model = effective_model

        if not settings.ollama_base_url:
            msg = "OLLAMA_BASE_URL is required when provider=ollama"
            raise ValueError(msg)

        if not model:
            msg = "OLLAMA_MODEL is required when provider=ollama"
            raise ValueError(msg)

        logger.info(
            "Using Ollama model: %s | base_url=%s | temperature=%s",
            model,
            settings.ollama_base_url,
            temperature,
        )

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    if provider == "openai":
        model = effective_model

        if not settings.openai_api_key:
            msg = "OPENAI_API_KEY is required when provider=openai"
            raise ValueError(msg)

        if not model:
            msg = "OPENAI_MODEL is required when provider=openai"
            raise ValueError(msg)

        timeout = kwargs.pop("timeout", settings.llm_timeout_seconds)
        max_retries = kwargs.pop("max_retries", settings.llm_max_retries)

        logger.info(
            "Using OpenAI model: %s | temperature=%s | timeout=%s | max_retries=%s",
            model,
            temperature,
            timeout,
            max_retries,
        )

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    msg = f"Unknown LLM provider: {provider}. Supported values: ollama, openai"
    raise ValueError(msg)


def build_chat_llm(
    temperature: float | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> BaseChatModel:
    """Build the configured chat LLM.

    This is the preferred entry point for new code.
    The active provider is controlled by settings.llm_provider / LLM_PROVIDER.
    """
    logger.info(
        "LLM_FACTORY_CALLED build_chat_llm | provider=%s | model=%s | temperature=%s",
        settings.normalized_llm_provider(),
        settings.active_llm_model(),
        temperature if temperature is not None else settings.llm_temperature,
    )

    if temperature is not None:
        kwargs["temperature"] = temperature

    return create_llm(**kwargs)


def get_llm(**kwargs: Any) -> BaseChatModel:  # noqa: ANN401
    """Get or create the global LLM instance.

    Args:
        **kwargs: Arguments passed to create_llm if creating a new instance.

    Returns:
        The global LLM instance.

    """
    global _llm_instance  # noqa: PLW0603

    if _llm_instance is None:
        logger.info(
            "LLM_FACTORY_CACHE_MISS get_llm | provider=%s | model=%s",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
        )
        _llm_instance = create_llm(**kwargs)
    else:
        logger.info(
            "LLM_FACTORY_CACHE_HIT get_llm | provider=%s | model=%s",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
        )

    return _llm_instance


def generate_text(prompt: str, temperature: float | None = None) -> str:
    """Generate plain text from the configured LLM."""
    started_at = time.perf_counter()

    logger.info(
        "LLM_FACTORY_CALLED generate_text | provider=%s | model=%s | prompt_chars=%d | temperature=%s",
        settings.normalized_llm_provider(),
        settings.active_llm_model(),
        len(prompt or ""),
        temperature if temperature is not None else settings.llm_temperature,
    )

    try:
        llm = build_chat_llm(temperature=temperature)

        logger.info(
            "LLM_INVOKE_START generate_text | provider=%s | model=%s",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
        )

        response = llm.invoke(prompt)
        content: Any = getattr(response, "content", response)

        output = content.strip() if isinstance(content, str) else str(content).strip()

        logger.info(
            "LLM_FACTORY_DONE generate_text | provider=%s | model=%s | elapsed_seconds=%.2f | output_chars=%d",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
            time.perf_counter() - started_at,
            len(output),
        )

    except Exception:
        logger.exception(
            "LLM_FACTORY_ERROR generate_text | provider=%s | model=%s | elapsed_seconds=%.2f",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
            time.perf_counter() - started_at,
        )
        raise
    else:
        return output


def reset_llm() -> None:
    """Reset the global LLM instance."""
    global _llm_instance  # noqa: PLW0603

    logger.info(
        "LLM_FACTORY_RESET reset_llm | provider=%s | model=%s",
        settings.normalized_llm_provider(),
        settings.active_llm_model(),
    )

    _llm_instance = None

"""Configuration settings for AgenticCyberSense."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # LLM Settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").strip().lower())
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    # Cloud LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))

    # Shared LLM runtime settings
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    llm_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "3600")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))

    # RAG Settings
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")),
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    pdf_docs_dir: Path = field(default_factory=lambda: Path(os.getenv("PDF_DOCS_DIR", "./data/documents")))

    # MCP Settings
    mcp_server_host: str = field(default_factory=lambda: os.getenv("MCP_SERVER_HOST", "localhost"))
    mcp_server_port: int = field(default_factory=lambda: int(os.getenv("MCP_SERVER_PORT", "8000")))
    # Target URL for MCP clients / SSE endpoints
    mcp_target_url: str = field(
        default_factory=lambda: os.getenv(
            "MCP_TARGET_URL",
            "http://127.0.0.1:8000/mcp/sse",
        ),
    )

    # API Server Settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))  # noqa: S104
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "7001")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Telegram client settings
    tg_api_id: int = field(default_factory=lambda: int(os.getenv("TG_API_ID", "0")))
    tg_api_hash: str = field(default_factory=lambda: os.getenv("TG_API_HASH", ""))
    tg_session_name: str = field(default_factory=lambda: os.getenv("TG_SESSION_NAME", "agentic_telegram_session"))

    # Agent / parser settings
    telegram_keywords: str = field(default_factory=lambda: os.getenv("TELEGRAM_KEYWORDS", ""))  # comma-separated

    def normalized_llm_provider(self) -> str:
        """Return normalized LLM provider name."""
        return (self.llm_provider or "ollama").strip().lower()

    def active_llm_model(self) -> str:
        """Return the active model name based on selected provider."""
        provider = self.normalized_llm_provider()

        if provider == "ollama":
            return self.ollama_model

        if provider == "openai":
            return self.openai_model

        msg = f"Unsupported LLM_PROVIDER: {self.llm_provider}. Supported values: ollama, openai"
        raise ValueError(msg)

    def validate_llm_settings(self) -> None:
        """Validate LLM provider configuration."""
        provider = self.normalized_llm_provider()

        if provider not in {"ollama", "openai"}:
            msg = f"Unsupported LLM_PROVIDER: {self.llm_provider}. Supported values: ollama, openai"
            raise ValueError(msg)

        if provider == "ollama":
            if not self.ollama_base_url:
                msg = "OLLAMA_BASE_URL is required when LLM_PROVIDER=ollama"
                raise ValueError(msg)

            if not self.ollama_model:
                msg = "OLLAMA_MODEL is required when LLM_PROVIDER=ollama"
                raise ValueError(msg)

        if provider == "openai":
            if not self.openai_api_key:
                msg = "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
                raise ValueError(msg)

            if not self.openai_model:
                msg = "OPENAI_MODEL is required when LLM_PROVIDER=openai"
                raise ValueError(msg)

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_docs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

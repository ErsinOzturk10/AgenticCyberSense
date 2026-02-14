"""Configuration settings for AgenticCyberSense."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # LLM Settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))

    # Cloud LLM (future use)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))

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
    mcp_server_port: int = field(default_factory=lambda: int(os.getenv("MCP_SERVER_PORT", "8765")))
    # Target URL for MCP clients / SSE endpoints
    mcp_target_url: str = field(
        default_factory=lambda: os.getenv(
            "MCP_TARGET_URL",
            "http://127.0.0.1:8000/mcp/sse",
        ),
    )

    # API Server Settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))  # noqa: S104
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_docs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

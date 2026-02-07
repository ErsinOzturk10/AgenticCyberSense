"""AgenticCyberSense - Agentic Cyber Threat Intelligence Platform."""

__version__ = "0.1.0"


def build_graph(**kwargs: object) -> object:
    """Lazy import to avoid circular imports."""
    from agenticcybersense.graph.build_graph import build_graph as _build_graph  # noqa: PLC0415

    return _build_graph(**kwargs)


def get_settings() -> object:
    """Get settings instance."""
    from agenticcybersense.settings import settings  # noqa: PLC0415

    return settings


__all__ = ["__version__", "build_graph", "get_settings"]

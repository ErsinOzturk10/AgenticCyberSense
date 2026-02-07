"""LangGraph orchestration module."""

from agenticcybersense.graph.state import GraphState


def build_graph(**kwargs: object) -> object:
    """Lazy import to avoid circular imports."""
    from agenticcybersense.graph.build_graph import build_graph as _build_graph  # noqa: PLC0415

    return _build_graph(**kwargs)


__all__ = ["GraphState", "build_graph"]

"""Agent registry for plugin-style agent management."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

from agenticcybersense.logging_utils import get_logger

if TYPE_CHECKING:
    from agenticcybersense.agents.base import BaseAgent

logger = get_logger("agents.registry")


class AgentRegistry:
    """Registry for managing available agents."""

    _instance: ClassVar[AgentRegistry] | None = None
    _agents: ClassVar[dict[str, type[BaseAgent]]]

    def __new__(cls) -> Self:
        """Create or return the singleton registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._agents = {}  # set class var to avoid private-member access
        return cls._instance

    def register(self, agent_class: type[BaseAgent]) -> type[BaseAgent]:
        """Register an agent class.

        Args:
            agent_class: The agent class to register

        Returns:
            The registered agent class (for use as decorator)

        """
        name = agent_class.name
        self._agents[name] = agent_class
        logger.info("Registered agent: %s", name)
        return agent_class

    def get(self, name: str) -> type[BaseAgent] | None:
        """Get an agent class by name."""
        return self._agents.get(name)

    def create(self, name: str, **kwargs: object) -> BaseAgent | None:
        """Create an agent instance by name."""
        agent_class = self.get(name)
        if agent_class is None:
            logger.warning("Agent not found: %s", name)
            return None
        return agent_class(**kwargs)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def get_agent_descriptions(self) -> dict[str, str]:
        """Get descriptions of all registered agents."""
        return {name: cls.description for name, cls in self._agents.items()}

    def clear(self) -> None:
        """Clear all registered agents (mainly for testing)."""
        self._agents.clear()


# Global registry instance
_registry = AgentRegistry()


def register_agent(agent_class: type[BaseAgent]) -> type[BaseAgent]:
    """Register an agent class with the global registry."""
    return _registry.register(agent_class)


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _registry

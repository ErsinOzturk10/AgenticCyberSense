"""Base agent interface and common helpers."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agenticcybersense.logging_utils import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from agenticcybersense.schemas.messages import AgentRequest, AgentResponse  # pragma: no cover

logger = get_logger("agents.base")


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    name: str = "base_agent"
    description: str = "Base agent interface"

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialize the agent.

        Args:
            llm: Language model instance. If None, will be created on first use.

        """
        self._llm = llm
        self.logger = get_logger(f"agents.{self.name}")

    @property
    def llm(self) -> BaseChatModel | None:
        """Get the language model, creating if necessary."""
        return self._ensure_llm()

    def _ensure_llm(self) -> BaseChatModel | None:
        if self._llm is None:
            try:
                factory_mod = importlib.import_module("agenticcybersense.llm.factory")
                get_llm = factory_mod.get_llm
                self._llm = get_llm()
            except (ImportError, AttributeError, RuntimeError) as e:
                self.logger.warning("Could not initialize LLM: %s", e)
                return None
        return self._llm

    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request.

        Args:
            request: The incoming agent request

        Returns:
            Agent response with results

        """

    def get_tools(self) -> list[Any]:
        """Get tools available to this agent."""
        return []

    def get_system_prompt(self, **_kwargs: object) -> str:
        """Get the system prompt for this agent."""
        return f"You are {self.name}. {self.description}"

    def to_messages(self, request: AgentRequest) -> list[object]:
        """Return messages appropriate for an LLM. Compatible fallback if langchain not present."""
        lc_messages: list[object] = []
        try:
            m = importlib.import_module("langchain_core.messages")
            human_message_cls = m.HumanMessage
            system_message_cls = m.SystemMessage
            lc_messages.append(system_message_cls(content=self.get_system_prompt()))
            lc_messages.append(human_message_cls(content=request.user_message))
        except (ImportError, AttributeError):
            # Fallback to simple dict-based messages if langchain_core is not installed.
            lc_messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": request.user_message},
            ]
        return lc_messages

    async def invoke_llm(self, messages: list[dict[str, str]]) -> str:
        """Invoke the LLM with messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            LLM response content

        """
        if self.llm is None:
            return "[LLM not available - returning dummy response]"

        try:
            m = importlib.import_module("langchain_core.messages")
            ai_message_cls = m.AIMessage
            human_message_cls = m.HumanMessage
            system_message_cls = m.SystemMessage

            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(system_message_cls(content=content))
                elif role == "assistant":
                    lc_messages.append(ai_message_cls(content=content))
                else:
                    lc_messages.append(human_message_cls(content=content))

            response = await self.llm.ainvoke(lc_messages)
            return str(response.content)
        except (ImportError, AttributeError, Exception):
            self.logger.exception("Error invoking LLM")
            return "[LLM Error]"

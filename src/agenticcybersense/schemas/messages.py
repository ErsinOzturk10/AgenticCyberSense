"""Message schemas for API and inter-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Role of a message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single chat message."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    """Request from OpenWebUI to the API server."""

    message: str
    conversation_id: str | None = None
    history: list[ChatMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from API server to OpenWebUI."""

    message: str
    conversation_id: str
    findings: list[Any] = field(default_factory=list)  # List of Finding objects
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentRequest:
    """Internal request passed between agents."""

    query: str
    context: dict[str, Any] = field(default_factory=dict)
    source_agent: str | None = None
    target_agent: str | None = None
    conversation_id: str | None = None


@dataclass
class AgentResponse:
    """Internal response from an agent."""

    content: str
    agent_name: str
    success: bool = True
    findings: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

"""LangGraph state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agenticcybersense.schemas.findings import Finding  # pragma: no cover
    from agenticcybersense.schemas.messages import AgentResponse  # pragma: no cover


@dataclass
class GraphState:
    """State object for the LangGraph orchestration."""

    query: str = ""
    conversation_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    current_agent: str = "orchestrator"
    agents_consulted: list[str] = field(default_factory=list)
    pending_agents: list[str] = field(default_factory=list)
    agent_responses: dict[str, AgentResponse] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    documentation_context: str = ""
    final_response: str = ""
    is_complete: bool = False
    error: str | None = None

    def add_response(self, agent_name: str, response: AgentResponse) -> None:
        """Add an agent response to the state."""
        self.agent_responses[agent_name] = response
        if agent_name not in self.agents_consulted:
            self.agents_consulted.append(agent_name)
        self.findings.extend(response.findings)
        if agent_name == "documentation":
            self.documentation_context = response.content

    def get_context_for_agent(self) -> dict[str, Any]:
        """Get context dict to pass to the next agent."""
        return {
            **self.context,
            "documentation_context": self.documentation_context,
            "previous_findings": [f.title for f in self.findings],
            "agents_consulted": self.agents_consulted,
        }

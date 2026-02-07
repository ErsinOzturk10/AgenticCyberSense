"""Build the LangGraph orchestration graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from agenticcybersense.graph.state import GraphState
from agenticcybersense.logging_utils import get_logger
from agenticcybersense.schemas.messages import AgentRequest

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = get_logger("graph.build")


def _from_dict(d: dict) -> GraphState:
    """Convert dict to GraphState."""
    return GraphState(
        query=d.get("query", ""),
        conversation_id=d.get("conversation_id", ""),
        context=d.get("context", {}),
        current_agent=d.get("current_agent", "orchestrator"),
        agents_consulted=list(d.get("agents_consulted", [])),
        pending_agents=list(d.get("pending_agents", [])),
        agent_responses=dict(d.get("agent_responses", {})),
        findings=list(d.get("findings", [])),
        documentation_context=d.get("documentation_context", ""),
        final_response=d.get("final_response", ""),
        is_complete=d.get("is_complete", False),
        error=d.get("error"),
    )


def _to_dict(state: GraphState) -> dict:
    """Convert GraphState to dict."""
    return {
        "query": state.query,
        "conversation_id": state.conversation_id,
        "context": state.context,
        "current_agent": state.current_agent,
        "agents_consulted": state.agents_consulted,
        "pending_agents": state.pending_agents,
        "agent_responses": state.agent_responses,
        "findings": state.findings,
        "documentation_context": state.documentation_context,
        "final_response": state.final_response,
        "is_complete": state.is_complete,
        "error": state.error,
    }


async def _process_agent(state: GraphState, agent_name: str) -> GraphState:
    """Process a single agent."""
    from agenticcybersense.agents.registry import get_registry  # noqa: PLC0415

    logger.info("Processing agent: %s", agent_name)
    registry = get_registry()
    agent = registry.create(agent_name)

    if agent is None:
        state.error = f"Agent not found: {agent_name}"
        return state

    request = AgentRequest(
        query=state.query,
        context=state.get_context_for_agent(),
        source_agent="graph",
        target_agent=agent_name,
        conversation_id=state.conversation_id,
    )

    try:
        response = await agent.process(request)
        state.add_response(agent_name, response)
        if agent_name in state.pending_agents:
            state.pending_agents.remove(agent_name)
    except Exception as e:
        logger.exception("Error in agent %s", agent_name)
        state.error = f"Error in {agent_name}: {e!s}"

    return state


def _determine_pending_agents(query: str) -> list[str]:
    """Determine which agents to consult based on query."""
    agents = []
    query_lower = query.lower()

    if any(word in query_lower for word in ["website", "web", "url", "news", "leak", "breach", "cve"]):
        agents.append("web")
    if any(word in query_lower for word in ["telegram", "channel", "group", "chat"]):
        agents.append("telegram")

    # If no specific match, consult all
    if not agents:
        agents = ["web", "telegram"]

    return agents


async def orchestrator_node(state_dict: dict) -> dict:
    """Entry point - sets up processing."""
    state = _from_dict(state_dict)
    logger.info("Orchestrator processing: %s", state.query[:100] if state.query else "empty query")
    state.current_agent = "orchestrator"
    state.pending_agents = _determine_pending_agents(state.query)
    return _to_dict(state)


async def documentation_node(state_dict: dict) -> dict:
    """Documentation agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "documentation")
    return _to_dict(state)


async def web_node(state_dict: dict) -> dict:
    """Web agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "web")
    return _to_dict(state)


async def telegram_node(state_dict: dict) -> dict:
    """Telegram agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "telegram")
    return _to_dict(state)


async def synthesize_node(state_dict: dict) -> dict:
    """Synthesize all agent responses."""
    state = _from_dict(state_dict)
    logger.info("Synthesizing responses from %d agents", len(state.agents_consulted))

    # Build final report
    parts = [
        "# 🛡️ Cyber Threat Intelligence Report\n\n",
        f"**Query:** {state.query}\n\n",
        "---\n\n",
    ]

    for agent_name, response in state.agent_responses.items():
        status = "✅" if response.success else "❌"
        parts.append(f"## {agent_name.title()} Agent {status}\n\n")
        parts.append(response.content)
        parts.append("\n\n")

    # Summary section
    parts.append("---\n\n")
    parts.append("## 📊 Summary\n\n")
    parts.append(f"- **Agents Consulted:** {', '.join(state.agents_consulted)}\n")
    parts.append(f"- **Total Findings:** {len(state.findings)}\n")

    if state.findings:
        parts.append("\n### Key Findings:\n")
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(state.findings, key=lambda f: severity_order.get(f.severity.value, 5))
        for finding in sorted_findings[:5]:
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}.get(finding.severity.value, "⚪")
            parts.append(f"- {emoji} **[{finding.severity.value.upper()}]** {finding.title}\n")

    state.final_response = "".join(parts)
    state.is_complete = True
    return _to_dict(state)


def router(state_dict: dict) -> str:
    """Route to the next node."""
    state = _from_dict(state_dict)

    if state.error:
        return "synthesize"

    if "documentation" not in state.agents_consulted:
        return "documentation"

    if state.pending_agents:
        next_agent = state.pending_agents[0]
        if next_agent == "web":
            return "web"
        if next_agent == "telegram":
            return "telegram"
        # Unknown agent, remove and continue
        state.pending_agents.pop(0)
        return router(_to_dict(state))

    return "synthesize"


def build_graph(**_kwargs: object) -> CompiledStateGraph:
    """Build and compile the orchestration graph."""
    logger.info("Building orchestration graph")

    # Ensure agents are registered by importing the agents module
    from agenticcybersense import agents as _agents  # noqa: PLC0415, F401

    # Create graph
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("documentation", documentation_node)
    workflow.add_node("web", web_node)
    workflow.add_node("telegram", telegram_node)
    workflow.add_node("synthesize", synthesize_node)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add conditional edges
    routing_map = {
        "documentation": "documentation",
        "web": "web",
        "telegram": "telegram",
        "synthesize": "synthesize",
    }

    workflow.add_conditional_edges("orchestrator", router, routing_map)
    workflow.add_conditional_edges("documentation", router, routing_map)
    workflow.add_conditional_edges("web", router, routing_map)
    workflow.add_conditional_edges("telegram", router, routing_map)

    # Synthesize goes to END
    workflow.add_edge("synthesize", END)

    # Compile
    graph = workflow.compile()
    logger.info("Graph compiled successfully")

    return graph

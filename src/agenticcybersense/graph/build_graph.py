"""Build the LangGraph orchestration graph."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph

from agenticcybersense.graph.state import GraphState
from agenticcybersense.logging_utils import get_logger
from agenticcybersense.query_analysis import QueryAnalysis, analyze_query, has_threat_intel_intent
from agenticcybersense.schemas.findings import Finding  # noqa: TC001
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse

logger = get_logger("graph.build")

WEB_ROUTING_TERMS = ("website", "web", "url", "news", "leak", "breach", "cve")
TELEGRAM_ROUTING_TERMS = ("telegram", "channel", "group", "chat")
LLM_SYNTHESIS_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_LLM_SYNTHESIS_MAX_AGENT_CHARS = 2500
MIN_LLM_SYNTHESIS_MAX_AGENT_CHARS = 500
KEY_FINDING_LIMIT = 5
FINDING_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
FINDING_SEVERITY_EMOJIS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
    "info": "🔵",
}


class GraphStateDict(TypedDict, total=False):
    """Typed state shape passed through LangGraph."""

    query: str
    conversation_id: str
    context: dict[str, Any]
    current_agent: str
    agents_consulted: list[str]
    pending_agents: list[str]
    agent_responses: dict[str, AgentResponse]
    findings: list[Finding]
    documentation_context: str
    final_response: str
    is_complete: bool
    error: str | None


def _from_dict(d: GraphStateDict) -> GraphState:
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


def _to_dict(state: GraphState) -> GraphStateDict:
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


def _should_consult_web(query_lower: str, query_analysis: QueryAnalysis) -> bool:
    return query_analysis.is_observable_lookup or any(word in query_lower for word in WEB_ROUTING_TERMS)


def _should_consult_telegram(
    query_lower: str,
    query_analysis: QueryAnalysis,
    *,
    has_breach_intent: bool,
    has_threat_intent: bool,
) -> bool:
    return query_analysis.is_observable_lookup or has_breach_intent or has_threat_intent or any(word in query_lower for word in TELEGRAM_ROUTING_TERMS)


def _append_checked_items(
    parts: list[str],
    response: AgentResponse | None,
    *,
    metadata_key: str,
    label: str,
) -> None:
    if response is None:
        return

    checked_items = response.metadata.get(metadata_key, [])
    if isinstance(checked_items, list) and checked_items:
        parts.append(f"- **{label} ({len(checked_items)}):** {', '.join(str(item) for item in checked_items)}\n")


def _append_key_findings(parts: list[str], findings: list[Finding]) -> None:
    if not findings:
        return

    parts.append("\n### Key Findings:\n")
    sorted_findings = sorted(
        findings,
        key=lambda finding: FINDING_SEVERITY_ORDER.get(
            finding.severity.value,
            len(FINDING_SEVERITY_ORDER),
        ),
    )

    for finding in sorted_findings[:KEY_FINDING_LIMIT]:
        emoji = FINDING_SEVERITY_EMOJIS.get(finding.severity.value, "⚪")
        parts.append(f"- {emoji} **[{finding.severity.value.upper()}]** {finding.title}\n")


def _build_deterministic_report(state: GraphState) -> str:
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

    parts.append("---\n\n")
    parts.append("## 📊 Summary\n\n")
    parts.append(f"- **Agents Consulted:** {', '.join(state.agents_consulted)}\n")
    parts.append(f"- **Total Findings:** {len(state.findings)}\n")
    _append_checked_items(
        parts,
        state.agent_responses.get("web"),
        metadata_key="checked_sources",
        label="Web Sources Checked",
    )
    _append_checked_items(
        parts,
        state.agent_responses.get("telegram"),
        metadata_key="checked_channels",
        label="Telegram Channels Checked",
    )
    _append_key_findings(parts, state.findings)
    return "".join(parts)


def _llm_synthesis_enabled() -> bool:
    import os  # noqa: PLC0415

    return os.getenv("API_ENABLE_LLM_SYNTHESIS", "false").strip().lower() in LLM_SYNTHESIS_ENABLED_VALUES


def _get_max_agent_chars() -> int:
    import os  # noqa: PLC0415

    max_agent_chars_raw = os.getenv(
        "API_LLM_SYNTHESIS_MAX_AGENT_CHARS",
        str(DEFAULT_LLM_SYNTHESIS_MAX_AGENT_CHARS),
    ).strip()
    try:
        return max(MIN_LLM_SYNTHESIS_MAX_AGENT_CHARS, int(max_agent_chars_raw))
    except ValueError:
        logger.warning(
            "Invalid API_LLM_SYNTHESIS_MAX_AGENT_CHARS=%r; using default %d",
            max_agent_chars_raw,
            DEFAULT_LLM_SYNTHESIS_MAX_AGENT_CHARS,
        )
        return DEFAULT_LLM_SYNTHESIS_MAX_AGENT_CHARS


def _build_agent_context(state: GraphState, max_agent_chars: int) -> str:
    agent_sections = []
    for agent_name, response in state.agent_responses.items():
        status = "success" if response.success else "failed"
        content = response.content or ""

        if len(content) > max_agent_chars:
            logger.info(
                "Truncating agent output for LLM synthesis | agent=%s | original_chars=%d | max_chars=%d",
                agent_name,
                len(content),
                max_agent_chars,
            )
            content = content[:max_agent_chars] + "\n\n[TRUNCATED: agent output shortened for LLM synthesis]"

        agent_sections.append(f"## Agent: {agent_name}\nStatus: {status}\nContent:\n{content}\n")

    return "\n\n---\n\n".join(agent_sections)


def _build_synthesis_prompt(state: GraphState, agent_context: str) -> str:
    return f"""
You are the final synthesis layer for AgenticCyberSense, a cyber threat intelligence system.

Create a concise Markdown Cyber Threat Intelligence Report using ONLY the agent outputs below.

Strict rules:
- Do NOT invent facts, CVEs, channels, sources, numbers, timestamps, or definitions.
- Do NOT change the subject of source statements.
- If a source says "cyber threats are defined as...", do NOT rewrite it as "CVE is defined as...".
- If the provided context does not define CVE directly, say that the retrieved documentation references CVE as part of vulnerability/exploit concepts.
- Preserve source names, page numbers, channel names, findings counts, and severity labels when present.
- If an agent says no results were found, keep that limitation.
- Keep the answer shorter than 700 words.

Required sections:
1. Executive Summary
2. Documentation Intelligence
3. Web Intelligence
4. Telegram Intelligence, only if Telegram data exists
5. Key Findings
6. Analyst Notes / Limitations

User query:
{state.query}

Agents consulted:
{", ".join(state.agents_consulted)}

Total findings:
{len(state.findings)}

Agent outputs:
{agent_context}
""".strip()


def _generate_llm_report(state: GraphState) -> str | None:
    import time  # noqa: PLC0415

    from agenticcybersense.llm.factory import generate_text  # noqa: PLC0415

    started_at = time.perf_counter()
    max_agent_chars = _get_max_agent_chars()
    agent_context = _build_agent_context(state, max_agent_chars)
    synthesis_prompt = _build_synthesis_prompt(state, agent_context)

    logger.info(
        "LLM synthesis prompt prepared | prompt_chars=%d | max_agent_chars=%d",
        len(synthesis_prompt),
        max_agent_chars,
    )

    llm_report = generate_text(synthesis_prompt)
    if not llm_report.strip():
        logger.warning("LLM synthesis returned empty output; falling back to deterministic report")
        return None

    logger.info(
        "LLM synthesis completed | elapsed_seconds=%.2f | output_chars=%d",
        time.perf_counter() - started_at,
        len(llm_report),
    )
    return llm_report.strip()


def _determine_pending_agents(query: str) -> list[str]:
    """Determine which agents to consult based on query."""
    agents = []
    query_lower = query.lower()
    query_analysis = analyze_query(query)
    has_breach_intent = any(term in query_analysis.intent_terms for term in ("breach", "leak"))
    has_threat_intent = has_threat_intel_intent(query_analysis)

    if _should_consult_web(query_lower, query_analysis):
        agents.append("web")
    if _should_consult_telegram(
        query_lower,
        query_analysis,
        has_breach_intent=has_breach_intent,
        has_threat_intent=has_threat_intent,
    ):
        agents.append("telegram")

    # If no specific match, consult all
    if not agents:
        agents = ["web", "telegram"]

    return agents


async def orchestrator_node(state_dict: GraphStateDict) -> GraphStateDict:
    """Entry point - sets up processing."""
    state = _from_dict(state_dict)
    logger.info("Orchestrator processing: %s", state.query[:100] if state.query else "empty query")
    state.current_agent = "orchestrator"
    state.pending_agents = _determine_pending_agents(state.query)
    return _to_dict(state)


async def documentation_node(state_dict: GraphStateDict) -> GraphStateDict:
    """Documentation agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "documentation")
    return _to_dict(state)


async def web_node(state_dict: GraphStateDict) -> GraphStateDict:
    """Web agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "web")
    return _to_dict(state)


async def telegram_node(state_dict: GraphStateDict) -> GraphStateDict:
    """Telegram agent node."""
    state = _from_dict(state_dict)
    state = await _process_agent(state, "telegram")
    return _to_dict(state)


async def synthesize_node(state_dict: GraphStateDict) -> GraphStateDict:
    """Synthesize all agent responses.

    By default, this builds the existing fast deterministic report.

    If API_ENABLE_LLM_SYNTHESIS=true is set in .env, the agent outputs are
    passed through the configured LLM provider for final CTI synthesis.

    If LLM synthesis fails, the deterministic report is returned as fallback.
    """
    state = _from_dict(state_dict)
    logger.info("Synthesizing responses from %d agents", len(state.agents_consulted))

    deterministic_report = _build_deterministic_report(state)
    if not _llm_synthesis_enabled():
        logger.info("LLM synthesis disabled; using deterministic synthesis report")
        state.final_response = deterministic_report
        state.is_complete = True
        return _to_dict(state)

    logger.info("LLM synthesis enabled; generating final report with configured LLM")

    try:
        state.final_response = _generate_llm_report(state) or deterministic_report
    except Exception:
        logger.exception("LLM synthesis failed; falling back to deterministic report")
        state.final_response = deterministic_report

    state.is_complete = True
    return _to_dict(state)


def router(state_dict: GraphStateDict) -> str:
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


def build_graph(**_kwargs: object) -> Any:  # noqa: ANN401
    """Build and compile the orchestration graph."""
    logger.info("Building orchestration graph")

    # Ensure agents are registered by importing the agents module
    from agenticcybersense import agents as _agents  # noqa: PLC0415, F401

    # Create graph
    workflow: Any = StateGraph(GraphStateDict)

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

    return cast("Any", graph)

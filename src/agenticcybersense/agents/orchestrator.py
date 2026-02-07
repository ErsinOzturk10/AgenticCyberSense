"""Orchestrator Agent - Main coordinator for the CTI platform."""

from __future__ import annotations

from typing import Any

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import get_registry, register_agent
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse


@register_agent
class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates other agents."""

    name: str = "orchestrator"
    description: str = "Main coordinator that routes queries to specialized agents and synthesizes results"

    def __init__(self, **_kwargs: object) -> None:
        """Initialize the orchestrator."""
        super().__init__(**(_kwargs or {}))
        self._agent_instances: dict[str, BaseAgent] = {}

    def _get_agent(self, name: str) -> BaseAgent | None:
        """Get or create an agent instance."""
        if name not in self._agent_instances:
            registry = get_registry()
            agent = registry.create(name, llm=self._llm)
            if agent:
                self._agent_instances[name] = agent
        return self._agent_instances.get(name)

    async def _consult_documentation(self, query: str, context: dict[str, object]) -> AgentResponse:
        """Consult documentation agent first."""
        doc_agent = self._get_agent("documentation")
        if doc_agent is None:
            return AgentResponse(
                content="Documentation agent not available",
                agent_name="documentation",
                success=False,
                error="Agent not found",
            )

        doc_request = AgentRequest(
            query=query,
            context=context,
            source_agent=self.name,
            target_agent="documentation",
        )
        return await doc_agent.process(doc_request)

    async def _route_to_agent(self, agent_name: str, query: str, context: dict[str, Any]) -> AgentResponse:
        """Route a query to a specific agent."""
        agent = self._get_agent(agent_name)
        if agent is None:
            return AgentResponse(
                content=f"Agent {agent_name} not available",
                agent_name=agent_name,
                success=False,
                error="Agent not found",
            )

        request = AgentRequest(
            query=query,
            context=context,
            source_agent=self.name,
            target_agent=agent_name,
        )
        return await agent.process(request)

    def _determine_next_agents(self, query: str) -> list[str]:
        """Determine which agents to consult based on query."""
        agents = []
        query_lower = query.lower()

        # Keyword-based routing
        if any(word in query_lower for word in ["website", "web", "url", "news", "leak", "breach", "cve"]):
            agents.append("web")
        if any(word in query_lower for word in ["telegram", "channel", "group", "chat", "message"]):
            agents.append("telegram")

        # If no specific agents identified, consult all
        if not agents:
            registry = get_registry()
            agents = [name for name in registry.list_agents() if name not in (self.name, "documentation")]

        return agents

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process an incoming request by coordinating agents."""
        self.logger.info("Orchestrator processing query: %s", request.query[:100])

        # Step 1: Always consult documentation first
        doc_response = await self._consult_documentation(request.query, request.context)
        self.logger.info("Documentation agent responded")

        # Update context with documentation findings
        context = {**request.context, "documentation_context": doc_response.content}

        # Step 2: Determine which other agents to consult
        next_agents = self._determine_next_agents(request.query)
        self.logger.info("Will consult agents: %s", next_agents)

        # Step 3: Consult each agent
        all_responses = [doc_response]
        all_findings = list(doc_response.findings)

        for agent_name in next_agents:
            self.logger.info("Consulting agent: %s", agent_name)
            response = await self._route_to_agent(agent_name, request.query, context)
            all_responses.append(response)
            all_findings.extend(response.findings)

        # Step 4: Synthesize results
        synthesized_content = self._synthesize_responses(request.query, all_responses)

        return AgentResponse(
            content=synthesized_content,
            agent_name=self.name,
            success=True,
            findings=all_findings,
            metadata={
                "consulted_agents": ["documentation", *next_agents],
                "num_findings": len(all_findings),
            },
        )

    def _synthesize_responses(self, query: str, responses: list[AgentResponse]) -> str:
        """Synthesize responses from multiple agents."""
        parts = [
            "# 🛡️ Cyber Threat Intelligence Report\n",
            f"**Query:** {query}\n",
            f"**Timestamp:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
        ]

        for response in responses:
            status = "✅" if response.success else "❌"
            parts.append(f"\n## {response.agent_name.title()} Agent {status}\n")
            parts.append(response.content)

            if response.findings:
                parts.append(f"\n**Findings:** {len(response.findings)} items identified\n")
                parts.extend([f"- [{finding.severity.value.upper()}] {finding.title}\n" for finding in response.findings[:3]])

        # Summary
        total_findings = sum(len(r.findings) for r in responses)
        parts.append("\n---\n")
        parts.append("## 📊 Summary\n")
        parts.append(f"- **Agents Consulted:** {len(responses)}\n")
        parts.append(f"- **Total Findings:** {total_findings}\n")

        return "".join(parts)

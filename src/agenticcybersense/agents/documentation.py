"""Documentation Agent - RAG-based knowledge retrieval."""

from __future__ import annotations

import re
from typing import Any

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse


@register_agent
class DocumentationAgent(BaseAgent):
    """Documentation agent with RAG capabilities."""

    name: str = "documentation"
    description: str = "Retrieves and analyzes security documentation, CVE databases, and technical references"

    def __init__(self, **_kwargs: object) -> None:
        """Initialize the documentation agent."""
        super().__init__(**(_kwargs or {}))
        self._retriever = None

    @property
    def retriever(self) -> object | None:
        """Lazy load the retriever."""
        if self._retriever is None:
            try:
                from agenticcybersense.rag.retriever import DocumentRetriever  # noqa: PLC0415

                self._retriever = DocumentRetriever()
            except ImportError as e:
                self.logger.warning("Could not initialize retriever: %s", e)
        return self._retriever

    async def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant documentation context."""
        if self.retriever is None:
            return "No documentation retriever available."
        return self.retriever.retrieve_as_context(query)

    async def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze the query to extract key security terms."""
        query_lower = query.lower()

        analysis = {
            "is_cve_query": "cve" in query_lower,
            "is_vulnerability_query": any(w in query_lower for w in ["vulnerability", "vuln", "exploit", "attack"]),
            "is_threat_query": any(w in query_lower for w in ["threat", "malware", "ransomware", "phishing"]),
            "is_compliance_query": any(w in query_lower for w in ["compliance", "gdpr", "hipaa", "pci", "iso"]),
            "extracted_terms": [],
        }

        # Extract potential CVE IDs
        cve_pattern = r"CVE-\d{4}-\d{4,}"
        cves = re.findall(cve_pattern, query.upper())
        if cves:
            analysis["extracted_terms"].extend(cves)

        return analysis

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a documentation query."""
        self.logger.info("Documentation agent processing: %s", request.query[:100])

        # Analyze the query
        analysis = await self._analyze_query(request.query)

        # Retrieve relevant context
        context = await self._retrieve_context(request.query)

        # Build response based on query type
        response_parts = ["### 📚 Documentation Analysis\n"]

        if analysis["is_cve_query"]:
            response_parts.append("**Query Type:** CVE/Vulnerability Lookup\n")
            if analysis["extracted_terms"]:
                response_parts.append(f"**Detected CVEs:** {', '.join(analysis['extracted_terms'])}\n")
        elif analysis["is_threat_query"]:
            response_parts.append("**Query Type:** Threat Intelligence\n")
        elif analysis["is_compliance_query"]:
            response_parts.append("**Query Type:** Compliance/Standards\n")
        else:
            response_parts.append("**Query Type:** General Security Query\n")

        response_parts.append("\n**Retrieved Context:**\n")
        response_parts.append(context)

        response_parts.append("\n\n**Recommendations:**\n")
        response_parts.append("1. Review the referenced documentation for detailed technical information\n")
        response_parts.append("2. Check for recent updates or patches related to this topic\n")
        response_parts.append("3. Consult additional threat intelligence sources for current threat landscape\n")

        # Create findings
        findings = []
        if analysis["is_cve_query"] or analysis["is_vulnerability_query"]:
            findings.append(
                Finding(
                    title=f"Documentation Reference: {request.query[:50]}",
                    description="Relevant security documentation was found for this query.",
                    severity=Severity.INFO,
                    source=SourceRef(
                        source_type=SourceType.DOCUMENTATION,
                        name="Internal Knowledge Base",
                    ),
                    tags=["documentation", "reference"],
                    recommendations=["Review full documentation for detailed information"],
                ),
            )

        return AgentResponse(
            content="".join(response_parts),
            agent_name=self.name,
            success=True,
            findings=findings,
            metadata={
                "query_analysis": analysis,
                "context_retrieved": bool(context),
            },
        )

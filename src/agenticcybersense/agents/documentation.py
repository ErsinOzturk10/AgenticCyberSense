"""Documentation Agent - RAG-based knowledge retrieval."""

from __future__ import annotations

import re
from typing import Any

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.query_analysis import analyze_query
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse


@register_agent
class DocumentationAgent(BaseAgent):
    """Documentation agent with RAG capabilities."""

    name: str = "documentation"
    description: str = "Retrieves and analyzes security documentation, CVE databases, and technical references"

    async def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant documentation context via RAG search."""
        from agenticcybersense.rag.rag import rag_search  # noqa: PLC0415

        return rag_search(query)

    async def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze the query to extract key security terms."""
        query_lower = query.lower()

        analysis: dict[str, Any] = {
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
        query_analysis = analyze_query(request.query)

        should_abstain = (
            query_analysis.is_observable_lookup
            and not analysis["is_cve_query"]
            and not analysis["is_vulnerability_query"]
            and not analysis["is_compliance_query"]
        )

        # Retrieve relevant context
        context = ""
        if not should_abstain:
            context = await self._retrieve_context(request.query)

        # Build response based on query type
        response_parts = ["### 📚 Documentation Analysis\n"]

        if should_abstain:
            response_parts.append("**Query Type:** Observable/Breach Lookup\n")
        elif analysis["is_cve_query"]:
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
        if should_abstain:
            response_parts.append(
                "No documentation-specific context retrieved. This query targets a specific observable, so direct web or Telegram matches are more reliable than generic reference material.\n",
            )
        else:
            response_parts.append(context)

        response_parts.append("\n\n**Recommendations:**\n")
        if should_abstain:
            response_parts.append("1. Review exact-match web or Telegram findings for the requested observable\n")
            response_parts.append("2. Broaden the lookup to related domains, usernames, or aliases if no direct match exists\n")
            response_parts.append("3. Confirm any suspected leak against the original source before treating it as verified\n")
        else:
            response_parts.append("1. Review the referenced documentation for detailed technical information\n")
            response_parts.append("2. Check for recent updates or patches related to this topic\n")
            response_parts.append("3. Consult additional threat intelligence sources for current threat landscape\n")

        # Create findings
        findings: list[Finding] = []
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

"""Web Intelligence Agent - Monitors websites for threat intelligence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse

RELEVANCE_THRESHOLD = 0.5
MEDIUM_THRESHOLD = 0.7


@register_agent
class WebAgent(BaseAgent):
    """Web intelligence agent for monitoring websites."""

    name: str = "web"
    description: str = "Monitors websites for security news, leaked data, and threat intelligence"

    # Default security news sources
    DEFAULT_SOURCES: ClassVar[list[dict[str, str]]] = [
        {"name": "NIST NVD", "url": "https://nvd.nist.gov", "type": "cve_database"},
        {"name": "CISA Alerts", "url": "https://www.cisa.gov/news-events/alerts", "type": "government"},
        {"name": "Krebs on Security", "url": "https://krebsonsecurity.com", "type": "news"},
        {"name": "The Hacker News", "url": "https://thehackernews.com", "type": "news"},
        {"name": "BleepingComputer", "url": "https://www.bleepingcomputer.com", "type": "news"},
    ]

    def __init__(self, target_websites: list[dict[str, str]] | None = None, **_kwargs: object) -> None:
        """Initialize the web agent."""
        super().__init__(**(_kwargs or {}))
        self.target_websites = target_websites or self.DEFAULT_SOURCES

    async def _fetch_website_content(self, source: dict[str, str]) -> dict[str, Any]:
        """Fetch content from a website (simulated for now).

        TODO: Implement real web scraping with httpx/aiohttp
        """
        self.logger.info("Checking source: %s", source["name"])

        # Simulated responses based on source type
        simulated_data = {
            "cve_database": {
                "status": "checked",
                "recent_cves": ["CVE-2024-1234", "CVE-2024-5678"],
                "summary": "Latest vulnerabilities from official database",
            },
            "government": {
                "status": "checked",
                "alerts": ["Active exploitation of critical vulnerability"],
                "summary": "Government security advisories",
            },
            "news": {
                "status": "checked",
                "headlines": ["New ransomware campaign targeting enterprises"],
                "summary": "Security news and analysis",
            },
        }

        return {
            "source": source,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": simulated_data.get(source.get("type", "news"), {"status": "checked"}),
        }

    async def _analyze_for_threats(self, query: str, results: list[dict[str, Any]]) -> list[Finding]:
        """Analyze fetched content for relevant threats."""
        findings = []
        query_lower = query.lower()

        for result in results:
            source = result["source"]
            data = result["data"]

            # Check if this source has relevant information
            relevance_score = 0

            # Simple relevance scoring
            if "cve" in query_lower and source.get("type") == "cve_database":
                relevance_score = 0.9
            elif any(word in query_lower for word in ["news", "recent", "latest"]):
                relevance_score = 0.7
            elif "alert" in query_lower and source.get("type") == "government":
                relevance_score = 0.85
            else:
                relevance_score = 0.5

            if relevance_score > RELEVANCE_THRESHOLD:
                severity = Severity.MEDIUM if relevance_score > MEDIUM_THRESHOLD else Severity.LOW

                findings.append(
                    Finding(
                        title=f"Web Intel: {source['name']}",
                        description=f"Relevant information found from {source['name']}: {data.get('summary', 'No summary')}",
                        severity=severity,
                        source=SourceRef(
                            source_type=SourceType.WEBSITE,
                            url=source.get("url"),
                            name=source["name"],
                        ),
                        tags=["web", "osint", source.get("type", "unknown")],
                        raw_data=data,
                    ),
                )

        return findings

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a web intelligence query."""
        self.logger.info("Web agent processing: %s", request.query[:100])

        # Fetch from all configured sources
        results = []
        for source in self.target_websites:
            try:
                result = await self._fetch_website_content(source)
                results.append(result)
            except (RuntimeError, ImportError) as e:
                self.logger.warning("Error fetching %s: %s", source["name"], e)

        # Analyze for threats
        findings = await self._analyze_for_threats(request.query, results)

        # Build response
        response_parts = [
            "### 🌐 Web Intelligence Report\n",
            f"**Query:** {request.query}\n",
            f"**Sources Checked:** {len(results)}\n",
            f"**Findings:** {len(findings)}\n\n",
        ]

        response_parts.append("**Source Summary:**\n")
        for result in results:
            source = result["source"]
            status = result["data"].get("status", "unknown")
            response_parts.append(f"- **{source['name']}** ({source['url']}): {status}\n")

            # Add details
            data = result["data"]
            if "headlines" in data:
                response_parts.append(f"  - Headlines: {', '.join(data['headlines'][:2])}\n")
            if "recent_cves" in data:
                response_parts.append(f"  - Recent CVEs: {', '.join(data['recent_cves'][:3])}\n")
            if "alerts" in data:
                response_parts.append(f"  - Alerts: {', '.join(data['alerts'][:2])}\n")

        response_parts.append("\n**Analysis:**\n")
        if findings:
            response_parts.append(f"Found {len(findings)} relevant items across monitored sources.\n")
            # Use extend for the transformed list
            response_parts.extend([f"- [{finding.severity.value.upper()}] {finding.title}\n" for finding in findings[:5]])
        else:
            response_parts.append("No specific threats matching your query were found in current monitoring.\n")

        return AgentResponse(
            content="".join(response_parts),
            agent_name=self.name,
            success=True,
            findings=findings,
            metadata={
                "sources_checked": len(results),
                "findings_count": len(findings),
            },
        )

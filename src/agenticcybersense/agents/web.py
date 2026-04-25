"""Web Intelligence Agent - Webcrawler makes queries for CTI from RAG index."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

# Severity thresholds used when mapping relevance score -> finding severity
SEVERITY_HIGH_THRESHOLD = 0.85
SEVERITY_MEDIUM_THRESHOLD = 0.70


def _severity_from_score(score: float) -> Severity:
    if score >= SEVERITY_HIGH_THRESHOLD:
        return Severity.HIGH
    if score >= SEVERITY_MEDIUM_THRESHOLD:
        return Severity.MEDIUM
    return Severity.LOW


@register_agent
class WebAgent(BaseAgent):
    """Web intelligence agent — webcrawler RAG index'ini sorgular."""

    name: str = "web"
    description: str = "Searches the webcrawler RAG index for threat intelligence from monitored security sites"

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Webcrawler RAG index'ine sorgu at, sonuçlari döndür."""
        self.logger.info("WebAgent processing: %s", request.query[:100])

        # RAG modulunu lazy import et
        try:
            from agenticcybersense.web_crawler.rag_ingest import query_webcrawler_rag  # noqa: PLC0415
        except ImportError:
            logger.exception("rag_ingest import failed")
            return self._error_response("RAG module unavailable")

        # ChromaDB'ye sorgu at
        try:
            rag_results = query_webcrawler_rag(request.query, n_results=8)
        except Exception:
            logger.exception("RAG query failed")
            return self._error_response("RAG query failed")

        # Sonuc yoksa bilgi ver
        if not rag_results:
            return AgentResponse(
                content=self._empty_response(request.query),
                agent_name=self.name,
                success=True,
                findings=[],
                metadata={"source": "webcrawler_rag", "results": 0},
            )

        # Finding'lere donustur (URL basina 1 finding)
        findings: list[Finding] = []
        seen_urls: set[str] = set()

        for item in rag_results:
            url = item.get("url") or item.get("site_url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            score = item.get("score", 0.0)
            title = item.get("title") or url
            preview = item.get("content", "")[:200]

            findings.append(
                Finding(
                    title=title[:120],
                    description=preview,
                    severity=_severity_from_score(score),
                    source=SourceRef(
                        source_type=SourceType.WEBSITE,
                        url=url,
                        name=item.get("site_url", url)[:80],
                    ),
                    tags=["web", "osint", "crawler"],
                    raw_data={
                        "relevance_score": score,
                        "last_updated": item.get("last_updated", ""),
                    },
                ),
            )

        return AgentResponse(
            content=self._format_response(request.query, rag_results, findings),
            agent_name=self.name,
            success=True,
            findings=findings,
            metadata={
                "source": "webcrawler_rag",
                "results": len(rag_results),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Yardimci metodlar
    # ------------------------------------------------------------------

    def _format_response(
        self,
        query: str,
        rag_results: list[dict[str, Any]],
        findings: list[Finding],
    ) -> str:
        lines = [
            "### 🌐 Web Intelligence Report\n",
            f"**Query:** {query}\n",
            f"**Sources found:** {len(rag_results)} relevant chunks\n",
            f"**Timestamp:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n",
            "\n#### Relevant Findings\n",
        ]

        for i, item in enumerate(rag_results[:5], 1):
            score = item.get("score", 0)
            url = item.get("url") or item.get("site_url", "N/A")
            title = item.get("title") or url
            content = item.get("content", "")[:300].replace("\n", " ")
            last_updated = item.get("last_updated", "")[:10] or "unknown"

            lines += [
                f"\n**[{i}] {title}**\n",
                f"- URL: {url}\n",
                f"- Relevance: {score:.0%}\n",
                f"- Last crawled: {last_updated}\n",
                f"- Preview: {content}...\n",
            ]

        high = [f for f in findings if f.severity == Severity.HIGH]
        medium = [f for f in findings if f.severity == Severity.MEDIUM]
        low = [f for f in findings if f.severity == Severity.LOW]

        lines.append(
            f"\n**Severity:** 🔴 High: {len(high)} | 🟡 Medium: {len(medium)} | 🟢 Low: {len(low)}\n",
        )

        return "".join(lines)

    def _empty_response(self, query: str) -> str:
        return (
            "### 🌐 Web Intelligence Report\n\n"
            f"**Query:** {query}\n\n"
            "⚠️ No results found in webcrawler RAG index.\n\n"
            "**To populate the index:**\n"
            "1. Run the crawler: `python main_trafilatura.py`\n"
            "2. Ingest results: `python rag_ingest.py output/latest_results.json`\n"
        )

    def _error_response(self, reason: str) -> AgentResponse:
        return AgentResponse(
            content=f"### 🌐 Web Intelligence Report\n\n❌ Error: {reason}\n",
            agent_name=self.name,
            success=False,
            error=reason,
        )

"""Web Intelligence Agent - Webcrawler makes queries for CTI from RAG index."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.query_analysis import QueryAnalysis, analyze_query
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

_WEB_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "web_crawler" / "output" / "latest_results.json"

# Severity thresholds used when mapping relevance score -> finding severity
SEVERITY_HIGH_THRESHOLD = 0.85
SEVERITY_MEDIUM_THRESHOLD = 0.70


def _severity_from_score(score: float) -> Severity:
    if score >= SEVERITY_HIGH_THRESHOLD:
        return Severity.HIGH
    if score >= SEVERITY_MEDIUM_THRESHOLD:
        return Severity.MEDIUM
    return Severity.LOW


@lru_cache(maxsize=1)
def _get_monitored_web_sources() -> tuple[str, ...]:
    try:
        raw_data = json.loads(_WEB_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return ()

    if not isinstance(raw_data, dict):
        return ()

    ordered_sources: list[str] = []
    seen: set[str] = set()

    for site_url in raw_data:
        domain = urlparse(site_url).netloc or str(site_url)
        domain = domain.removeprefix("www.")
        if not domain or domain in seen:
            continue
        seen.add(domain)
        ordered_sources.append(domain)

    return tuple(ordered_sources)


@register_agent
class WebAgent(BaseAgent):
    """Web intelligence agent — webcrawler RAG index'ini sorgular."""

    name: str = "web"
    description: str = "Searches the webcrawler RAG index for threat intelligence from monitored security sites"

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Webcrawler RAG index'ine sorgu at, sonuçlari döndür."""
        self.logger.info("WebAgent processing: %s", request.query[:100])
        query_analysis = analyze_query(request.query)
        checked_sources = list(_get_monitored_web_sources())

        # RAG modulunu lazy import et
        try:
            from agenticcybersense.web_crawler.rag_ingest import query_webcrawler_rag  # noqa: PLC0415
        except ImportError:
            logger.exception("rag_ingest import failed")
            return self._error_response("RAG module unavailable")

        # ChromaDB'ye sorgu at
        try:
            rag_results = query_webcrawler_rag(query_analysis.search_query, n_results=8)
        except Exception:
            logger.exception("RAG query failed")
            return self._error_response("RAG query failed")

        rag_results = self._filter_results(query_analysis, rag_results)

        # Sonuc yoksa bilgi ver
        if not rag_results:
            return AgentResponse(
                content=self._empty_response(request.query, query_analysis.observables),
                agent_name=self.name,
                success=True,
                findings=[],
                metadata={"source": "webcrawler_rag", "results": 0, "checked_sources": checked_sources},
            )

        # Finding'lere donustur (URL basina 1 finding)
        findings: list[Finding] = []
        seen_urls: set[str] = set()

        for item in rag_results:
            url = str(item.get("url") or item.get("site_url", ""))
            if url in seen_urls:
                continue
            seen_urls.add(url)

            score = float(item.get("score", 0.0) or 0.0)
            title = str(item.get("title") or url)
            preview = str(item.get("content", ""))[:200]

            findings.append(
                Finding(
                    title=title[:120],
                    description=preview,
                    severity=_severity_from_score(score),
                    source=SourceRef(
                        source_type=SourceType.WEBSITE,
                        url=url,
                        name=str(item.get("site_url", url))[:80],
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
                "refined_query": query_analysis.search_query,
                "checked_sources": checked_sources,
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
            self._format_checked_sources(),
            "\n#### Relevant Findings\n",
        ]

        for i, item in enumerate(rag_results[:5], 1):
            score = float(item.get("score", 0.0) or 0.0)
            url = str(item.get("url") or item.get("site_url", "N/A"))
            title = str(item.get("title") or url)
            content = str(item.get("content", ""))[:300].replace("\n", " ")
            last_updated = str(item.get("last_updated", ""))[:10] or "unknown"

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

    def _empty_response(self, query: str, observables: tuple[str, ...]) -> str:
        coverage = self._format_checked_sources()

        if observables:
            observable_list = ", ".join(observables)
            return (
                "### 🌐 Web Intelligence Report\n\n"
                f"**Query:** {query}\n\n"
                f"⚠️ No direct matches found in stored web sources for: {observable_list}.\n\n"
                f"{coverage}"
                "Try a broader lookup with domain variants, usernames, or alternative spellings if needed.\n"
            )

        return (
            "### 🌐 Web Intelligence Report\n\n"
            f"**Query:** {query}\n\n"
            "⚠️ No results found in webcrawler RAG index.\n\n"
            f"{coverage}"
            "**To populate the index:**\n"
            "1. Run the crawler: `python main_trafilatura.py`\n"
            "2. Ingest results: `python rag_ingest.py output/latest_results.json`\n"
        )

    def _format_checked_sources(self) -> str:
        checked_sources = _get_monitored_web_sources()
        if not checked_sources:
            return ""

        return f"**Checked Web Sources ({len(checked_sources)}):**\n{', '.join(checked_sources)}\n\n"

    def _filter_results(
        self,
        query_analysis: QueryAnalysis,
        rag_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not query_analysis.observables:
            return rag_results

        filtered_results: list[dict[str, Any]] = []
        for item in rag_results:
            searchable_text = " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("content", "")),
                    str(item.get("url", "")),
                    str(item.get("site_url", "")),
                ],
            ).lower()

            if any(observable.lower() in searchable_text for observable in query_analysis.observables):
                filtered_results.append(item)

        return filtered_results

    def _error_response(self, reason: str) -> AgentResponse:
        return AgentResponse(
            content=f"### 🌐 Web Intelligence Report\n\n❌ Error: {reason}\n",
            agent_name=self.name,
            success=False,
            error=reason,
        )

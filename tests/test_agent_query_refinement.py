"""Regression tests for observable-style query handling across agents."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import agenticcybersense.agents.web as web_module
from agenticcybersense.agents.documentation import DocumentationAgent
from agenticcybersense.agents.telegram.telegram import TelegramAgent
from agenticcybersense.agents.web import WebAgent
from agenticcybersense.graph.build_graph import _determine_pending_agents
from agenticcybersense.query_analysis import analyze_query, query_matches_text
from agenticcybersense.schemas.messages import AgentRequest
from agenticcybersense.settings import settings
from agenticcybersense.web_crawler import rag_ingest

if TYPE_CHECKING:
    import pytest


EXPECTED_WEB_RAG_RESULTS = 8


def test_web_agent_refines_observable_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web queries should be reduced to the observable plus breach intent terms."""
    raw_query = "could you check eo@gmail.com email adress had a leakage on website that you strored"
    captured: dict[str, str] = {}

    def fake_query_webcrawler_rag(query: str, n_results: int = 5) -> list[dict[str, object]]:
        assert n_results == EXPECTED_WEB_RAG_RESULTS
        captured["query"] = query
        return [
            {
                "content": "eo@gmail.com appears in a breach report.",
                "url": "https://example.com/breach",
                "title": "Example breach report",
                "site_url": "https://example.com",
                "last_updated": "2026-05-25",
                "score": 0.91,
            },
        ]

    monkeypatch.setattr(rag_ingest, "query_webcrawler_rag", fake_query_webcrawler_rag)

    response = asyncio.run(WebAgent().process(AgentRequest(query=raw_query)))

    assert response.success is True
    assert captured["query"] != raw_query
    assert "eo@gmail.com" in captured["query"]
    assert "breach" in captured["query"]
    assert "leak" in captured["query"]


def test_telegram_agent_matches_email_breach_prompts() -> None:
    """Telegram filtering should match messages containing the extracted observable."""
    raw_query = "could you check eo@gmail.com email adress had a leakage on website that you strored"

    assert query_matches_text(
        analyze_query(raw_query),
        "New breach exposes eo@gmail.com credentials on a criminal forum.",
    )


def test_web_agent_discards_irrelevant_semantic_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observable lookups should not return pages that omit the requested observable."""
    raw_query = "could you check eo@gmail.com email adress had a leakage on website that you strored"

    def fake_query_webcrawler_rag(_query: str, n_results: int = 5) -> list[dict[str, object]]:
        assert n_results == EXPECTED_WEB_RAG_RESULTS
        return [
            {
                "content": "General write-up about phishing and account security.",
                "url": "https://example.com/phishing",
                "title": "Phishing guide",
                "site_url": "https://example.com",
                "last_updated": "2026-05-25",
                "score": 0.97,
            },
        ]

    monkeypatch.setattr(rag_ingest, "query_webcrawler_rag", fake_query_webcrawler_rag)

    response = asyncio.run(WebAgent().process(AgentRequest(query=raw_query)))

    assert response.success is True
    assert response.findings == []
    assert "No direct matches found" in response.content


def test_documentation_agent_abstains_for_observable_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documentation RAG should not run for exact observable breach lookups."""

    async def fail_if_called(_self: DocumentationAgent, query: str) -> str:
        message = f"unexpected RAG lookup for {query}"
        raise AssertionError(message)

    raw_query = "could you check eo@gmail.com email adress had a leakage on website that you strored"
    monkeypatch.setattr(DocumentationAgent, "_retrieve_context", fail_if_called)

    response = asyncio.run(DocumentationAgent().process(AgentRequest(query=raw_query)))

    assert response.success is True
    assert "Observable/Breach Lookup" in response.content
    assert "No documentation-specific context retrieved" in response.content


def test_graph_routes_observable_lookups_to_telegram() -> None:
    """Observable lookups should consult Telegram as well as web."""
    raw_query = 'could you check if my mail adress is seen on web: "eray.umut100@gmail.com"'

    agents = _determine_pending_agents(raw_query)

    assert "web" in agents
    assert "telegram" in agents


def test_graph_routes_ddos_website_queries_to_telegram() -> None:
    """Threat-activity website prompts should consult Telegram as well as web."""
    raw_query = "do you feel any ddos attack to specifially infotech academy website"

    agents = _determine_pending_agents(raw_query)

    assert agents == ["web", "telegram"]


def test_web_report_lists_checked_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """The web report should tell the user which monitored sources were checked."""
    raw_query = 'could you check if my mail adress is seen on web: "eray.umut100@gmail.com"'

    def fake_query_webcrawler_rag(_query: str, n_results: int = 5) -> list[dict[str, object]]:
        assert n_results == EXPECTED_WEB_RAG_RESULTS
        return []

    monkeypatch.setattr(rag_ingest, "query_webcrawler_rag", fake_query_webcrawler_rag)
    monkeypatch.setattr(web_module, "_get_monitored_web_sources", lambda: ("otx.alienvault.com", "misp-project.org"))

    response = asyncio.run(WebAgent().process(AgentRequest(query=raw_query)))

    assert response.success is True
    assert "Checked Web Sources (2)" in response.content
    assert "otx.alienvault.com" in response.content
    assert "misp-project.org" in response.content


def test_telegram_report_lists_checked_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telegram reports should show the channels that were checked."""
    raw_query = 'could you check if my mail adress is seen on web: "eray.umut100@gmail.com"'

    monkeypatch.setattr(settings, "tg_api_id", None)
    monkeypatch.setattr(settings, "tg_api_hash", None)

    agent = TelegramAgent(
        target_groups=[
            {"name": "vx-underground", "id": "@vxunderground", "type": "threat_intel"},
            {"name": "FalconFeedsIO", "id": "@falconfeedsio", "type": "threat_intel"},
        ],
    )
    response = asyncio.run(agent.process(AgentRequest(query=raw_query)))

    assert response.success is True
    assert "Checked Telegram Channels" in response.content
    assert "@vxunderground" in response.content
    assert "@falconfeedsio" in response.content

"""Regression tests for the Telegram agent."""

import asyncio
from unittest.mock import AsyncMock, patch

from agenticcybersense.agents.telegram import TelegramAgent
from agenticcybersense.schemas.messages import AgentRequest


async def _run_agent_with_mocked_fetch(query: str) -> str:
    """Run TelegramAgent.process with mocked channel fetches and return report text."""
    agent = TelegramAgent()

    async def _mock_fetch_channel_messages(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "channel": {"name": "Mock Channel", "id": "@mock", "type": "threat_intel"},
            "timestamp": "2026-01-01T00:00:00+00:00",
            "messages": [
                {
                    "id": 1,
                    "text": "General cyber incident update for Infotech Academy.",
                    "text_preview": "General cyber incident update for Infotech Academy.",
                    "date": "2026-01-01T00:00:00+00:00",
                    "matched_keywords": [],
                    "message_url": "https://t.me/mock/1",
                },
            ],
            "status": "monitored",
            "error": None,
        }

    with patch.object(agent, "_fetch_channel_messages", AsyncMock(side_effect=_mock_fetch_channel_messages)):
        response = await agent.process(AgentRequest(query=query))
    return response.content


def test_generic_leakage_query_is_not_treated_as_credential_investigation() -> None:
    """Generic leakage wording should not trigger credential-investigation mode."""
    content = asyncio.run(
        _run_agent_with_mocked_fetch(
            "could you check telegram channels if there is a cyber leakage about infotech academy",
        ),
    )

    assert "Credential Leak Investigation Result" not in content


def test_explicit_credential_leak_query_is_treated_as_credential_investigation() -> None:
    """Explicit credential wording should trigger credential-investigation mode."""
    content = asyncio.run(
        _run_agent_with_mocked_fetch(
            "check telegram channels for leaked credentials about infotech academy",
        ),
    )

    assert "Credential Leak Investigation Result" in content

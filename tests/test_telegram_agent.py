"""Regression tests for the Telegram agent."""

from agenticcybersense.agents.telegram import TelegramAgent


def test_generic_leakage_query_is_not_treated_as_credential_investigation() -> None:
    agent = TelegramAgent()

    assert not agent._is_credential_investigation_query(
        "could you check telegram channels if there is a cyber leakage about infotech academy",
    )


def test_explicit_credential_leak_query_is_treated_as_credential_investigation() -> None:
    agent = TelegramAgent()

    assert agent._is_credential_investigation_query(
        "check telegram channels for leaked credentials about infotech academy",
    )
"""Parser module for Telegram messages.

This module provides utilities for parsing, normalizing, and extracting information
from Telegram message objects. It includes functions for building message URLs,
extracting text previews, matching keywords and CVE identifiers, and normalizing
message data into structured dictionaries.

The module uses regex patterns to identify CVE identifiers and keyword matches,
with special handling for zero-day vulnerability references.
"""

import re
from datetime import UTC
from typing import Any

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)


def build_message_url(channel: str, message_id: int) -> str:
    """Build a Telegram message URL from a channel identifier and message ID.

    Args:
        channel: The Telegram channel name or identifier (may include '@' prefix).
        message_id: The unique identifier of the message within the channel.

    Returns:
        A fully-formed Telegram message URL in the format https://t.me/{channel}/{message_id}.

    """
    return f"https://t.me/{channel.lstrip('@')}/{message_id}"


def text_preview_from_msg(msg: Any) -> str:  # noqa: ANN401
    """Extract and return a preview of text from a message object.

    This function attempts to retrieve text content from a message object by checking
    for 'message' and 'raw_text' attributes in order of preference. If neither attribute
    exists or contains a value, an empty string is returned.

    Args:
        msg: A message object that may contain 'message' or 'raw_text' attributes.

    Returns:
        A string containing the text preview, stripped of leading and trailing whitespace.
        Returns an empty string if no text content is found.

    """
    text = getattr(msg, "message", None) or getattr(msg, "raw_text", None) or ""
    return (text or "").strip()


def match_keywords(text: str, keywords: list[str]) -> list[str]:
    """Match keywords in text and extract CVE identifiers.

    Searches for keyword matches in the provided text with word boundary checks.
    Handles special cases for zero-day vulnerabilities and automatically extracts
    CVE identifiers matching the CVE_RE pattern.

    Args:
        text: The text to search within.
        keywords: A list of keywords to match in the text.

    Returns:
        A sorted list of matched keywords and CVE identifiers found in the text.

    """
    if not text:
        return []
    tl = text.lower()
    found = set()
    for k in keywords:
        if not k:
            continue
        kk = k.lower().strip()
        if kk in ("0day", "zero-day", "zero day"):
            if re.search(r"\b(?:0day|zero[\s-]?day)\b", tl):
                found.add("zero-day")
        elif re.search(r"\b" + re.escape(kk) + r"\b", tl):
            found.add(k)
    for m in CVE_RE.finditer(text):
        found.add(m.group(0).upper())
    return sorted(found)


def normalize_message(msg: Any, channel_username: str, keywords: list[str]) -> dict[str, Any]:  # noqa: ANN401
    """Normalize a Telegram message into a structured dictionary.

    Args:
        msg: The raw Telegram message object to normalize.
        channel_username: The username of the Telegram channel where the message was posted.
        keywords: A list of keywords to match against the message text.

    Returns:
        A dictionary containing normalized message data with the following keys:
            - channel: The channel username.
            - message_id: The unique message identifier (int or None).
            - date_utc: ISO format UTC timestamp of the message (str or None).
            - matched_keywords: List of keywords matched in the message text.
            - text_preview: First 500 characters of message text with newlines replaced by spaces.
            - message_url: URL link to the message (str or None).
            - raw: The original unmodified message object.

    """
    text = text_preview_from_msg(msg)
    matched = match_keywords(text, keywords)
    date = getattr(msg, "date", None)
    date_iso = date.astimezone(UTC).isoformat() if date else None
    mid = getattr(msg, "id", None) or getattr(msg, "message_id", None)
    return {
        "channel": channel_username,
        "message_id": int(mid) if mid is not None else None,
        "date_utc": date_iso,
        "matched_keywords": matched,
        "text_preview": (text[:500].replace("\n", " ") if text else ""),
        "message_url": build_message_url(channel_username, int(mid)) if mid else None,
        "raw": msg,
    }

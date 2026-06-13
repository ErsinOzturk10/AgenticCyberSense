"""Parser module for Telegram messages.

This module provides utilities for parsing, normalizing, and extracting information
from Telegram message objects. It includes functions for building message URLs,
extracting full text, creating text previews, matching keywords and CVE identifiers,
and normalizing message data into structured dictionaries.

The module uses regex patterns to identify CVE identifiers and keyword matches,
with special handling for zero-day vulnerability references.
"""

from __future__ import annotations

import re
from datetime import UTC
from typing import Any

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)


def build_message_url(channel: str, message_id: int) -> str:
    """Build a Telegram message URL from a channel identifier and message ID.

    Args:
        channel: The Telegram channel name or identifier, may include '@' prefix.
        message_id: The unique identifier of the message within the channel.

    Returns:
        A fully formed Telegram message URL in the format:
        https://t.me/{channel}/{message_id}

    """
    return f"https://t.me/{channel.lstrip('@')}/{message_id}"


def text_from_msg(msg: Any) -> str:  # noqa: ANN401
    """Extract full text content from a Telegram message object.

    Telethon message objects commonly expose message text through either
    the 'message' attribute or the 'raw_text' attribute. This function returns
    the full available text without truncating it.

    Args:
        msg: A Telegram message object that may contain 'message' or 'raw_text'.

    Returns:
        Full message text as a stripped string. Returns an empty string if no
        text content is available.

    """
    text = getattr(msg, "message", None) or getattr(msg, "raw_text", None) or ""
    return (text or "").strip()


def text_preview_from_msg(msg: Any, max_chars: int = 500) -> str:  # noqa: ANN401
    """Extract a short preview from a Telegram message object.

    Args:
        msg: A Telegram message object that may contain 'message' or 'raw_text'.
        max_chars: Maximum number of characters to include in the preview.

    Returns:
        A preview string with newlines replaced by spaces.

    """
    text = text_from_msg(msg)
    return text[:max_chars].replace("\n", " ") if text else ""


def match_keywords(text: str, keywords: list[str]) -> list[str]:
    """Match keywords in text and extract CVE identifiers.

    Searches for keyword matches in the provided text with word boundary checks.
    Handles special cases for zero-day vulnerability references and automatically
    extracts CVE identifiers matching the CVE_RE pattern.

    Args:
        text: The text to search within.
        keywords: A list of keywords to match in the text.

    Returns:
        A sorted list of matched keywords and CVE identifiers found in the text.

    """
    if not text:
        return []

    tl = text.lower()
    found: set[str] = set()

    for k in keywords:
        if not k:
            continue

        original_keyword = k.strip()
        kk = original_keyword.lower()

        if not kk:
            continue

        if kk in {"0day", "zero-day", "zero day"}:
            if re.search(r"\b(?:0day|zero[\s-]?day)\b", tl):
                found.add("zero-day")
            continue

        # Keep word-boundary matching for normal keywords so short keywords do
        # not accidentally match unrelated words.
        if re.search(r"\b" + re.escape(kk) + r"\b", tl):
            found.add(original_keyword)

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
            - message_id: The unique message identifier.
            - date_utc: ISO format UTC timestamp of the message.
            - matched_keywords: List of keywords matched in the full message text.
            - text: Full message text without truncation.
            - text_preview: First 500 characters of message text.
            - message_url: URL link to the message.
            - raw: The original unmodified message object.

    """
    text = text_from_msg(msg)
    matched = match_keywords(text, keywords)

    date = getattr(msg, "date", None)
    date_iso = date.astimezone(UTC).isoformat() if date else None

    mid = getattr(msg, "id", None) or getattr(msg, "message_id", None)
    message_id = int(mid) if mid is not None else None

    return {
        "channel": channel_username,
        "message_id": message_id,
        "date_utc": date_iso,
        "matched_keywords": matched,
        "text": text,
        "text_preview": text[:500].replace("\n", " ") if text else "",
        "message_url": build_message_url(channel_username, message_id) if message_id else None,
        "raw": msg,
    }
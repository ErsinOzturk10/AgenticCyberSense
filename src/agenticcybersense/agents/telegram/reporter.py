"""LLM-based Telegram reporter (adapted from your local telegram_analyze script).

Functional API:
- summarize_rows(rows: list[dict]) -> dict

No file I/O. It uses settings.ollama_base_url and settings.ollama_model.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import requests  # type: ignore[import-untyped]

from agenticcybersense.settings import settings

logger = logging.getLogger(__name__)

MODEL_NAME = settings.ollama_model
OLLAMA_URL = settings.ollama_base_url.rstrip("/") + "/api/generate"

RETRY_COUNT = 3
RETRY_BACKOFF = 3  # exponential base
CVE_REGEX = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
URL_USERNAME_RE = re.compile(r"https?://t\.me/([^/]+)/?\d*", re.IGNORECASE)
URL_TME_C_RE = re.compile(r"https?://(?:t\.me|telegram\.me)/c/(\d+)/(\d+)", re.IGNORECASE)


def _extract_first_json_object(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    if s.startswith("{") and s.endswith("}"):
        return s
    if s.startswith("["):
        return s
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s


def parse_llm_json_text(llm_text: str) -> tuple[Any | None, str]:
    """Try to extract and parse the first JSON object from LLM output.

    Returns a tuple (obj_or_None, method_string).
    """
    candidate = _extract_first_json_object(llm_text)
    if not candidate:
        return None, "empty"

    # Try plain json.loads first
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.debug("json.loads on candidate failed: %s", e)
    else:
        return obj, "json.loads"

    # Try replacing newlines with escaped newlines and parse again
    fixed = candidate.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    try:
        obj = json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.debug("json.loads on escaped candidate failed: %s", e)
        return None, f"failed_to_parse: {type(e).__name__}: {e}"
    else:
        return obj, "escaped_newlines_then_json.loads"


def call_ollama_with_retries(prompt: str, attempts: int = RETRY_COUNT) -> str:
    """Call local Ollama endpoint with retries and return plain text response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    session = requests.Session()
    session.trust_env = False
    last_err: Exception | None = None

    for i in range(1, attempts + 1):
        try:
            resp = session.post(OLLAMA_URL, json=payload, timeout=(10, 900))
            resp.raise_for_status()
            obj = resp.json()
            text = obj.get("response") or ""
            return text.strip()
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = e
            sleep = RETRY_BACKOFF**i
            logger.warning(
                "Warning: Ollama call failed (attempt %d/%d): %s. Retrying in %ds...",
                i,
                attempts,
                e,
                sleep,
            )
            time.sleep(sleep)

    # All attempts failed
    if last_err is not None:
        raise last_err
    msg = "Ollama call failed without an exception"
    raise RuntimeError(msg)


def _all_text_blob(rows: list[dict[str, Any]]) -> str:
    return "\n".join((r.get("text_preview") or "").strip() for r in rows)


def extract_username_from_url(url: str) -> str | None:
    """Extract a username from a Telegram URL.

    Args:
        url: A Telegram URL string to extract the username from.

    Returns:
        A string containing the username prefixed with '@' if found,
        None if the URL is empty, is a t.me/c/... URL (which contains
        numeric IDs instead of usernames), or no username is found.

    """
    if not url:
        return None
    if URL_TME_C_RE.search(url):
        # t.me/c/... URLs don't contain a username; they contain numeric ids
        return None
    m = URL_USERNAME_RE.search(url)
    if m:
        return "@" + m.group(1)
    return None


def sanitize_report(llm_text: str, fallback_rows: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: PLR0915, PLR0912, C901
    """Sanitize and normalize raw LLM JSON text into a stable findings dict."""
    parsed, method = parse_llm_json_text(llm_text)

    data: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "findings": [],
    }

    # Normalize parsed -> findings_raw list
    if isinstance(parsed, list):
        findings_raw = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("findings"), list):
            findings_raw = parsed.get("findings", [])
            gen = parsed.get("generated_at_utc")
            if isinstance(gen, str) and gen.strip():
                data["generated_at_utc"] = gen.strip()
        else:
            findings_raw = [parsed]
    else:
        findings_raw = []

    def fallback_channels() -> list[str]:
        return sorted(
            {(r.get("channel") or "").strip() for r in fallback_rows if (r.get("channel") or "").strip()},
        )

    def first_url_from_rows() -> str:
        for r in fallback_rows:
            u = (r.get("message_url") or "").strip()
            if u.startswith("http"):
                return u
        return ""

    fb_channels = fallback_channels()
    fb_url = first_url_from_rows()

    text_blob = _all_text_blob(fallback_rows)
    cves_in_input = {m.group(0).upper() for m in CVE_REGEX.finditer(text_blob)}
    blob_l = text_blob.lower()
    input_mentions_poc = any(x in blob_l for x in (" poc", "proof of concept", "p.o.c"))
    input_mentions_0day = any(x in blob_l for x in ("0day", "zero-day", "zero day"))
    input_mentions_exploited = any(x in blob_l for x in ("exploited", "in the wild", "actively exploited", "being exploited"))

    normalized: list[dict[str, Any]] = []
    for f in findings_raw:
        if not isinstance(f, dict):
            continue

        title = str(f.get("title") or "").strip() or "Untitled finding"
        channels = [str(c).strip() for c in (f.get("channels") or []) if str(c).strip()]
        if not channels:
            channels = fb_channels

        severity = str(f.get("severity") or "medium").lower().strip()
        if severity not in ("low", "medium", "high"):
            severity = "medium"

        exploit_status = str(f.get("exploit_status") or "unknown").lower().strip()
        if exploit_status not in ("unknown", "poc", "exploited", "0day"):
            exploit_status = "unknown"
        if exploit_status == "poc" and not input_mentions_poc:
            exploit_status = "unknown"
        if exploit_status == "0day" and not input_mentions_0day:
            exploit_status = "unknown"
        if exploit_status == "exploited" and not input_mentions_exploited:
            exploit_status = "unknown"

        cve_val = f.get("cve")
        if cve_val is None:
            cve_out = None
        else:
            cve_s = str(cve_val).strip().upper()
            cve_out = cve_s if cve_s in cves_in_input else None

        source_urls = [str(u).strip() for u in (f.get("source_message_urls") or []) if str(u).strip()]
        if not source_urls and fb_url:
            source_urls = [fb_url]

        normalized.append(
            {
                "title": title,
                "channels": channels,
                "severity": severity,
                "why_it_matters": str(f.get("why_it_matters") or "").strip(),
                "key_technical_details": str(
                    f.get("key_technical_details") or "",
                ).strip(),
                "cve": cve_out,
                "exploit_status": exploit_status,
                "source_message_urls": source_urls,
                "evidence_quotes": [str(q).strip() for q in (f.get("evidence_quotes") or []) if str(q).strip()],
            },
        )

    # Fill up to 5 with placeholders (before dedupe)
    max_findings = 5
    while len(normalized) < max_findings:
        normalized.append(
            {
                "title": f"Finding {len(normalized) + 1}",
                "channels": fb_channels,
                "severity": "medium",
                "why_it_matters": "",
                "key_technical_details": "",
                "cve": None,
                "exploit_status": "unknown",
                "source_message_urls": [fb_url] if fb_url else [],
                "evidence_quotes": [],
            },
        )

    # Simple dedupe: merge by title or first URL
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in normalized:
        key_title = (item.get("title") or "").strip().lower()
        first_url = (item.get("source_message_urls") or [None])[0] or ""
        key = key_title or ("url:" + first_url)

        if key not in merged:
            merged[key] = item
            order.append(key)
        else:
            e = merged[key]
            for c in item.get("channels") or []:
                if c and c not in e["channels"]:
                    e["channels"].append(c)
            for u in item.get("source_message_urls") or []:
                if u and u not in e["source_message_urls"]:
                    e["source_message_urls"].append(u)

            # severity: keep the higher severity
            sev_order = {"low": 0, "medium": 1, "high": 2}
            if sev_order.get(item.get("severity", "medium"), 1) > sev_order.get(
                e.get("severity", "medium"),
                1,
            ):
                e["severity"] = item["severity"]

    deduped = [merged[k] for k in order]

    # If we can extract a username from the URL, normalize the channel to that username
    for d in deduped:
        uname = None
        for u in d.get("source_message_urls", []):
            candidate = extract_username_from_url(u)
            if candidate:
                uname = candidate
                break
        if uname:
            d["channels"] = [uname]

    data["findings"] = deduped[:5]
    data["_parse_method"] = method
    return data


def build_prompt_for_rows(rows: list[dict[str, Any]]) -> str:
    """Build LLM prompt from normalized rows."""
    return f"""
You are a CTI analyst. You will be given Telegram messages that matched security keywords.
Return ONLY valid JSON (no markdown, no commentary) with fields:
generated_at_utc, findings[ title, channels, severity, why_it_matters, key_technical_details, cve, exploit_status, source_message_urls, evidence_quotes ].
Input messages:
{json.dumps(rows, ensure_ascii=False, indent=2)}
""".strip()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize normalized rows and return sanitized findings dict."""
    if not rows:
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "findings": [],
        }

    prompt = build_prompt_for_rows(rows)

    llm_text = call_ollama_with_retries(prompt)
    return sanitize_report(llm_text, rows)

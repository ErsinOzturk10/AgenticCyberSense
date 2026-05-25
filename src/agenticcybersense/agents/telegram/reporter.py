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

from agenticcybersense.llm import generate_text

logger = logging.getLogger(__name__)


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


def call_llm_with_retries(prompt: str, attempts: int = RETRY_COUNT) -> str:
    """Call configured LLM with retries and return plain text response."""
    last_err: Exception | None = None

    for i in range(1, attempts + 1):
        try:
            return generate_text(prompt, temperature=0.2)
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = RETRY_BACKOFF**i
            logger.warning(
                "Warning: LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                i,
                attempts,
                e,
                sleep,
            )
            time.sleep(sleep)

    if last_err is not None:
        raise last_err

    msg = "LLM call failed without an exception"
    raise RuntimeError(msg)


def _row_text(r: dict[str, Any]) -> str:
    """Return the best-effort text content from a row.

    Supports both schema variants:
    - parser.normalize_message() rows: 'text_preview'
    - TelegramAgent.process() rows: 'text'
    """
    return (r.get("text_preview") or r.get("text") or "").strip()


def _all_text_blob(rows: list[dict[str, Any]]) -> str:
    return "\n".join(_row_text(r) for r in rows if _row_text(r))


def extract_username_from_url(url: str) -> str | None:
    """Extract a username from a Telegram URL."""
    if not url:
        return None
    if URL_TME_C_RE.search(url):
        return None
    m = URL_USERNAME_RE.search(url)
    if m:
        return "@" + m.group(1)
    return None


def _extract_cves_from_text(text: str) -> list[str]:
    """Extract unique CVEs from text, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in CVE_REGEX.finditer(text or ""):
        c = m.group(0).upper()
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _pick_cve_for_finding(
    finding: dict[str, Any],
    input_cves: list[str],
    cves_in_input: set[str],
) -> str | None:
    """Best-effort deterministic CVE selection for a finding.

    Priority:
    1) Any CVE mentioned in evidence_quotes/title/details/why_it_matters.
    2) If exactly one CVE exists in the input, use it.
    3) Otherwise None (avoid wrong assignment).
    """
    parts: list[str] = []
    parts.append(str(finding.get("title") or ""))
    parts.append(str(finding.get("why_it_matters") or ""))
    parts.append(str(finding.get("key_technical_details") or ""))
    parts.extend([str(q or "") for q in finding.get("evidence_quotes") or []])

    blob = "\n".join(parts)
    cves_in_finding = _extract_cves_from_text(blob)
    for c in cves_in_finding:
        if c in cves_in_input:
            return c

    if len(input_cves) == 1:
        return input_cves[0]

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
    input_cves = _extract_cves_from_text(text_blob)
    cves_in_input = set(input_cves)

    blob_l = text_blob.lower()
    input_mentions_poc = any(x in blob_l for x in (" poc", "proof of concept", "p.o.c"))
    input_mentions_0day = any(x in blob_l for x in ("0day", "zero-day", "zero day"))
    input_mentions_exploited = any(x in blob_l for x in ("exploited", "in the wild", "actively exploited", "being exploited"))

    normalized: list[dict[str, Any]] = []
    for f in findings_raw:
        if not isinstance(f, dict):
            continue

        title = str(f.get("title") or "").strip() or "Untitled finding"

        channels_val = f.get("channels")
        if isinstance(channels_val, str):
            channels = [channels_val.strip()] if channels_val.strip() else []
        else:
            channels = [str(c).strip() for c in (channels_val or []) if str(c).strip()]
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

        why = str(f.get("why_it_matters") or "").strip()
        details = str(f.get("key_technical_details") or "").strip()

        cve_val = f.get("cve")
        if cve_val is None:
            cve_out = None
        else:
            cve_s = str(cve_val).strip().upper()
            cve_out = cve_s if cve_s in cves_in_input else None

        source_urls = [str(u).strip() for u in (f.get("source_message_urls") or []) if str(u).strip()]
        if not source_urls and fb_url:
            source_urls = [fb_url]

        evidence_quotes = [str(q).strip() for q in (f.get("evidence_quotes") or []) if str(q).strip()]

        item = {
            "title": title,
            "channels": channels,
            "severity": severity,
            "why_it_matters": why,
            "key_technical_details": details,
            "cve": cve_out,
            "exploit_status": exploit_status,
            "source_message_urls": source_urls,
            "evidence_quotes": evidence_quotes,
        }

        # Deterministic CVE fill: if LLM didn't set it, try to infer safely.
        if item["cve"] is None and cves_in_input:
            picked = _pick_cve_for_finding(item, input_cves=input_cves, cves_in_input=cves_in_input)
            if picked:
                item["cve"] = picked

        normalized.append(item)

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

            sev_order = {"low": 0, "medium": 1, "high": 2}
            if sev_order.get(item.get("severity", "medium"), 1) > sev_order.get(e.get("severity", "medium"), 1):
                e["severity"] = item["severity"]

            # keep cve if one has it and the other doesn't
            if not e.get("cve") and item.get("cve"):
                e["cve"] = item["cve"]

    deduped = [merged[k] for k in order]

    # Normalize channel to username if URL contains it
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
You are a CTI analyst.

TASK
- You will be given Telegram messages (JSON array). Extract up to 5 actionable findings.

OUTPUT RULES (STRICT)
- Return ONLY a single JSON object. No markdown. No commentary. No code fences.
- Must be valid JSON parseable by json.loads().
- Do NOT include trailing commas.
- If you cannot find any meaningful security finding, return:
  {{"generated_at_utc":"<iso8601>","findings":[]}}

FIELD RULES
- findings is a list (0..5 items).
- severity must be one of: "low" | "medium" | "high".
- exploit_status must be one of: "unknown" | "poc" | "exploited" | "0day".
- cve must be either null OR an EXACT CVE string found in the input text (e.g., "CVE-2022-29498").
- If a CVE appears in the input, you MUST include it in cve for the relevant finding (do not set it to null).

SCHEMA
{{
  "generated_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "findings": [
    {{
      "title": "string",
      "channels": ["string"],
      "severity": "low|medium|high",
      "why_it_matters": "string",
      "key_technical_details": "string",
      "cve": "CVE-YYYY-NNNN..." or null,
      "exploit_status": "unknown|poc|exploited|0day",
      "source_message_urls": ["string"],
      "evidence_quotes": ["string"]
    }}
  ]
}}

EXAMPLE (follow this structure exactly)
{{
  "generated_at_utc": "2026-04-24T00:00:00Z",
  "findings": [
    {{
      "title": "Blazer SQL injection mentioned",
      "channels": ["@CVE_Feed"],
      "severity": "high",
      "why_it_matters": "SQL injection can lead to data exposure or worse depending on context.",
      "key_technical_details": "Blazer before 2.6.0 allows SQL Injection under certain circumstances.",
      "cve": "CVE-2022-29498",
      "exploit_status": "unknown",
      "source_message_urls": [],
      "evidence_quotes": ["CVE-2022-29498", "SQL Injection"]
    }}
  ]
}}

INPUT MESSAGES (JSON)
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
    llm_text = call_llm_with_retries(prompt)
    return sanitize_report(llm_text, rows)

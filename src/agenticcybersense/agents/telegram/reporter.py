"""LLM-based Telegram reporter (adapted from your local telegram_analyze script).

Functional API:
- summarize_rows(rows: list[dict]) -> dict

No file I/O. It uses settings.ollama_base_url and settings.ollama_model.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from agenticcybersense.settings import settings

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


def parse_llm_json_text(llm_text: str) -> Tuple[Optional[Any], str]:
    candidate = _extract_first_json_object(llm_text)
    if not candidate:
        return None, "empty"
    try:
        obj = json.loads(candidate)
        return obj, "json.loads"
    except Exception:
        pass

    fixed = candidate.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    try:
        obj = json.loads(fixed)
        return obj, "escaped_newlines_then_json.loads"
    except Exception as e:
        return None, f"failed_to_parse: {type(e).__name__}: {e}"


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
    last_err: Optional[Exception] = None

    for i in range(1, attempts + 1):
        try:
            resp = session.post(OLLAMA_URL, json=payload, timeout=(10, 900))
            resp.raise_for_status()
            obj = resp.json()
            text = obj.get("response") or ""
            return text.strip()
        except Exception as e:
            last_err = e
            sleep = RETRY_BACKOFF ** i
            print(f"Warning: Ollama call failed (attempt {i}/{attempts}): {e}. Retrying in {sleep}s...")
            time.sleep(sleep)

    # All attempts failed
    raise last_err  # type: ignore[misc]


def _all_text_blob(rows: List[Dict[str, Any]]) -> str:
    return "\n".join((r.get("text_preview") or "").strip() for r in rows)


def extract_username_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    if URL_TME_C_RE.search(url):
        # t.me/c/... URL'lerinde username yok, numeric id var
        return None
    m = URL_USERNAME_RE.search(url)
    if m:
        return "@" + m.group(1)
    return None


def sanitize_report(llm_text: str, fallback_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize and normalize raw LLM JSON text into a stable findings dict."""
    from datetime import datetime, timezone

    parsed, method = parse_llm_json_text(llm_text)

    data: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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

    def fallback_channels() -> List[str]:
        return sorted(
            {
                (r.get("channel") or "").strip()
                for r in fallback_rows
                if (r.get("channel") or "").strip()
            },
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
    input_mentions_exploited = any(
        x in blob_l
        for x in ("exploited", "in the wild", "actively exploited", "being exploited")
    )

    normalized: List[Dict[str, Any]] = []
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

        source_urls = [
            str(u).strip()
            for u in (f.get("source_message_urls") or [])
            if str(u).strip()
        ]
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
                "evidence_quotes": [
                    str(q).strip()
                    for q in (f.get("evidence_quotes") or [])
                    if str(q).strip()
                ],
            },
        )

    # Placeholder ile 5’e tamamla (dedupe öncesi)
    while len(normalized) < 5:
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

    # Basit dedupe: title veya ilk URL’ye göre birleştir
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
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

            # severity: daha yüksek olanı koru
            sev_order = {"low": 0, "medium": 1, "high": 2}
            if sev_order.get(item.get("severity", "medium"), 1) > sev_order.get(
                e.get("severity", "medium"),
                1,
            ):
                e["severity"] = item["severity"]

    deduped = [merged[k] for k in order]

    # URL’den username türetebilirsek kanalı normalize et
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


def build_prompt_for_rows(rows: List[Dict[str, Any]]) -> str:
    """Build LLM prompt from normalized rows."""
    prompt = f"""
You are a CTI analyst. You will be given Telegram messages that matched security keywords.
Return ONLY valid JSON (no markdown, no commentary) with fields:
generated_at_utc, findings[ title, channels, severity, why_it_matters, key_technical_details, cve, exploit_status, source_message_urls, evidence_quotes ].
Input messages:
{json.dumps(rows, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main entrypoint: takes normalized rows (list of dicts) and returns sanitized findings dict."""
    from datetime import datetime, timezone

    if not rows:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "findings": [],
        }

    prompt = build_prompt_for_rows(rows)
    llm_text = call_ollama_with_retries(prompt)
    return sanitize_report(llm_text, rows)
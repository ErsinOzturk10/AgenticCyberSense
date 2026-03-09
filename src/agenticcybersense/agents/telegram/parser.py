import re
from datetime import timezone
from typing import Any, List, Dict

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

def build_message_url(channel: str, message_id: int) -> str:
    return f"https://t.me/{channel.lstrip('@')}/{message_id}"

def text_preview_from_msg(msg: Any) -> str:
    text = getattr(msg, "message", None) or getattr(msg, "raw_text", None) or ""
    return (text or "").strip()

def match_keywords(text: str, keywords: List[str]) -> List[str]:
    if not text:
        return []
    tl = text.lower()
    found = set()
    for k in keywords:
        if not k:
            continue
        kk = k.lower().strip()
        if kk in ("0day","zero-day","zero day"):
            if re.search(r"\b(?:0day|zero[\s-]?day)\b", tl):
                found.add("zero-day")
        else:
            if re.search(r"\b" + re.escape(kk) + r"\b", tl):
                found.add(k)
    for m in CVE_RE.finditer(text):
        found.add(m.group(0).upper())
    return sorted(found)

def normalize_message(msg: Any, channel_username: str, keywords: List[str]) -> Dict[str, Any]:
    text = text_preview_from_msg(msg)
    matched = match_keywords(text, keywords)
    date = getattr(msg, "date", None)
    date_iso = date.astimezone(timezone.utc).isoformat() if date else None
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

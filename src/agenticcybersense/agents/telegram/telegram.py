"""Telegram Intelligence Agent - Monitors Telegram for threat intelligence."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.agents.telegram.client import TelegramClientWrapper
from agenticcybersense.agents.telegram.parser import CVE_RE, normalize_message
from agenticcybersense.agents.telegram.telegram_channels import TELEGRAM_CHANNELS
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse
from agenticcybersense.settings import settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@register_agent
class TelegramAgent(BaseAgent):
    """Telegram intelligence agent for monitoring groups and channels."""

    name: str = "telegram"
    description: str = "Monitors Telegram groups and channels for leaked data and threat actor activity"
    MIN_QUERY_TERM_LENGTH: ClassVar[int] = 3
    MAX_EMAIL_LOCAL_VISIBLE_CHARS: ClassVar[int] = 2

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialize the Telegram agent."""
        super().__init__(llm=llm)
        self.target_groups = TELEGRAM_CHANNELS

    def _empty_channel_result(
        self,
        channel: dict[str, str],
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Return an empty channel result."""
        return {
            "channel": channel,
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": [],
            "status": status,
            "error": error,
        }

    def _extract_requested_channel_id(self, query: str) -> str | None:
        """Extract a Telegram @channel identifier from the user query."""
        m = re.search(r"@[\w\d_]+", query or "")
        return m.group(0) if m else None

    def _is_raw_channel_retrieval_query(self, query: str) -> bool:
        """Detect requests that ask to retrieve messages without CTI filtering."""
        q = (query or "").lower()

        raw_retrieval_phrases = [
            "all messages",
            "retrieve all",
            "get all messages",
            "fetch all messages",
            "without any missing",
            "without missing",
            "no missing parts",
            "full messages",
            "complete messages",
            "entire channel",
            "all channels",
            "all telegram channels",
        ]

        return any(phrase in q for phrase in raw_retrieval_phrases)

    def _is_credential_investigation_query(self, query: str) -> bool:
        """Detect questions asking whether usernames, passwords, or credentials were leaked."""
        q = (query or "").lower()

        credential_terms = [
            "credential",
            "credentials",
            "username",
            "usernames",
            "user name",
            "user names",
            "email",
            "emails",
            "password",
            "passwords",
            "passwd",
            "pwd",
            "login",
            "logins",
            "account",
            "accounts",
            "combo",
            "combolist",
            "combo list",
            "stealer log",
            "stealer logs",
            "leaked",
            "leak",
            "compromised",
        ]

        telegram_terms = ["telegram", "channel", "channels"]

        has_credential_intent = any(term in q for term in credential_terms)
        has_telegram_context = any(term in q for term in telegram_terms) or self._extract_requested_channel_id(query) is not None

        return has_credential_intent and has_telegram_context

    def _credential_indicators(self, text: str) -> list[str]:
        """Return credential-related indicators detected in a message."""
        if not text:
            return []

        indicators: set[str] = set()

        if re.search(r"\b[\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}\b", text):
            indicators.add("email_or_username")

        if re.search(r"\b(?:user(?:name)?|login|email|account)\s*[:=]\s*\S+", text, flags=re.IGNORECASE):
            indicators.add("username_field")

        if re.search(r"\b(?:pass(?:word)?|passwd|pwd|pass)\s*[:=]\s*\S+", text, flags=re.IGNORECASE):
            indicators.add("password_field")

        if re.search(
            r"(?im)^(\s*)([\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}|[\w.\-+%@]{3,})(\s*[:;|]\s*)(\S{4,})(\s*)$",
            text,
        ):
            indicators.add("username_password_pair")

        if re.search(r"\burl\s*[:=]\s*\S+", text, flags=re.IGNORECASE) and re.search(
            r"\b(?:login|user(?:name)?|email|pass(?:word)?)\s*[:=]\s*\S+",
            text,
            flags=re.IGNORECASE,
        ):
            indicators.add("stealer_log_style_entry")

        return sorted(indicators)

    def _looks_like_credential_leak(self, text: str) -> bool:
        """Detect username/password-style credential leak content."""
        return bool(self._credential_indicators(text))

    def _message_is_credential_match(self, msg: dict[str, Any]) -> bool:
        """Return True if a normalized message appears to contain credential material."""
        text = msg.get("text", "") or ""
        matched_keywords = [str(k).lower() for k in (msg.get("matched_keywords") or [])]

        credential_keywords = {
            "credential",
            "credentials",
            "username",
            "user",
            "email",
            "password",
            "passwd",
            "pwd",
            "login",
            "account",
            "combo",
            "combolist",
            "leaked credentials",
            "compromised credentials",
            "stealer logs",
            "ulp",
        }

        return self._looks_like_credential_leak(text) or any(k in credential_keywords for k in matched_keywords)

    def _credential_indicator_counts(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Count credential indicator types across matched messages."""
        counts = {
            "email_or_username": 0,
            "username_field": 0,
            "password_field": 0,
            "username_password_pair": 0,
            "stealer_log_style_entry": 0,
        }

        for msg in messages:
            for indicator in self._credential_indicators(msg.get("text", "") or ""):
                if indicator in counts:
                    counts[indicator] += 1

        return counts

    def _mask_email(self, value: str) -> str:
        """Mask an email address while preserving minimal context."""
        value = value.strip()
        if "@" not in value:
            return self._mask_username(value)

        local, domain = value.split("@", 1)
        if not local:
            return f"***@{domain}"

        masked_local = local[0] + "***" if len(local) <= self.MAX_EMAIL_LOCAL_VISIBLE_CHARS else local[: self.MAX_EMAIL_LOCAL_VISIBLE_CHARS] + "***"

        return f"{masked_local}@{domain}"

    def _mask_username(self, value: str) -> str:
        """Mask a username-like value without revealing it fully."""
        value = value.strip()
        if not value:
            return "[REDACTED_USERNAME]"

        return f"[REDACTED_USERNAME len={len(value)}]"

    def _mask_password(self, value: str) -> str:
        """Mask a password-like value without revealing it."""
        value = value.strip()
        if not value:
            return "[REDACTED_PASSWORD]"

        return f"[REDACTED_PASSWORD len={len(value)}]"

    def _mask_identity_value(self, value: str) -> str:
        """Mask either email-like or username-like values."""
        value = value.strip()
        if re.fullmatch(r"[\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}", value):
            return self._mask_email(value)
        return self._mask_username(value)

    def _redact_sensitive_credentials(self, text: str) -> str:
        """Mask credential values before displaying evidence."""
        if not text:
            return ""

        redacted = text

        pair_re = re.compile(
            r"(?im)^(\s*)([\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}|[\w.\-+%@]{3,})(\s*[:;|]\s*)(\S{4,})(\s*)$",
        )

        def pair_repl(match: re.Match[str]) -> str:
            username = self._mask_identity_value(match.group(2))
            password = self._mask_password(match.group(4))
            return f"{match.group(1)}{username}{match.group(3)}{password}{match.group(5)}"

        redacted = pair_re.sub(pair_repl, redacted)

        identity_field_re = re.compile(
            r"(?im)\b((?:user(?:name)?|login|email|account)\s*[:=]\s*)([^\s,;]+)",
        )

        def identity_field_repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{self._mask_identity_value(match.group(2))}"

        redacted = identity_field_re.sub(identity_field_repl, redacted)

        password_field_re = re.compile(
            r"(?im)\b((?:pass(?:word)?|passwd|pwd|pass)\s*[:=]\s*)([^\s,;]+)",
        )

        def password_field_repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{self._mask_password(match.group(2))}"

        redacted = password_field_re.sub(password_field_repl, redacted)

        standalone_email_re = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}\b")
        return standalone_email_re.sub(lambda m: self._mask_email(m.group(0)), redacted)

    def _credential_summary_lines(self, text: str) -> list[str]:
        """Extract masked credential summary lines for a message."""
        if not text:
            return []

        lines: list[str] = []

        pair_re = re.compile(
            r"(?im)^(\s*)([\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}|[\w.\-+%@]{3,})(\s*[:;|]\s*)(\S{4,})(\s*)$",
        )
        lines.extend(
            [f"- Username/password pair: {self._mask_identity_value(m.group(2))} : {self._mask_password(m.group(4))}" for m in pair_re.finditer(text)],
        )

        email_re = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[a-zA-Z]{2,}\b")
        lines.extend(f"- Email/username-like value: {self._mask_email(m.group(0))}" for m in email_re.finditer(text))

        identity_field_re = re.compile(
            r"(?im)\b(?:user(?:name)?|login|email|account)\s*[:=]\s*([^\s,;]+)",
        )
        lines.extend(f"- Username-like field: {self._mask_identity_value(m.group(1))}" for m in identity_field_re.finditer(text))

        password_field_re = re.compile(
            r"(?im)\b(?:pass(?:word)?|passwd|pwd|pass)\s*[:=]\s*([^\s,;]+)",
        )
        lines.extend(f"- Password-like field: {self._mask_password(m.group(1))}" for m in password_field_re.finditer(text))

        # Keep order but remove duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for line in lines:
            if line not in seen:
                seen.add(line)
                deduped.append(line)

        return deduped

    def _channel_selected_for_credential_check(
        self,
        channel: dict[str, str],
        *,
        is_credential_investigation: bool,
        requested_channel_id: str | None,
    ) -> bool:
        """Return whether this channel should be considered for credential investigation."""
        return not (is_credential_investigation and requested_channel_id and channel.get("id", "").lower() != requested_channel_id.lower())

    def _is_message_relevant(
        self,
        *,
        msg: dict[str, Any],
        msg_text_raw: str,
        query_terms: list[str],
        channel_type: str,
        query_mode: str,
    ) -> bool:
        """Determine whether a Telegram message is relevant for the current query mode."""
        msg_text = msg_text_raw.lower()

        if query_mode == "cve":
            return bool(CVE_RE.search(msg_text_raw))

        if query_mode == "credential":
            return self._message_is_credential_match(msg)

        return (
            query_mode == "raw"
            or bool(msg.get("matched_keywords"))
            or any(word in msg_text for word in query_terms)
            or channel_type in ["breach", "threat_intel", "credential_leak"]
            or self._looks_like_credential_leak(msg_text_raw)
        )

    def _determine_message_severity(
        self,
        *,
        msg_text: str,
        msg_text_raw: str,
        is_credential_investigation: bool,
    ) -> Severity:
        """Assign severity to a message based on indicators and keywords."""
        critical_keywords = ["critical", "rce", "zero-day", "0day", "actively exploited", "breach"]
        high_keywords = [
            "high",
            "vulnerability",
            "exploit",
            "apt",
            "ransomware",
            "credential",
            "credentials",
            "password",
            "passwd",
            "combolist",
            "stealer log",
            "stealer logs",
        ]
        medium_keywords = ["medium", "phishing", "malware", "suspicious", "username", "login", "account"]

        if is_credential_investigation or self._looks_like_credential_leak(msg_text_raw):
            return Severity.HIGH
        if any(kw in msg_text for kw in critical_keywords):
            return Severity.CRITICAL
        if any(kw in msg_text for kw in high_keywords):
            return Severity.HIGH
        if any(kw in msg_text for kw in medium_keywords):
            return Severity.MEDIUM
        return Severity.LOW

    def _build_finding_message_data(
        self,
        *,
        msg: dict[str, Any],
        msg_text_raw: str,
        indicators: list[str],
        is_credential_investigation: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Build finding description and raw data with credential-aware masking."""
        if is_credential_investigation or indicators:
            masked_text = self._redact_sensitive_credentials(msg_text_raw)
            return (
                masked_text[:200],
                {
                    **msg,
                    "text": masked_text,
                    "text_preview": masked_text[:500].replace("\n", " ") if masked_text else "",
                    "credential_indicators": indicators,
                    "credential_summary": self._credential_summary_lines(msg_text_raw),
                },
            )

        return (msg_text_raw[:200], msg)

    async def _fetch_channel_messages(
        self,
        channel: dict[str, str],
        limit: int = 10,
        client: TelegramClientWrapper | None = None,
    ) -> dict[str, Any]:
        """Fetch messages from a Telegram channel."""
        self.logger.info("Checking channel: %s", channel["name"])

        if client is None:
            self.logger.info(
                "Telegram client is not configured; returning empty result for %s",
                channel["name"],
            )
            return self._empty_channel_result(channel, status="not_configured")

        try:
            self.logger.debug("Using Telethon client for channel %s", channel["id"])
            msgs = await client.fetch_channel_messages(channel_username=channel["id"], limit=limit)

            keywords = [s.strip() for s in (settings.telegram_keywords or "").split(",") if s.strip()]

            normalized_messages = [normalize_message(m, channel_username=channel["id"], keywords=keywords) for m in msgs]

            messages = [
                {
                    "id": m.get("message_id"),
                    "text": m.get("text", "") or m.get("text_preview", ""),
                    "text_preview": m.get("text_preview", ""),
                    "date": m.get("date_utc"),
                    "matched_keywords": m.get("matched_keywords", []),
                    "message_url": m.get("message_url"),
                }
                for m in normalized_messages
            ]

            return {
                "channel": channel,
                "timestamp": datetime.now(UTC).isoformat(),
                "messages": messages,
                "status": "monitored",
                "error": None,
            }

        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "Telegram fetch failed for %s: %s",
                channel["name"],
                e,
            )
            return self._empty_channel_result(channel, status="fetch_failed", error=str(e))

    async def _analyze_messages(self, query: str, results: list[dict[str, Any]]) -> list[Finding]:
        """Analyze messages for relevant threat intelligence."""
        findings: list[Finding] = []
        query_lower = (query or "").lower().strip()
        query_terms = [word for word in query_lower.split() if len(word) >= self.MIN_QUERY_TERM_LENGTH]
        is_cve_query = query_lower.startswith("cve-")
        is_raw_retrieval = self._is_raw_channel_retrieval_query(query)
        is_credential_investigation = self._is_credential_investigation_query(query)
        requested_channel_id = self._extract_requested_channel_id(query)
        query_mode = "cve" if is_cve_query else "credential" if is_credential_investigation else "raw" if is_raw_retrieval else "default"

        for result in results:
            channel = result["channel"]
            messages = result["messages"]

            if not self._channel_selected_for_credential_check(
                channel,
                is_credential_investigation=is_credential_investigation,
                requested_channel_id=requested_channel_id,
            ):
                continue

            for msg in messages:
                msg_text_raw = msg.get("text", "") or ""
                msg_text = msg_text_raw.lower()

                if not self._is_message_relevant(
                    msg=msg,
                    msg_text_raw=msg_text_raw,
                    query_terms=query_terms,
                    channel_type=channel.get("type", "unknown"),
                    query_mode=query_mode,
                ):
                    continue

                severity = self._determine_message_severity(
                    msg_text=msg_text,
                    msg_text_raw=msg_text_raw,
                    is_credential_investigation=is_credential_investigation,
                )

                indicators = self._credential_indicators(msg_text_raw)
                finding_description, raw_data_for_finding = self._build_finding_message_data(
                    msg=msg,
                    msg_text_raw=msg_text_raw,
                    indicators=indicators,
                    is_credential_investigation=is_credential_investigation,
                )

                findings.append(
                    Finding(
                        title=f"Telegram: {channel['name']}",
                        description=finding_description,
                        severity=severity,
                        source=SourceRef(
                            source_type=SourceType.TELEGRAM,
                            name=channel["name"],
                            metadata={
                                "channel_id": channel["id"],
                                "message_id": msg.get("id"),
                                "credential_indicators": indicators,
                            },
                        ),
                        tags=["telegram", "osint", channel.get("type", "unknown")],
                        raw_data=raw_data_for_finding,
                    ),
                )

        return findings

    def _message_matches_query(self, query: str, text: str) -> bool:
        """Hard filter messages by the user query."""
        q = (query or "").strip()
        if not q:
            return True

        t = text or ""
        tl = t.lower()
        ql = q.lower()

        cve_keywords = [
            "cve",
            "cves",
            "vulnerability",
            "vulnerabilities",
            "exploit",
            "exploits",
            "poc",
            "proof of concept",
            "weaponization",
            "weaponized",
        ]

        credential_keywords = [
            "credential",
            "credentials",
            "username",
            "usernames",
            "email",
            "emails",
            "password",
            "passwords",
            "passwd",
            "pwd",
            "login",
            "account",
            "combolist",
            "combo list",
            "stealer log",
            "stealer logs",
            "ulp",
            "leaked credential",
            "compromised credential",
        ]

        matches_query = False

        # Check CVE prefix match
        if ql.startswith("cve-"):
            m = re.match(r"^cve-(\d{4})", ql)
            if m:
                year = m.group(1)
                matches_query = re.search(rf"\bCVE-{year}-\d+\b", t, flags=re.IGNORECASE) is not None
            else:
                matches_query = CVE_RE.search(t) is not None

        # Check CVE keywords
        elif any(term in ql for term in cve_keywords):
            cve_indicators = [
                "cve",
                "vulnerability",
                "vulnerabilities",
                "cvss",
                "rce",
                "sql injection",
                "xss",
                "zero-day",
                "zero day",
                "0day",
                "exploit",
                "exploited",
                "poc",
                "proof of concept",
                "weaponized",
                "weaponization",
            ]
            matches_query = CVE_RE.search(t) is not None or any(term in tl for term in cve_indicators)

        # Check credential keywords
        elif any(term in ql for term in credential_keywords):
            credential_indicators = [
                "credential",
                "credentials",
                "username",
                "email",
                "password",
                "passwd",
                "pwd",
                "login",
                "account",
                "combolist",
                "combo",
                "stealer",
                "ulp",
            ]
            matches_query = self._looks_like_credential_leak(t) or any(term in tl for term in credential_indicators)

        # Check 0day/zero-day variations
        elif ql in {"0day", "zero day", "zero-day"}:
            matches_query = re.search(r"\b(?:0day|zero[\s-]?day)\b", tl) is not None

        # Default text matching
        else:
            matches_query = ql in tl

        return matches_query

    async def process(self, request: AgentRequest) -> AgentResponse:  # noqa: PLR0912, C901, PLR0915
        """Process a Telegram intelligence query."""
        query = request.query or ""
        self.logger.info("Telegram agent processing: %s", query[:100])

        results: list[dict[str, Any]] = []

        tg_api_id = settings.tg_api_id
        tg_api_hash = settings.tg_api_hash

        requested_channel_id = self._extract_requested_channel_id(query)
        is_raw_retrieval = self._is_raw_channel_retrieval_query(query)
        is_credential_investigation = self._is_credential_investigation_query(query)

        channels_to_fetch = self.target_groups
        fetch_limit = 100 if (is_raw_retrieval or is_credential_investigation) else 10

        if tg_api_id and tg_api_hash:
            try:
                async with TelegramClientWrapper(
                    api_id=tg_api_id,
                    api_hash=tg_api_hash,
                    session_name=settings.tg_session_name,
                ) as tg_client:
                    results.extend(
                        [
                            await self._fetch_channel_messages(
                                channel,
                                limit=fetch_limit,
                                client=tg_client,
                            )
                            for channel in channels_to_fetch
                        ],
                    )
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Telegram client initialization failed: %s", e)
                results.extend(
                    self._empty_channel_result(
                        channel,
                        status="client_unavailable",
                        error=str(e),
                    )
                    for channel in channels_to_fetch
                )
        else:
            results.extend(self._empty_channel_result(channel, status="not_configured") for channel in channels_to_fetch)

        findings = await self._analyze_messages(query, results)

        all_messages: list[dict[str, Any]] = []
        all_messages.extend(
            {
                "id": msg.get("id"),
                "text": msg.get("text", ""),
                "text_preview": msg.get("text_preview", ""),
                "date": msg.get("date"),
                "matched_keywords": msg.get("matched_keywords", []),
                "message_url": msg.get("message_url"),
                "channel": result["channel"]["name"],
                "channel_id": result["channel"]["id"],
                "channel_type": result["channel"].get("type", "unknown"),
            }
            for result in results
            for msg in result.get("messages", [])
        )

        if is_raw_retrieval:
            matched_messages = all_messages
        elif is_credential_investigation:
            candidate_messages = all_messages

            if requested_channel_id:
                candidate_messages = [m for m in candidate_messages if str(m.get("channel_id", "")).lower() == requested_channel_id.lower()]

            matched_messages = [m for m in candidate_messages if self._message_is_credential_match(m)]
        else:
            matched_messages = [m for m in all_messages if self._message_matches_query(query, m.get("text", ""))]

        if not matched_messages and not is_raw_retrieval:
            findings = []

        if is_credential_investigation:
            matched_ids = {m.get("id") for m in matched_messages}
            matched_channel_ids = {str(m.get("channel_id", "")).lower() for m in matched_messages}

            findings = [
                f
                for f in findings
                if str(f.source.metadata.get("channel_id", "")).lower() in matched_channel_ids and f.source.metadata.get("message_id") in matched_ids
            ]

        llm_report: dict[str, Any] = {}

        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        findings.sort(key=lambda f: severity_order.get(f.severity, 5))

        response_parts: list[str] = [
            "### 📱 Telegram Intelligence Report\n",
            f"**Query:** {query}\n",
            f"**Channels Monitored:** {len(results)}\n",
            f"**Messages Analyzed:** {sum(len(r['messages']) for r in results)}\n",
            f"**Relevant Findings:** {len(findings)}\n",
            f"**Query Matches (messages):** {len(matched_messages)}\n\n",
        ]

        response_parts.append("**Channel Summary:**\n")
        for result in results:
            channel = result["channel"]
            msg_count = len(result["messages"])
            status = result.get("status", "unknown")
            response_parts.append(f"- **{channel['name']}** ({channel['id']}): {msg_count} messages, status={status}\n")

        response_parts.append("\n**Recent Intelligence:**\n")

        if not matched_messages:
            if is_credential_investigation:
                target = requested_channel_id or "configured Telegram channels"
                response_parts.append(
                    f"**Credential Leak Investigation Result:** No username/password-like leaked credential content "
                    f"was detected for **{target}** in the current fetch window.\n",
                )
            else:
                response_parts.append("No messages matched the query in the last fetch window.\n")
        elif is_raw_retrieval:
            if requested_channel_id:
                response_parts.append(
                    f"Raw retrieval requested. Mentioned channel: **{requested_channel_id}**. "
                    "Returning messages from **all configured Telegram channels**.\n\n",
                )
            else:
                response_parts.append(
                    "Raw retrieval requested. Returning messages from **all configured Telegram channels**.\n\n",
                )

            for idx, msg in enumerate(matched_messages, start=1):
                display_text = msg.get("text") or ""
                if self._looks_like_credential_leak(display_text):
                    display_text = self._redact_sensitive_credentials(display_text)

                response_parts.append(f"**Message {idx}**\n")
                response_parts.append(f"- **Channel:** {msg.get('channel')} ({msg.get('channel_id')})\n")
                response_parts.append(f"- **Date:** {msg.get('date') or 'N/A'}\n")
                response_parts.append(f"- **URL:** {msg.get('message_url') or 'N/A'}\n")
                response_parts.append(f"- **Matched Keywords:** {', '.join(msg.get('matched_keywords') or []) or 'None'}\n")
                response_parts.append("```text\n")
                response_parts.append(display_text.strip())
                response_parts.append("\n```\n\n")

        elif is_credential_investigation:
            target = requested_channel_id or "configured Telegram channels"
            indicator_counts = self._credential_indicator_counts(matched_messages)

            response_parts.append(
                f"**Credential Leak Investigation Result:** Yes. Username/password-like credential content was detected for **{target}**.\n\n",
            )

            response_parts.append("**Detected Indicator Summary:**\n")
            response_parts.append(f"- Email/username-like values: {indicator_counts.get('email_or_username', 0)}\n")
            response_parts.append(f"- Username fields: {indicator_counts.get('username_field', 0)}\n")
            response_parts.append(f"- Password fields: {indicator_counts.get('password_field', 0)}\n")
            response_parts.append(f"- Username/password pair lines: {indicator_counts.get('username_password_pair', 0)}\n")
            response_parts.append(f"- Stealer-log style entries: {indicator_counts.get('stealer_log_style_entry', 0)}\n\n")

            response_parts.append(
                "Credential values are partially masked or redacted in the UI.\n\n",
            )

            for idx, msg in enumerate(matched_messages, start=1):
                raw_text = msg.get("text", "") or ""
                indicators = self._credential_indicators(raw_text)
                summary_lines = self._credential_summary_lines(raw_text)
                display_text = self._redact_sensitive_credentials(raw_text)

                response_parts.append(f"**Credential Evidence {idx}**\n")
                response_parts.append(f"- **Channel:** {msg.get('channel')} ({msg.get('channel_id')})\n")
                response_parts.append(f"- **Date:** {msg.get('date') or 'N/A'}\n")
                response_parts.append(f"- **URL:** {msg.get('message_url') or 'N/A'}\n")
                response_parts.append(f"- **Indicators:** {', '.join(indicators) or 'credential-like content'}\n")
                response_parts.append(f"- **Matched Keywords:** {', '.join(msg.get('matched_keywords') or []) or 'None'}\n")

                if summary_lines:
                    response_parts.append("- **Masked Extracted Values:**\n")
                    response_parts.extend(f"  {line}\n" for line in summary_lines)

                response_parts.append("```text\n")
                response_parts.append(display_text.strip())
                response_parts.append("\n```\n\n")

        else:
            if llm_report and llm_report.get("findings"):
                response_parts.append("\n**🤖 LLM Analysis:**\n")
                for f in llm_report["findings"]:
                    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f.get("severity", ""), "⚪")
                    response_parts.append(f"{severity_emoji} **{f.get('title', 'N/A')}**\n")
                    response_parts.append(f"   - **Why it matters:** {f.get('why_it_matters', 'N/A')}\n")
                    response_parts.append(f"   - **CVE:** {f.get('cve') or 'N/A'}\n")
                    response_parts.append(f"   - **Exploit Status:** {f.get('exploit_status', 'unknown')}\n\n")

            if findings:
                for finding in findings[:5]:
                    emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                        "info": "🔵",
                    }.get(finding.severity.value, "⚪")
                    response_parts.append(f"{emoji} **[{finding.severity.value.upper()}]** {finding.title}\n")
                    response_parts.append(f"   {finding.description[:100]}...\n\n")
            else:
                response_parts.append("No specific threats matching your query were found in monitored channels.\n")

        has_messages = any(len(r.get("messages", [])) > 0 for r in results)
        has_fetch_failures = any(r.get("status") in {"fetch_failed", "client_unavailable"} for r in results)
        not_configured = all(r.get("status") == "not_configured" for r in results)

        if not_configured:
            monitoring_status = "Not configured"
        elif has_fetch_failures and has_messages:
            monitoring_status = "Active (partial)"
        elif has_fetch_failures:
            monitoring_status = "Fetch failed"
        elif has_messages:
            monitoring_status = "Active"
        else:
            monitoring_status = "Active (no messages)"

        response_parts.append("\n**Monitoring Status:**\n")
        response_parts.append(f"- Telegram monitoring: {monitoring_status}\n")
        response_parts.append("- Alert threshold: Medium and above\n")
        response_parts.append(f"- Last check: {datetime.now(UTC).isoformat()}\n")

        return AgentResponse(
            content="".join(response_parts),
            agent_name=self.name,
            success=True,
            findings=findings,
            metadata={
                "channels_monitored": len(results),
                "messages_analyzed": sum(len(r["messages"]) for r in results),
                "findings_count": len(findings),
                "monitoring_status": monitoring_status,
                "query_matches": len(matched_messages),
                "requested_channel_id": requested_channel_id,
                "raw_channel_retrieval": is_raw_retrieval,
                "credential_investigation": is_credential_investigation,
            },
        )


async def telegram_search(query: str) -> str:
    """Search Telegram threat intelligence channels."""
    agent = TelegramAgent()
    response = await agent.process(AgentRequest(query=query))
    return response.content

"""Telegram Intelligence Agent - Monitors Telegram for threat intelligence."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.agents.telegram.client import TelegramClientWrapper
from agenticcybersense.agents.telegram.parser import CVE_RE, normalize_message
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

    DEFAULT_CHANNELS: ClassVar[list[dict[str, str]]] = [
        {"name": "vx-underground", "id": "@vxunderground", "type": "threat_intel"},
        {"name": "FalconFeedsIO", "id": "@falconfeedsio", "type": "threat_intel"},
        {"name": "Malpedia", "id": "@malpedia", "type": "malware_intel"},
        {"name": "CVE Feed", "id": "@CVE_Feed", "type": "cve"},
        
    ]


    def __init__(self, target_groups: list[dict[str, str]] | None = None, llm: BaseChatModel | None = None) -> None:
        """Initialize the Telegram agent.

        Args:
            target_groups: A list of dictionaries containing target group configurations with string keys and values.
                           Defaults to DEFAULT_CHANNELS if not provided.
            llm: Optional language model to pass to the parent class initializer.

        """
        super().__init__(llm=llm)
        self.target_groups = target_groups or self.DEFAULT_CHANNELS

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
                    "text": m.get("text_preview", ""),
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
        is_cve_query = query_lower.startswith("cve-")

        critical_keywords = ["critical", "rce", "zero-day", "0day", "actively exploited", "breach"]
        high_keywords = ["high", "vulnerability", "exploit", "apt", "ransomware"]
        medium_keywords = ["medium", "phishing", "malware", "suspicious"]

        for result in results:
            channel = result["channel"]
            messages = result["messages"]

            for msg in messages:
                msg_text_raw = msg.get("text", "") or ""
                msg_text = msg_text_raw.lower()

                # For CVE queries, require an actual CVE ID in the message text.
                if is_cve_query:
                    if not CVE_RE.search(msg_text_raw):
                        continue
                else:
                    is_relevant = any(word in msg_text for word in query_lower.split()) or channel.get("type") in [
                        "breach",
                        "threat_intel",
                    ]
                    if not is_relevant:
                        continue

                if any(kw in msg_text for kw in critical_keywords):
                    severity = Severity.CRITICAL
                elif any(kw in msg_text for kw in high_keywords):
                    severity = Severity.HIGH
                elif any(kw in msg_text for kw in medium_keywords):
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                findings.append(
                    Finding(
                        title=f"Telegram: {channel['name']}",
                        description=msg_text_raw[:200],
                        severity=severity,
                        source=SourceRef(
                            source_type=SourceType.TELEGRAM,
                            name=channel["name"],
                            metadata={"channel_id": channel["id"], "message_id": msg.get("id")},
                        ),
                        tags=["telegram", "osint", channel.get("type", "unknown")],
                        raw_data=msg,
                    ),
                )

        return findings

    def _message_matches_query(self, query: str, text: str) -> bool:
        """Hard filter messages by the user query.

        This prevents unrelated summaries when the last N messages don't contain the query.
        """
        q = (query or "").strip()
        if not q:
            return True

        t = text or ""
        tl = t.lower()
        ql = q.lower()

        # Specific CVE queries: match specific year if provided, else any CVE pattern.
        if ql.startswith("cve-"):
            m = re.match(r"^cve-(\d{4})", ql)
            if m:
                year = m.group(1)
                return re.search(rf"\bCVE-{year}-\d+\b", t, flags=re.IGNORECASE) is not None
            return CVE_RE.search(t) is not None

        # General CVE / vulnerability queries.
        if any(
            term in ql
            for term in [
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
        ):
            return CVE_RE.search(t) is not None or any(
                term in tl
                for term in [
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
            )

        # 0day queries.
        if ql in {"0day", "zero day", "zero-day"}:
            return re.search(r"\b(?:0day|zero[\s-]?day)\b", tl) is not None

        # General substring match.
        return ql in tl

    async def process(self, request: AgentRequest) -> AgentResponse:  # noqa: PLR0912, C901, PLR0915
        """Process a Telegram intelligence query."""
        self.logger.info("Telegram agent processing: %s", (request.query or "")[:100])

        results: list[dict[str, Any]] = []

        tg_api_id = settings.tg_api_id
        tg_api_hash = settings.tg_api_hash

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
                                limit=10,
                                client=tg_client,
                            )
                            for channel in self.target_groups
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
                    for channel in self.target_groups
                )
        else:
            results.extend(self._empty_channel_result(channel, status="not_configured") for channel in self.target_groups)

        findings = await self._analyze_messages(request.query, results)

        # Build all_messages without duplication.
        all_messages: list[dict[str, Any]] = []
        all_messages.extend(
            {
                "id": msg.get("id"),
                "text": msg.get("text", ""),
                "date": msg.get("date"),
                "matched_keywords": msg.get("matched_keywords", []),
                "message_url": msg.get("message_url"),
                "channel": result["channel"]["name"],
            }
            for result in results
            for msg in result.get("messages", [])
        )

        # Hard-filter the messages by query before LLM summary.
        matched_messages = [m for m in all_messages if self._message_matches_query(request.query, m.get("text", ""))]

        # Keep report consistent — if nothing matched, don't return findings either.
        if not matched_messages:
            findings = []

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
            f"**Query:** {request.query}\n",
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

        # If nothing matched, don't print unrelated findings/LLM analysis.
        if not matched_messages:
            response_parts.append("No messages matched the query in the last fetch window.\n")
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
            },
        )


async def telegram_search(query: str) -> str:
    """Search Telegram threat intelligence channels."""
    agent = TelegramAgent()
    response = await agent.process(AgentRequest(query=query))
    return response.content

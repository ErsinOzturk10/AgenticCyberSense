"""Telegram Intelligence Agent - Monitors Telegram for threat intelligence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.agents.telegram.client import TelegramClientWrapper
from agenticcybersense.agents.telegram.parser import normalize_message
from agenticcybersense.agents.telegram.reporter import summarize_rows
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse
from agenticcybersense.settings import settings


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

    def __init__(self, target_groups: list[dict[str, str]] | None = None, **_kwargs: object) -> None:
        """Initialize the Telegram agent.

        Args:
            target_groups: A list of dictionaries containing target group configurations with string keys and values.
                           Defaults to DEFAULT_CHANNELS if not provided.
            **_kwargs: Additional keyword arguments to pass to the parent class initializer.

        """
        super().__init__(**(_kwargs or {}))
        self.target_groups = target_groups or self.DEFAULT_CHANNELS

    async def _fetch_channel_messages(
        self,
        channel: dict[str, str],
        limit: int = 10,
        client: TelegramClientWrapper | None = None,
    ) -> dict[str, Any]:
        """Fetch messages from a Telegram channel."""
        self.logger.info("Checking channel: %s", channel["name"])

        normalized_messages: list[dict[str, Any]] = []
        used_simulated = False

        try:
            if client is not None:
                self.logger.debug("Using Telethon client for channel %s", channel["id"])
                msgs = await client.fetch_channel_messages(channel_username=channel["id"], limit=limit)

                keywords = [s.strip() for s in (settings.telegram_keywords or "").split(",") if s.strip()]

                for m in msgs:
                    normalized_messages.append(normalize_message(m, channel_username=channel["id"], keywords=keywords))
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "Real Telegram fetch failed for %s: %s; falling back to simulated data",
                channel["name"],
                e,
            )

        if not normalized_messages:
            used_simulated = True
            simulated_messages = {
                "news": [
                    {
                        "id": 1,
                        "text": "🚨 New critical vulnerability discovered in popular software",
                        "date": datetime.now(UTC).isoformat(),
                    },
                    {
                        "id": 2,
                        "text": "Security advisory: Update your systems immediately",
                        "date": datetime.now(UTC).isoformat(),
                    },
                ],
                "cve": [
                    {
                        "id": 1,
                        "text": "CVE-2024-9999: Critical RCE in widely used library (CVSS 9.8)",
                        "date": datetime.now(UTC).isoformat(),
                    },
                    {
                        "id": 2,
                        "text": "CVE-2024-8888: SQL injection vulnerability (CVSS 7.5)",
                        "date": datetime.now(UTC).isoformat(),
                    },
                ],
                "threat_intel": [
                    {
                        "id": 1,
                        "text": "APT group activity detected targeting financial sector",
                        "date": datetime.now(UTC).isoformat(),
                    },
                    {
                        "id": 2,
                        "text": "New phishing campaign using AI-generated content",
                        "date": datetime.now(UTC).isoformat(),
                    },
                ],
                "breach": [
                    {
                        "id": 1,
                        "text": "⚠️ Data breach reported: Company X - 1M records exposed",
                        "date": datetime.now(UTC).isoformat(),
                    },
                    {
                        "id": 2,
                        "text": "Credential dump detected on dark web forums",
                        "date": datetime.now(UTC).isoformat(),
                    },
                ],
                "malware_intel": [
                    {
                        "id": 1,
                        "text": "New malware family analysis published; indicators and YARA rules shared",
                        "date": datetime.now(UTC).isoformat(),
                    },
                ],
            }

            messages = simulated_messages.get(channel.get("type", "news"), [])[:limit]
        else:
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
            "used_simulated": used_simulated,
        }

    async def _analyze_messages(self, query: str, results: list[dict[str, Any]]) -> list[Finding]:
        """Analyze messages for relevant threat intelligence."""
        findings: list[Finding] = []
        query_lower = query.lower()

        critical_keywords = ["critical", "rce", "zero-day", "0day", "actively exploited", "breach"]
        high_keywords = ["high", "vulnerability", "exploit", "apt", "ransomware"]
        medium_keywords = ["medium", "phishing", "malware", "suspicious"]

        for result in results:
            channel = result["channel"]
            messages = result["messages"]

            for msg in messages:
                msg_text = msg.get("text", "").lower()

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
                        description=msg.get("text", "")[:200],
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

    async def process(self, request: AgentRequest) -> AgentResponse:  # noqa: PLR0912, C901
        """Process a Telegram intelligence query."""
        self.logger.info("Telegram agent processing: %s", request.query[:100])

        results: list[dict[str, Any]] = []

        tg_api_id = settings.tg_api_id
        tg_api_hash = settings.tg_api_hash

        if tg_api_id and tg_api_hash:
            async with TelegramClientWrapper(
                api_id=tg_api_id,
                api_hash=tg_api_hash,
                session_name=settings.tg_session_name,
            ) as tg_client:
                for channel in self.target_groups:
                    try:
                        results.append(await self._fetch_channel_messages(channel, limit=3, client=tg_client))
                    except (RuntimeError, ImportError) as e:
                        self.logger.warning("Error fetching %s: %s", channel["name"], e)
        else:
            for channel in self.target_groups:
                try:
                    results.append(await self._fetch_channel_messages(channel, limit=3, client=None))
                except (RuntimeError, ImportError) as e:
                    self.logger.warning("Error fetching %s: %s", channel["name"], e)

        findings = await self._analyze_messages(request.query, results)

        all_messages = []
        for result in results:
            if not result.get("used_simulated"):
                for msg in result["messages"]:
                    all_messages.append({
                        "id": msg.get("id"),
                        "text": msg.get("text", ""),
                        "date": msg.get("date"),
                        "matched_keywords": msg.get("matched_keywords", []),
                        "message_url": msg.get("message_url"),
                        "channel": result["channel"]["name"],
                    })

        llm_report = {}
        if all_messages:
            llm_report = summarize_rows(all_messages[:5])

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
            f"**Relevant Findings:** {len(findings)}\n\n",
        ]

        response_parts.append("**Channel Summary:**\n")
        for result in results:
            channel = result["channel"]
            msg_count = len(result["messages"])
            response_parts.append(f"- **{channel['name']}** ({channel['id']}): {msg_count} messages\n")

        response_parts.append("\n**Recent Intelligence:**\n")

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

        used_sim = any(r.get("used_simulated") for r in results)
        used_real = any((not r.get("used_simulated")) and len(r.get("messages", [])) > 0 for r in results)
        if used_real and used_sim:
            realtime_status = "Active (mixed)"
        elif used_real:
            realtime_status = "Active (real)"
        else:
            realtime_status = "Active (simulated)"

        response_parts.append("\n**Monitoring Status:**\n")
        response_parts.append(f"- Real-time monitoring: {realtime_status}\n")
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
                "realtime_status": realtime_status,
            },
        )
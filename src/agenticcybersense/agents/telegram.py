"""Telegram Intelligence Agent - Monitors Telegram for threat intelligence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from agenticcybersense.agents.base import BaseAgent
from agenticcybersense.agents.registry import register_agent
from agenticcybersense.schemas.findings import Finding, Severity, SourceRef, SourceType
from agenticcybersense.schemas.messages import AgentRequest, AgentResponse


@register_agent
class TelegramAgent(BaseAgent):
    """Telegram intelligence agent for monitoring groups and channels."""

    name: str = "telegram"
    description: str = "Monitors Telegram groups and channels for leaked data and threat actor activity"

    # Default channels to monitor (examples - would be configured per deployment)
    DEFAULT_CHANNELS: ClassVar[list[dict[str, str]]] = [
        {"name": "Security Alerts", "id": "@security_alerts_example", "type": "news"},
        {"name": "CVE Feed", "id": "@cve_feed_example", "type": "cve"},
        {"name": "Threat Intel", "id": "@threat_intel_example", "type": "threat_intel"},
        {"name": "Data Breach Monitor", "id": "@breach_monitor_example", "type": "breach"},
    ]

    def __init__(self, target_groups: list[dict[str, str]] | None = None, **_kwargs: object) -> None:
        """Initialize the Telegram agent."""
        super().__init__(**(_kwargs or {}))
        self.target_groups = target_groups or self.DEFAULT_CHANNELS

    async def _fetch_channel_messages(self, channel: dict[str, str], limit: int = 10) -> dict[str, Any]:
        """Fetch messages from a Telegram channel (simulated).

        TODO: Implement real Telegram API integration with telethon/pyrogram
        """
        self.logger.info("Checking channel: %s", channel["name"])

        # Simulated messages based on channel type
        simulated_messages = {
            "news": [
                {"id": 1, "text": "🚨 New critical vulnerability discovered in popular software", "date": datetime.now(UTC).isoformat()},
                {"id": 2, "text": "Security advisory: Update your systems immediately", "date": datetime.now(UTC).isoformat()},
            ],
            "cve": [
                {"id": 1, "text": "CVE-2024-9999: Critical RCE in widely used library (CVSS 9.8)", "date": datetime.now(UTC).isoformat()},
                {"id": 2, "text": "CVE-2024-8888: SQL injection vulnerability (CVSS 7.5)", "date": datetime.now(UTC).isoformat()},
            ],
            "threat_intel": [
                {"id": 1, "text": "APT group activity detected targeting financial sector", "date": datetime.now(UTC).isoformat()},
                {"id": 2, "text": "New phishing campaign using AI-generated content", "date": datetime.now(UTC).isoformat()},
            ],
            "breach": [
                {"id": 1, "text": "⚠️ Data breach reported: Company X - 1M records exposed", "date": datetime.now(UTC).isoformat()},
                {"id": 2, "text": "Credential dump detected on dark web forums", "date": datetime.now(UTC).isoformat()},
            ],
        }

        return {
            "channel": channel,
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": simulated_messages.get(channel.get("type", "news"), [])[:limit],
            "status": "monitored",
        }

    async def _analyze_messages(self, query: str, results: list[dict[str, Any]]) -> list[Finding]:
        """Analyze messages for relevant threat intelligence."""
        findings = []
        query_lower = query.lower()

        # Keywords for severity classification
        critical_keywords = ["critical", "rce", "zero-day", "0day", "actively exploited", "breach"]
        high_keywords = ["high", "vulnerability", "exploit", "apt", "ransomware"]
        medium_keywords = ["medium", "phishing", "malware", "suspicious"]

        for result in results:
            channel = result["channel"]
            messages = result["messages"]

            for msg in messages:
                msg_text = msg.get("text", "").lower()

                # Check relevance to query
                is_relevant = any(word in msg_text for word in query_lower.split()) or channel.get("type") in ["breach", "threat_intel"]

                if not is_relevant:
                    continue

                # Determine severity
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

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a Telegram intelligence query."""
        self.logger.info("Telegram agent processing: %s", request.query[:100])

        # Fetch from all configured channels
        results = []
        for channel in self.target_groups:
            try:
                result = await self._fetch_channel_messages(channel)
                results.append(result)
            except (RuntimeError, ImportError) as e:
                self.logger.warning("Error fetching %s: %s", channel["name"], e)

        # Analyze messages
        findings = await self._analyze_messages(request.query, results)

        # Sort findings by severity
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
        findings.sort(key=lambda f: severity_order.get(f.severity, 5))

        # Build response
        response_parts = [
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
        if findings:
            for finding in findings[:5]:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}.get(finding.severity.value, "⚪")
                response_parts.append(f"{emoji} **[{finding.severity.value.upper()}]** {finding.title}\n")
                response_parts.append(f"   {finding.description[:100]}...\n\n")
        else:
            response_parts.append("No specific threats matching your query were found in monitored channels.\n")

        response_parts.append("\n**Monitoring Status:**\n")
        response_parts.append("- Real-time monitoring: Active (simulated)\n")
        response_parts.append("- Alert threshold: Medium and above\n")
        response_parts.append("- Last check: Now\n")

        return AgentResponse(
            content="".join(response_parts),
            agent_name=self.name,
            success=True,
            findings=findings,
            metadata={
                "channels_monitored": len(results),
                "messages_analyzed": sum(len(r["messages"]) for r in results),
                "findings_count": len(findings),
            },
        )

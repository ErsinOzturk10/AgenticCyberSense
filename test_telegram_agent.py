"""Test runner for the TelegramAgent.

This script creates a TelegramAgent instance and sends a test AgentRequest
(with query "rce exploit") to exercise the agent processing pipeline.
It logs the success state, number of findings, and the generated report.
"""

import asyncio
import logging

from agenticcybersense.agents.telegram import TelegramAgent
from agenticcybersense.schemas.messages import AgentRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Execute the main async function to test the TelegramAgent.

    Creates a TelegramAgent instance, processes a request with an "rce exploit" query,
    and logs the success status, number of findings, and the generated report content.
    """
    agent = TelegramAgent()
    req = AgentRequest(query="rce exploit")
    resp = await agent.process(req)

    logger.info("Success: %s", resp.success)
    logger.info("Findings: %d", len(resp.findings))
    logger.info("--- Report ---")
    logger.info("%s", resp.content)


if __name__ == "__main__":
    asyncio.run(main())

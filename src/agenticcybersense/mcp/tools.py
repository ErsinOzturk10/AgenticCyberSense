"""MCP Server Tools."""

import logging
import time
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

from agenticcybersense.agents.telegram.telegram import telegram_search as _telegram_search
from agenticcybersense.rag.rag import rag_search as _rag_search

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("agenticcybersense-mcp-tools-server")


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def safe_log_input(value: str, max_len: int = 300) -> str:
    """Prevent very long or multiline user input from polluting logs."""
    cleaned = (value or "").replace("\n", "\\n").strip()
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "...[truncated]"
    return cleaned


@mcp.tool()
def rag_search(user_input: str) -> str:
    """Mandatory first tool for every user request.

    This tool provides information from uploaded PDFs about Cyber Threat Intelligence.

    Tool orchestration rules:
    - rag_search must be the first tool call for every user request.
    - After rag_search returns, use its output as context for deciding what to do next.
    - After rag_search, decide whether any additional tool is required.
    - Do not produce the final answer until all necessary tool calls are completed.

    If relevant information is not found in the PDFs, return:
    "Insufficient information in the provided documents."
    """
    start_time = time.perf_counter()
    started_at = utc_now_iso()

    logger.info(
        "[TOOL START] name=rag_search started_at_utc=%s input=%s",
        started_at,
        safe_log_input(user_input),
    )

    try:
        result = _rag_search(query=user_input)
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "[TOOL ERROR] name=rag_search started_at_utc=%s ended_at_utc=%s duration_ms=%s",
            started_at,
            utc_now_iso(),
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "[TOOL END] name=rag_search started_at_utc=%s ended_at_utc=%s duration_ms=%s chars=%d",
        started_at,
        utc_now_iso(),
        duration_ms,
        len(result or ""),
    )

    return result


@mcp.tool()
async def telegram_search(user_input: str) -> str:
    """Search Telegram threat intelligence channels for cyber threats.

    Use this tool after rag_search when the user asks about:
    - Recent cyber attacks or threat actor activity on Telegram
    - CVEs or vulnerabilities mentioned in Telegram channels
    - Leaked data, breach news, malware campaigns, APT activity

    If no relevant data is found, return:
    "No relevant Telegram intelligence found."
    """
    start_time = time.perf_counter()
    started_at = utc_now_iso()

    logger.info(
        "[TOOL START] name=telegram_search started_at_utc=%s input=%s",
        started_at,
        safe_log_input(user_input),
    )

    try:
        result = await _telegram_search(query=user_input)
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "[TOOL ERROR] name=telegram_search started_at_utc=%s ended_at_utc=%s duration_ms=%s",
            started_at,
            utc_now_iso(),
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "[TOOL END] name=telegram_search started_at_utc=%s ended_at_utc=%s duration_ms=%s chars=%d",
        started_at,
        utc_now_iso(),
        duration_ms,
        len(result or ""),
    )

    return result

"""Webcrawler periodic scheduler.

Called from api_server.py lifespan.
Runs the full crawl cycle once per day at a configurable time (default 02:00).
After the crawl finishes, the Scrapy pipeline automatically ingests the
updated JSON into ChromaDB so the web agent always has fresh data.

Usage — add to api_server.py lifespan::

    from agenticcybersense.web_crawler.crawler_scheduler import start_scheduler, stop_scheduler

    await start_scheduler(hour=2, minute=0)

Yield:
    await stop_scheduler()

"""

# ruff: noqa: RUF006, TRY401

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from agenticcybersense.web_crawler.config import OUTPUT_FILE

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def run_crawler_and_ingest() -> None:
    """Run the full crawl cycle using Scrapy.

    Calls run_scrapy_crawl() which:
    1. Crawls all configured sites via Scrapy spider (hash-based incremental by default)
    2. Saves results to output/latest_results.json (via JsonExportPipeline)
    3. Ingests the JSON into ChromaDB (via RagIngestPipeline)
    """
    logger.info("=" * 60)
    logger.info("Scheduled Scrapy crawler starting...")
    logger.info("=" * 60)

    try:
        from agenticcybersense.web_crawler.run_scrapy import run_scrapy_crawl  # noqa: PLC0415

        await run_scrapy_crawl()
        logger.info("Scheduled crawl completed successfully")
    except Exception as exc:
        logger.exception("Scheduled crawl failed: %s", exc)


async def start_scheduler(hour: int = 2, minute: int = 0) -> None:
    """Start the APScheduler async scheduler.

    Registers a daily cron job that runs the full crawl cycle.
    If no crawler output JSON is found on startup, an immediate crawl
    is triggered so the RAG index is populated before the first query.

    Args:
        hour: Hour of day to run the crawl (0-23, default 2 = 02:00).
        minute: Minute of the hour (0-59, default 0).

    """
    global _scheduler  # noqa: PLW0603

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        run_crawler_and_ingest,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="webcrawler_job",
        name="Webcrawler + RAG Ingest",
        replace_existing=True,
        misfire_grace_time=3600,  # tolerate up to 1 hour of delay before skipping
    )
    _scheduler.start()
    logger.info("Crawler scheduler started — runs daily at %02d:%02d", hour, minute)

    # If the crawler output JSON does not exist, trigger an immediate crawl.
    # Without the JSON, rag_ingest has nothing to load into ChromaDB.
    if not Path(OUTPUT_FILE).exists():
        logger.info("No crawler output found at %s — triggering immediate crawl...", OUTPUT_FILE)
        asyncio.create_task(run_crawler_and_ingest())
    else:
        logger.info("Crawler output found — waiting for next scheduled run at %02d:%02d", hour, minute)


async def stop_scheduler() -> None:
    """Stop the scheduler gracefully on server shutdown."""
    global _scheduler  # noqa: PLW0603

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Crawler scheduler stopped")
    _scheduler = None

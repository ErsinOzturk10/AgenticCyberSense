"""Webcrawler periodic scheduler.

Called from api_server.py lifespan.
Runs the full crawl cycle once per day at a configurable time (default 02:00).
After the crawl finishes, main_trafilatura.py automatically ingests the
updated JSON into ChromaDB so the web agent always has fresh data.

Usage — add to api_server.py lifespan:

    from agenticcybersense.web_crawler.crawler_scheduler import start_scheduler, stop_scheduler

    await start_scheduler(hour=2, minute=0, run_on_startup=False)

Yield:
    await stop_scheduler()

"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def run_crawler_and_ingest() -> None:
    """Run the full crawl cycle.

    Calls main() from main_trafilatura.py which:
    1. Crawls all configured sites (hash-based incremental by default)
    2. Saves results to output/latest_results.json
    3. Ingests the JSON into ChromaDB via rag_ingest.py
    """
    logger.info("=" * 60)
    logger.info("Scheduled crawler starting...")
    logger.info("=" * 60)

    try:
        from agenticcybersense.web_crawler.main_trafilatura import main  # noqa: PLC0415

        await main()
        logger.info("Scheduled crawl completed successfully")
    except Exception as exc:
        logger.exception("Scheduled crawl failed: %s", exc)


async def start_scheduler(
    hour: int = 2,
    minute: int = 0,
    run_on_startup: bool = False,
) -> None:
    """Start the APScheduler async scheduler.

    Args:
        hour: Hour of day to run the crawl (0-23, default 2 = 02:00).
        minute: Minute of the hour (0-59, default 0).
        run_on_startup: If True, trigger an immediate crawl when called.
                        Useful for development or first-run bootstrapping.

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

    if run_on_startup:
        logger.info("run_on_startup=True — triggering immediate crawl...")
        asyncio.create_task(run_crawler_and_ingest())


async def stop_scheduler() -> None:
    """Stop the scheduler gracefully on server shutdown."""
    global _scheduler  # noqa: PLW0603

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Crawler scheduler stopped")
    _scheduler = None

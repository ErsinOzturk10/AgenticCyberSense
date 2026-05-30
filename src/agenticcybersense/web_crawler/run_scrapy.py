# ruff: noqa: ASYNC240
"""Programmatic Scrapy runner for the AgenticCyberSense crawler.

This module replaces main_trafilatura.py as the crawler entry point.
It runs the CyberSpider using Scrapy's CrawlerRunner within an existing
asyncio event loop, making it compatible with the api_server.py lifespan
and the APScheduler-based crawler_scheduler.

Usage:
    # Standalone
    python -m agenticcybersense.web_crawler.run_scrapy

    # From scheduler or api_server
    await run_scrapy_crawl()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path

from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import asyncioreactor

# Install the asyncio reactor before any other twisted imports.
# This is safe to call multiple times — it will be a no-op if already installed.
with contextlib.suppress(Exception):
    asyncioreactor.install()

from agenticcybersense.web_crawler.config import OUTPUT_FILE

logger = logging.getLogger(__name__)


def _get_scrapy_settings() -> dict:
    """Load Scrapy settings from our settings module."""
    import agenticcybersense.web_crawler.scrapy_crawler.settings as crawler_settings  # noqa: PLC0415

    settings = get_project_settings()

    # Apply all uppercase attributes from our settings module
    for key in dir(crawler_settings):
        if key.isupper():
            settings.set(key, getattr(crawler_settings, key))

    return settings


async def run_scrapy_crawl() -> None:
    """Run the Scrapy CyberSpider crawl.

    This function is designed to be called from an async context
    (e.g., the crawler scheduler or api_server lifespan).
    It runs the spider and blocks until completion.
    """
    from agenticcybersense.web_crawler.scrapy_crawler.spiders.cyber_spider import CyberSpider  # noqa: PLC0415

    logger.info("=" * 80)
    logger.info("SCRAPY CRAWLER STARTING")
    logger.info("   Output: %s", OUTPUT_FILE)
    logger.info("=" * 80)

    settings = _get_scrapy_settings()
    runner = CrawlerRunner(settings)

    # Use Twisted's deferred in the asyncio reactor
    d = runner.crawl(CyberSpider)

    # Convert Twisted Deferred to asyncio Future
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def on_success(result: object) -> None:  # noqa: ARG001
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, None)

    def on_error(failure: object) -> None:
        if not future.done():
            loop.call_soon_threadsafe(
                future.set_exception,
                Exception(f"Scrapy crawl failed: {failure}"),
            )

    d.addCallback(on_success)
    d.addErrback(on_error)

    await future

    logger.info("=" * 80)
    logger.info("SCRAPY CRAWLER COMPLETED")
    logger.info("   Output: %s", OUTPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info("   Size: %.1f MB", size_mb)
    logger.info("=" * 80)


def main() -> None:
    """Run the crawler synchronously (for standalone execution)."""
    logging.basicConfig(level=logging.INFO)

    # Set the Scrapy project settings module
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "agenticcybersense.web_crawler.scrapy_crawler.settings")

    asyncio.run(run_scrapy_crawl())


if __name__ == "__main__":
    main()

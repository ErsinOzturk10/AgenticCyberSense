"""RAG ingest pipeline.

After all items are collected and JSON is written, this pipeline
triggers the ChromaDB ingest so the RAG index is always up to date.
This mirrors the post-crawl ingest from the original main_trafilatura.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scrapy import Spider

    from agenticcybersense.web_crawler.scrapy_crawler.items import CrawlPageItem

logger = logging.getLogger(__name__)


class RagIngestPipeline:
    """Trigger RAG ingest when the spider finishes crawling."""

    def __init__(self) -> None:
        """Initialize item counter."""
        self._item_count = 0

    def process_item(self, item: CrawlPageItem, spider: Spider) -> CrawlPageItem:  # noqa: ARG002
        """Count items (pass-through — actual ingest happens on close)."""
        self._item_count += 1
        return item

    def close_spider(self, spider: Spider) -> None:  # noqa: ARG002
        """Run the RAG ingest after the crawl is complete."""
        if self._item_count == 0:
            logger.info("RagIngestPipeline: no items crawled, skipping ingest")
            return

        logger.info("RagIngestPipeline: updating webcrawler RAG index (%d items crawled)...", self._item_count)
        try:
            from agenticcybersense.web_crawler.rag_ingest import ingest_crawler_json  # noqa: PLC0415

            stats = ingest_crawler_json()
            logger.info("RAG index updated: %s", stats)
        except Exception:
            logger.exception("RAG ingest failed — crawler results are still saved to disk")

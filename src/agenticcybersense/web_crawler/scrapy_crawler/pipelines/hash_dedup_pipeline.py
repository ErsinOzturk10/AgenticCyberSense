"""Hash-based deduplication pipeline.

Uses the existing CrawlHistoryManager to skip pages whose content
has not changed since the last crawl. This preserves the incremental
crawling behavior from the original system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scrapy.exceptions import DropItem

from agenticcybersense.web_crawler.config import ENABLE_INCREMENTAL, FORCE_FULL_CRAWL, HISTORY_FILE
from agenticcybersense.web_crawler.crawl_history_manager import CrawlHistoryManager

if TYPE_CHECKING:
    from scrapy import Spider

    from agenticcybersense.web_crawler.scrapy_crawler.items import CrawlPageItem

logger = logging.getLogger(__name__)


class HashDedupPipeline:
    """Drop items whose content hash matches the stored version."""

    def __init__(self) -> None:
        """Initialize pipeline state."""
        self._history: CrawlHistoryManager | None = None
        self._skipped = 0
        self._processed = 0

    def open_spider(self, spider: Spider) -> None:  # noqa: ARG002
        """Initialize the history manager when the spider starts."""
        if ENABLE_INCREMENTAL and not FORCE_FULL_CRAWL:
            self._history = CrawlHistoryManager(HISTORY_FILE)
            logger.info("HashDedupPipeline: incremental mode enabled (history: %s)", HISTORY_FILE)
        else:
            logger.info("HashDedupPipeline: full crawl mode (no dedup)")

    def close_spider(self, spider: Spider) -> None:  # noqa: ARG002
        """Log dedup statistics when spider finishes."""
        logger.info(
            "HashDedupPipeline stats: processed=%d, skipped=%d",
            self._processed,
            self._skipped,
        )

    def process_item(self, item: CrawlPageItem, spider: Spider) -> CrawlPageItem:  # noqa: ARG002
        """Check content hash and drop unchanged items."""
        if not self._history:
            self._processed += 1
            return item

        url = item.get("url", "")
        content = item.get("main_content", "")
        depth = item.get("depth", 0)

        # Skip dedup for failed extractions (let them pass through for error tracking)
        metadata = item.get("metadata", {})
        if metadata.get("status") == "failed":
            self._processed += 1
            return item

        if not content:
            self._processed += 1
            return item

        # Main page (depth 0): check if deep crawl is needed
        if depth == 0:
            should_crawl, _reason = self._history.should_deep_crawl(url, content)
            if not should_crawl:
                self._skipped += 1
                msg = f"Content unchanged (hash match): {url}"
                raise DropItem(msg)
        else:
            # Subpage: check subpage hash
            should_process, _reason = self._history.should_crawl_subpage(url, content)
            if not should_process:
                self._skipped += 1
                msg = f"Subpage unchanged (hash match): {url}"
                raise DropItem(msg)
            # Update subpage hash for next run
            self._history.update_subpage(url, content)

        self._processed += 1
        return item

"""JSON export pipeline.

Collects crawled items and writes them to the same JSON format as the
original main_trafilatura.py output. This ensures backward compatibility
with rag_ingest.py and any downstream consumers.

Output format (per site):
{
  "https://example.com": {
    "total_pages": 3,
    "duration_seconds": 12,
    "last_updated": "2024-01-01T00:00:00+00:00",
    "crawl_mode": "scrapy",
    "pages": [
      {
        "url": "...",
        "title": "...",
        "main_content": "...",
        "metadata": {...},
        "links": [...],
        "structured_data": {...}
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticcybersense.web_crawler.config import ENABLE_INCREMENTAL, FORCE_FULL_CRAWL, HISTORY_FILE, OUTPUT_FILE
from agenticcybersense.web_crawler.crawl_history_manager import CrawlHistoryManager

if TYPE_CHECKING:
    from scrapy import Spider

    from agenticcybersense.web_crawler.scrapy_crawler.items import CrawlPageItem

logger = logging.getLogger(__name__)


class JsonExportPipeline:
    """Accumulate crawled items and write to JSON on spider close."""

    def __init__(self) -> None:
        """Initialize storage."""
        self._results: dict[str, Any] = {}
        self._start_time: datetime | None = None
        self._history: CrawlHistoryManager | None = None

    def open_spider(self, spider: Spider) -> None:  # noqa: ARG002
        """Load existing results (for incremental updates) and record start time."""
        self._start_time = datetime.now(tz=UTC)

        # Load existing results to merge with new data
        output_path = Path(OUTPUT_FILE)
        if output_path.exists():
            try:
                with output_path.open(encoding="utf-8") as f:
                    self._results = json.load(f)
                logger.info("Loaded %d existing site results", len(self._results))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load existing results: %s", exc)
                self._results = {}

        # Initialize history manager for updating hashes
        if ENABLE_INCREMENTAL and not FORCE_FULL_CRAWL:
            self._history = CrawlHistoryManager(HISTORY_FILE)

    def close_spider(self, spider: Spider) -> None:  # noqa: ARG002
        """Write final JSON output when the spider finishes."""
        output_path = Path(OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

        total_pages = sum(v.get("total_pages", 0) for v in self._results.values())
        logger.info("JSON export complete: %d sites, %d total pages → %s", len(self._results), total_pages, OUTPUT_FILE)

    def process_item(self, item: CrawlPageItem, spider: Spider) -> CrawlPageItem:  # noqa: ARG002
        """Add item to the results dict grouped by site URL."""
        site_url = item.get("site_url", item.get("url", ""))
        url = item.get("url", "")

        # Initialize site entry if needed
        if site_url not in self._results:
            self._results[site_url] = {
                "total_pages": 0,
                "duration_seconds": 0,
                "last_updated": datetime.now(tz=UTC).isoformat(),
                "crawl_mode": "scrapy",
                "pages": [],
            }

        # Build page data in the expected format
        page_data = {
            "url": url,
            "title": item.get("title", ""),
            "main_content": item.get("main_content", ""),
            "metadata": item.get("metadata", {}),
            "links": item.get("links", []),
            "structured_data": item.get("structured_data", {}),
        }

        site_entry = self._results[site_url]
        site_entry["pages"].append(page_data)
        site_entry["total_pages"] = len(site_entry["pages"])
        site_entry["last_updated"] = datetime.now(tz=UTC).isoformat()

        if self._start_time:
            site_entry["duration_seconds"] = (datetime.now(tz=UTC) - self._start_time).seconds

        # Update history hash for main pages
        if self._history and item.get("depth", 0) == 0:
            content = item.get("main_content", "")
            if content:
                metadata = item.get("metadata", {})
                self._history.update_history(
                    site_url,
                    content,
                    site_entry["total_pages"],
                    {
                        "extraction_type": metadata.get("extraction_type"),
                        "method": metadata.get("method"),
                        "link_count": len(item.get("links", [])),
                        "force_crawled": FORCE_FULL_CRAWL,
                    },
                )

        return item

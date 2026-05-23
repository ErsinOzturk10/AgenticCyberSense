"""Trafilatura content extraction pipeline.

Extracts clean text from raw HTML using Trafilatura with BeautifulSoup fallback.
This mirrors the extraction logic from the original TrafilaturaOllamaAgent.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import trafilatura
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from scrapy import Spider

    from agenticcybersense.web_crawler.scrapy_crawler.items import CrawlPageItem

logger = logging.getLogger(__name__)

# Minimum content length to consider extraction successful
MIN_CONTENT_LENGTH = 100


class TrafilaturaPipeline:
    """Extract clean text content from raw HTML using Trafilatura."""

    def process_item(self, item: CrawlPageItem, spider: Spider) -> CrawlPageItem:  # noqa: ARG002
        """Extract text from HTML and populate main_content field."""
        html = item.get("html", "")
        url = item.get("url", "")

        if not html:
            item["metadata"] = {
                "status": "failed",
                "error": "No HTML content",
                "extraction_type": item.get("extraction_type", "general"),
                "method": item.get("method", "unknown"),
            }
            return item

        # Primary extraction with Trafilatura
        content = self._extract_with_trafilatura(html, url)

        # Fallback to BeautifulSoup if Trafilatura returns insufficient content
        if not content or len(content) < MIN_CONTENT_LENGTH:
            content = self._fallback_extraction(html)

        item["main_content"] = content or ""

        # Build metadata
        status = "success" if content and len(content) >= MIN_CONTENT_LENGTH else "partial"
        item["metadata"] = {
            "status": status,
            "extraction_type": item.get("extraction_type", "general"),
            "method": item.get("method", "unknown"),
            "content_length": len(content) if content else 0,
            "link_count": len(item.get("links", [])),
        }

        # Remove raw HTML to save memory (no longer needed downstream)
        item["html"] = ""

        logger.debug(
            "Extracted %d chars from %s (status=%s)",
            len(content) if content else 0,
            url[:60],
            status,
        )
        return item

    def _extract_with_trafilatura(self, html: str, url: str) -> str:
        """Use Trafilatura for content extraction."""
        try:
            return (
                trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_precision=False,
                    url=url,
                )
                or ""
            )
        except Exception:
            logger.exception("Trafilatura extraction failed for %s", url[:60])
            return ""

    def _fallback_extraction(self, html: str) -> str:
        """BeautifulSoup fallback when Trafilatura fails or returns too little."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text)
        except Exception:
            logger.exception("Fallback extraction failed")
            return ""

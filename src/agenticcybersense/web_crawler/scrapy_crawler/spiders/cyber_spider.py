# ruff: noqa: PLC0415
"""Main Scrapy spider for cyber threat intelligence crawling.

Replicates the behavior of the original DeepCrawler + TrafilaturaOllamaAgent:
- Loads URLs from the Excel config file
- Skips blacklisted URLs
- Uses static HTTP for initial fetch
- Falls back to Playwright for JS-heavy sites
- Follows same-domain links up to configured depth limits
- Passes raw HTML to pipelines for extraction and deduplication
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import scrapy
from scrapy_playwright.page import PageMethod

from agenticcybersense.web_crawler.config import BLACKLIST, DEPTH_LIMITS, SITES_FILE
from agenticcybersense.web_crawler.scrapy_crawler.items import CrawlPageItem

if TYPE_CHECKING:
    from collections.abc import Generator

    from scrapy.http import Response

logger = logging.getLogger(__name__)

# Sites that require JavaScript rendering
JS_REQUIRED_INDICATORS = [
    "alienvault.com",
    "virustotal.com",
    "shodan.io",
    "greynoise.io",
    "any.run",
    "criminalip.io",
    "hudsonrock.com",
    "socradar.io",
    "solarwinds.com",
    "att.com",
    "disinfox.com",
    "app.",
    "dashboard.",
    "console.",
    "portal.",
    "/app/",
    "/dashboard/",
    "/console/",
    "/portal/",
]


def _load_urls_from_excel(excel_path: str) -> list[str]:
    """Load URLs from the first column of an Excel file."""
    import pandas as pd  # type: ignore[import-untyped]

    try:
        df = pd.read_excel(excel_path, header=None)
        urls = [str(url).strip() for url in df.iloc[:, 0].dropna() if str(url).startswith("http")]
    except Exception:
        logger.exception("Failed to load URLs from Excel")
        return []
    else:
        logger.info("Loaded %d URLs from %s", len(urls), excel_path)
        return urls


def _requires_javascript(url: str) -> bool:
    """Determine whether a URL likely requires JS rendering."""
    url_lower = url.lower()
    return any(ind in url_lower for ind in JS_REQUIRED_INDICATORS)


def _detect_extraction_type(url: str) -> str:
    """Infer the content extraction mode from a URL."""
    url_lower = url.lower()
    if "github.com" in url_lower:
        return "github"
    if any(kw in url_lower for kw in ["threat", "security", "malware", "attack.mitre"]):
        return "threat_intel"
    return "general"


class CyberSpider(scrapy.Spider):
    """Spider that crawls cyber threat intelligence sources.

    Behavior mirrors the original main_trafilatura.py orchestrator:
    1. Reads URLs from Excel config
    2. Fetches each site (static or Playwright depending on URL)
    3. Follows same-domain links respecting DEPTH_LIMITS
    4. Yields CrawlPageItem for each page to the pipeline chain
    """

    name = "cyber_spider"
    custom_settings: dict[str, Any] = {  # noqa: RUF012
        "DOWNLOADER_MIDDLEWARES": {
            "agenticcybersense.web_crawler.scrapy_crawler.middlewares.RotateUserAgentMiddleware": 400,
            "agenticcybersense.web_crawler.scrapy_crawler.middlewares.PlaywrightFallbackMiddleware": 550,
        },
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize spider with URL list from Excel."""
        super().__init__(*args, **kwargs)
        self.urls = _load_urls_from_excel(SITES_FILE)
        self._visited: set[str] = set()

    def start_requests(self) -> Generator[scrapy.Request, None, None]:
        """Generate initial requests for all configured URLs."""
        for url in self.urls:
            if url in BLACKLIST:
                logger.info("BLACKLISTED — skipped: %s", url)
                continue

            extraction_type = _detect_extraction_type(url)
            use_playwright = _requires_javascript(url)

            meta: dict[str, Any] = {
                "site_url": url,
                "extraction_type": extraction_type,
                "current_depth": 0,
            }

            if use_playwright:
                meta["playwright"] = True
                meta["playwright_include_page"] = False
                meta["playwright_page_methods"] = [
                    PageMethod("wait_for_timeout", 5000),
                    PageMethod("evaluate", "window.scrollTo(0, document.body.scrollHeight)"),
                    PageMethod("wait_for_timeout", 2000),
                    PageMethod("evaluate", "window.scrollTo(0, 0)"),
                    PageMethod("wait_for_timeout", 1000),
                ]

            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta=meta,
                errback=self.handle_error,
                dont_filter=True,
            )

    def parse(self, response: Response) -> Generator[CrawlPageItem | scrapy.Request, None, None]:
        """Parse a page response and follow same-domain links."""
        url = response.url
        site_url = response.meta.get("site_url", url)
        extraction_type = response.meta.get("extraction_type", "general")
        current_depth = response.meta.get("current_depth", 0)

        if url in self._visited:
            return
        self._visited.add(url)

        # Extract same-domain links from the page
        base_domain = urlparse(url).netloc
        page_links = []
        for link in response.css("a::attr(href)").getall():
            abs_url = response.urljoin(link)
            parsed = urlparse(abs_url)
            if parsed.netloc == base_domain:
                clean = abs_url.split("#")[0].split("?")[0]
                if clean and clean != url and clean not in self._visited and clean not in page_links:
                    page_links.append(clean)

        # Yield the item for pipeline processing
        item = CrawlPageItem()
        item["url"] = url
        item["title"] = response.css("title::text").get() or ""
        item["html"] = response.text
        item["main_content"] = ""  # Filled by TrafilaturaPipeline
        item["links"] = page_links
        item["metadata"] = {}
        item["structured_data"] = {}
        item["extraction_type"] = extraction_type
        item["site_url"] = site_url
        item["depth"] = current_depth
        item["method"] = "playwright" if response.meta.get("playwright") else "static"

        yield item

        # Follow same-domain links based on depth limits
        max_links = DEPTH_LIMITS.get(current_depth, 0)
        if max_links == 0 or current_depth >= (max(DEPTH_LIMITS.keys()) if DEPTH_LIMITS else 1):
            return

        # Select links to follow
        selected_links = page_links if max_links is None else random.sample(page_links, min(len(page_links), max_links))

        next_depth = current_depth + 1
        for link_url in selected_links:
            if link_url in self._visited:
                continue

            meta: dict[str, Any] = {
                "site_url": site_url,
                "extraction_type": extraction_type,
                "current_depth": next_depth,
            }

            # Use Playwright for JS-heavy sub-pages too
            if _requires_javascript(link_url):
                meta["playwright"] = True
                meta["playwright_include_page"] = False
                meta["playwright_page_methods"] = [
                    PageMethod("wait_for_timeout", 3000),
                ]

            yield scrapy.Request(
                url=link_url,
                callback=self.parse,
                meta=meta,
                errback=self.handle_error,
                dont_filter=True,
            )

    def handle_error(self, failure: object) -> None:
        """Log request failures."""
        logger.warning("Request failed: %s", str(failure)[:200])

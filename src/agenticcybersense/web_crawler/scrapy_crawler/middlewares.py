"""Custom Scrapy middlewares for the CTI crawler."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from scrapy import signals

if TYPE_CHECKING:
    from scrapy import Spider
    from scrapy.crawler import Crawler
    from scrapy.http import Request, Response

logger = logging.getLogger(__name__)

# User-Agent pool matching the original crawler
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class RotateUserAgentMiddleware:
    """Rotate User-Agent header on every request."""

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> RotateUserAgentMiddleware:
        """Instantiate from crawler and connect signals."""
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def process_request(self, request: Request, spider: Spider) -> None:  # noqa: ARG002
        """Set a random User-Agent on each outgoing request."""
        request.headers["User-Agent"] = random.choice(USER_AGENTS)  # noqa: S311

    def spider_opened(self, spider: Spider) -> None:
        """Log when the middleware is active."""
        logger.info("RotateUserAgentMiddleware enabled for spider: %s", spider.name)


class PlaywrightFallbackMiddleware:
    """Re-dispatch requests through Playwright when static fetch returns thin HTML.

    If a response body is smaller than MIN_HTML_SIZE, the request is retried
    with playwright enabled so JS-rendered content can be captured.
    """

    MIN_HTML_SIZE = 5000

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> PlaywrightFallbackMiddleware:
        """Instantiate from crawler."""
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def process_response(self, request: Request, response: Response, spider: Spider) -> Request | Response:  # noqa: ARG002
        """Check response size and retry with Playwright if too small."""
        # Skip if already a Playwright request
        if request.meta.get("playwright"):
            return response

        if len(response.body) < self.MIN_HTML_SIZE:
            logger.info("Response too small (%d bytes), retrying with Playwright: %s", len(response.body), request.url)
            return request.replace(
                meta={**request.meta, "playwright": True, "playwright_include_page": False, "_fallback_retry": True},
                dont_filter=True,
            )

        return response

    def spider_opened(self, spider: Spider) -> None:
        """Log when the middleware is active."""
        logger.info("PlaywrightFallbackMiddleware enabled for spider: %s", spider.name)

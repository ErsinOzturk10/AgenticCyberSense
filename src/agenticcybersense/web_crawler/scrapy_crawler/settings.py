"""Scrapy settings for the AgenticCyberSense web crawler.

Uses scrapy-playwright for JavaScript-rendered pages and falls back to
standard HTTP requests for static sites. All settings are derived from
the existing web_crawler config module to keep configuration centralized.
"""

from agenticcybersense.web_crawler.config import (
    BLACKLIST,
    CONCURRENT_SITES,
    DEPTH_LIMITS,
    INACTIVITY_TIMEOUT,
)

BOT_NAME = "agenticcybersense_crawler"

SPIDER_MODULES = ["agenticcybersense.web_crawler.scrapy_crawler.spiders"]
NEWSPIDER_MODULE = "agenticcybersense.web_crawler.scrapy_crawler.spiders"

# Obey robots.txt — disabled for threat intel sites that may block crawlers
ROBOTSTXT_OBEY = False

# Concurrency settings (mirrors existing CONCURRENT_SITES)
CONCURRENT_REQUESTS = CONCURRENT_SITES
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Download timeout (mirrors INACTIVITY_TIMEOUT)
DOWNLOAD_TIMEOUT = INACTIVITY_TIMEOUT

# Retry configuration
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# AutoThrottle for polite crawling
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = float(CONCURRENT_SITES)

# Depth limit from existing config (use depth 1 limit as max)
DEPTH_LIMIT = max(k for k, v in DEPTH_LIMITS.items() if v != 0) if DEPTH_LIMITS else 1

# Disable cookies globally (avoids tracking/session issues)
COOKIES_ENABLED = False

# User-Agent rotation is handled in the spider via middleware
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Playwright integration via scrapy-playwright
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": [
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-gpu",
    ],
}

# Default Playwright context options
PLAYWRIGHT_CONTEXTS = {
    "default": {
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "java_script_enabled": True,
    },
}

# Item pipelines — order matters
ITEM_PIPELINES = {
    "agenticcybersense.web_crawler.scrapy_crawler.pipelines.trafilatura_pipeline.TrafilaturaPipeline": 100,
    "agenticcybersense.web_crawler.scrapy_crawler.pipelines.hash_dedup_pipeline.HashDedupPipeline": 200,
    "agenticcybersense.web_crawler.scrapy_crawler.pipelines.json_export_pipeline.JsonExportPipeline": 300,
    "agenticcybersense.web_crawler.scrapy_crawler.pipelines.rag_pipeline.RagIngestPipeline": 400,
}

# Custom settings for reference by pipelines
BLACKLISTED_URLS = BLACKLIST

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Request fingerprinting (Scrapy 2.7+)
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"

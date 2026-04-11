"""Trafilatura + Ollama extraction agent with SharedBrowser support."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import trafilatura
from playwright.async_api import Browser, BrowserContext, async_playwright

logger = logging.getLogger(__name__)


class SharedBrowser:
    """A single Chromium instance reused across crawls."""

    def __init__(self, max_concurrent_pages: int = 5):
        self._pw = None
        self._browser: Browser | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent_pages)
        self._started = False
        self._launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-gpu",
        ]

    async def start(self) -> None:
        """Start the browser once."""
        if self._started:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=self._launch_args,
        )
        self._started = True
        logger.info("SharedBrowser started")

    async def stop(self) -> None:
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        self._started = False
        logger.info("SharedBrowser stopped")

    async def new_context(self, user_agent: str) -> BrowserContext:
        """Create a new isolated browser context."""
        ctx = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        await ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        return ctx

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def is_started(self) -> bool:
        return self._started


@dataclass
class ExtractionResult:
    url: str
    title: str | None = None
    main_content: str | None = None
    metadata: dict[str, Any] | None = None
    links: list[str] | None = None
    structured_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.links is None:
            self.links = []
        if self.structured_data is None:
            self.structured_data = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "main_content": self.main_content,
            "metadata": self.metadata,
            "links": self.links,
            "structured_data": self.structured_data,
        }


class TrafilaturaOllamaAgent:
    """Production extraction agent with static and browser fallback."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://localhost:11434",
        shared_browser: SharedBrowser | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.shared_browser = shared_browser
        self._check_ollama()

    def _check_ollama(self) -> None:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                names = [m["name"] for m in response.json().get("models", [])]
                if self.model not in names:
                    logger.warning("Model '%s' not found. Run: ollama pull %s", self.model, self.model)
                else:
                    logger.info("Ollama is running, model: %s", self.model)
        except Exception as exc:
            logger.warning("No Ollama connection (IOC extraction disabled): %s", exc)

    async def extract_from_url(self, url: str, extraction_type: str = "general") -> ExtractionResult:
        start_time = datetime.now()
        logger.info("Extracting: %s...", url[:80])

        for attempt in range(2):
            try:
                if attempt > 0:
                    logger.info("Retry %s/2...", attempt + 1)
                    await asyncio.sleep(random.uniform(2, 4))

                requires_js = self._requires_javascript(url)

                if not requires_js and attempt == 0:
                    logger.info("Trying static fetch...")
                    html, title, links = self._fetch_static(url)
                    if html and len(html) > 5000:
                        content = self._extract_with_trafilatura(html, url)
                        if content and len(content) > 200:
                            logger.info("Static extraction successful")
                            return self._build_result(
                                url,
                                title,
                                content,
                                links,
                                extraction_type,
                                start_time,
                                "static",
                            )
                    logger.warning("Static extraction insufficient, switching to Playwright")

                logger.info("Using Playwright (stealth mode)...")
                html, title, links = await self._fetch_with_playwright(url)

                if not html:
                    if attempt == 0:
                        continue
                    return ExtractionResult(
                        url=url,
                        metadata={
                            "error": "Fetch failed after retry",
                            "status": "failed",
                        },
                    )

                content = self._extract_with_trafilatura(html, url)
                if not content or len(content) < 100:
                    content = self._fallback_extraction(html)

                if content and len(content) > 100:
                    return self._build_result(
                        url,
                        title,
                        content,
                        links,
                        extraction_type,
                        start_time,
                        "playwright",
                    )

                if attempt == 0:
                    continue

                return ExtractionResult(
                    url=url,
                    title=title,
                    main_content=content or "",
                    links=links,
                    metadata={
                        "error": "Content too short",
                        "status": "partial",
                        "method": "playwright",
                    },
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("First attempt failed: %s", str(exc)[:80])
                    continue
                logger.exception("All attempts failed")
                return ExtractionResult(
                    url=url,
                    metadata={
                        "error": str(exc)[:200],
                        "status": "failed",
                    },
                )

        return ExtractionResult(
            url=url,
            metadata={
                "error": "Max retries exceeded",
                "status": "failed",
            },
        )

    async def _fetch_with_playwright(self, url: str) -> tuple[str | None, str | None, list[str]]:
        try:
            return await asyncio.wait_for(
                self._fetch_playwright_internal(url),
                timeout=150.0,
            )
        except TimeoutError:
            logger.warning("Timeout (150s)")
            return None, None, []
        except Exception:
            logger.exception("Playwright fetch failed")
            return None, None, []

    async def _fetch_playwright_internal(self, url: str) -> tuple[str, str | None, list[str]]:
        ua = random.choice(self.USER_AGENTS)

        if self.shared_browser and self.shared_browser.is_started:
            async with self.shared_browser.semaphore:
                context = await self.shared_browser.new_context(user_agent=ua)
                try:
                    return await self._run_page(context, url)
                finally:
                    await context.close()
        else:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                )
                context = await browser.new_context(
                    user_agent=ua,
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
                try:
                    return await self._run_page(context, url)
                finally:
                    await browser.close()

    async def _run_page(self, context: BrowserContext, url: str) -> tuple[str, str | None, list[str]]:
        """Load the page and extract content."""
        page = await context.new_page()
        await self._dismiss_cookie_banners(page)

        logger.info("Loading page (90s timeout)...")
        try:
            await page.goto(url, wait_until="networkidle", timeout=90000)
        except Exception:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception:
                await page.close()
                return "", None, []

        logger.info("Waiting for JS rendering (5s)...")
        await page.wait_for_timeout(5000)
        await self._dismiss_cookie_banners(page)
        await self._wait_for_content(page, url)

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)

        html = await page.content()
        title = await page.title()

        if "Loading..." in html[:1000] or len(html) < 5000:
            logger.warning("Waiting an additional 5s for content...")
            await page.wait_for_timeout(5000)
            html = await page.content()

        logger.info("Fetched %s bytes", f"{len(html):,}")
        links = await self._extract_links_with_playwright(page, url)
        await page.close()

        return html, title, links

    def _requires_javascript(self, url: str) -> bool:
        url_lower = url.lower()
        js_sites = [
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
        return any(ind in url_lower for ind in js_sites)

    def _fetch_static(self, url: str) -> tuple[str | None, str | None, list[str]]:
        try:
            ua = random.choice(self.USER_AGENTS)
            downloaded = trafilatura.fetch_url(url, config={"USER_AGENT": ua})
            if not downloaded:
                return None, None, []
            links = self._extract_links_from_html(downloaded, url)
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata else None
            return downloaded, title, links
        except Exception:
            logger.exception("Static fetch failed")
            return None, None, []

    def _build_result(
        self,
        url: str,
        title: str | None,
        content: str,
        links: list[str],
        extraction_type: str,
        start_time: datetime,
        method: str,
    ) -> ExtractionResult:
        structured_data: dict[str, Any] = {}
        if extraction_type == "threat_intel" and len(content) > 200:
            logger.info("Extracting IOCs...")
            try:
                loop = asyncio.get_event_loop()
                structured_data = loop.run_until_complete(
                    self._extract_iocs_with_ollama(content),
                )
            except Exception:
                structured_data = self._empty_structured_data()

        result = ExtractionResult(url=url)
        result.title = title
        result.main_content = content
        result.links = links
        result.structured_data = structured_data
        result.metadata = {
            "extraction_type": extraction_type,
            "method": method,
            "model": self.model if structured_data else None,
            "status": "success",
            "content_length": len(content),
            "link_count": len(links),
            "duration_seconds": (datetime.now() - start_time).seconds,
        }
        logger.info("Extraction successful: %s chars, %s links", f"{len(content):,}", len(links))
        return result

    async def _wait_for_content(self, page, url: str) -> None:
        url_lower = url.lower()
        selectors_map = {
            "alienvault.com": [".pulse-list", '[class*="pulse"]'],
            "virustotal.com": ['[class*="report"]', ".vt-ui-main"],
            "shodan.io": [".search-result", ".banner"],
            "any.run": [".report", '[class*="analysis"]'],
            "criminalip.io": [".scan-result", '[class*="scan"]'],
            "socradar.io": [".threat", '[class*="threat"]'],
            "hudsonrock.com": [".content", '[class*="data"]'],
        }
        for domain, selectors in selectors_map.items():
            if domain in url_lower:
                for sel in selectors:
                    try:
                        await page.wait_for_selector(sel, timeout=10000)
                        return
                    except Exception:
                        continue
                await page.wait_for_timeout(8000)
                return
        await page.wait_for_timeout(3000)

    async def _dismiss_cookie_banners(self, page) -> None:
        from playwright._impl._errors import TargetClosedError

        selectors = [
            'button:has-text("Accept")',
            'button:has-text("I agree")',
            'button:has-text("OK")',
            'button:has-text("Allow")',
            '[id*="cookie"] button',
            '[class*="cookie"] button',
        ]
        for sel in selectors:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(1000)
                    break
            except TargetClosedError:
                return
            except Exception:
                continue

    async def _extract_links_with_playwright(self, page, base_url: str) -> list[str]:
        from urllib.parse import urljoin, urlparse

        base_domain = urlparse(base_url).netloc
        links: list[str] = []
        try:
            elements = await page.query_selector_all("a[href]")
            for elem in elements[:200]:
                href = await elem.get_attribute("href")
                if href:
                    abs_url = urljoin(base_url, href)
                    if urlparse(abs_url).netloc == base_domain:
                        clean = abs_url.split("#")[0].split("?")[0]
                        if clean and clean != base_url and clean not in links:
                            links.append(clean)
        except Exception:
            logger.exception("Failed to extract links with Playwright")
        return links

    def _extract_links_from_html(self, html: str, base_url: str) -> list[str]:
        from urllib.parse import urljoin, urlparse

        from bs4 import BeautifulSoup

        base_domain = urlparse(base_url).netloc
        links: list[str] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.find_all("a", href=True):
                abs_url = urljoin(base_url, anchor["href"])
                if urlparse(abs_url).netloc == base_domain:
                    clean = abs_url.split("#")[0].split("?")[0]
                    if clean and clean != base_url and clean not in links:
                        links.append(clean)
        except Exception:
            logger.exception("Failed to extract links from HTML")
        return links[:200]

    def _extract_with_trafilatura(self, html: str, url: str) -> str:
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
            logger.exception("Trafilatura extraction failed")
            return ""

    def _fallback_extraction(self, html: str) -> str:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text)
        except Exception:
            logger.exception("Fallback extraction failed")
            return ""

    async def _extract_iocs_with_ollama(self, content: str) -> dict[str, Any]:
        preview = content[:20000]
        prompt = (
            "Extract IOCs. Return ONLY JSON:\n"
            '{"iocs": {"ips": [], "domains": [], "hashes": []}, '
            '"cves": [], "malware": [], "threat_actors": []}\n\n'
            f"Text: {preview}"
        )
        try:
            result = await self._call_ollama(prompt, 2000)
            if result:
                parsed = self._parse_json_robust(result)
                if parsed and "iocs" in parsed:
                    return parsed
        except Exception:
            logger.exception("IOC extraction failed")
        return self._empty_structured_data()

    def _empty_structured_data(self) -> dict[str, Any]:
        return {
            "iocs": {"ips": [], "domains": [], "hashes": []},
            "cves": [],
            "malware": [],
            "threat_actors": [],
        }

    async def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0,
                            "num_predict": max_tokens,
                        },
                    },
                )
            return resp.json().get("response", "") if resp.status_code == 200 else ""
        except Exception:
            logger.exception("Ollama call failed")
            return ""

    def _parse_json_robust(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        if "```json" in text:
            try:
                return json.loads(text.split("```json")[1].split("```")[0].strip())
            except Exception:
                pass
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {}

    async def extract_batch(self, urls: list[str], extraction_types: list[str]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for i, (url, ext_type) in enumerate(zip(urls, extraction_types), 1):
            logger.info("[%s/%s] %s...", i, len(urls), url[:80])
            result = await self.extract_from_url(url, ext_type)
            status = "success" if result.metadata.get("status") == "success" else "failed"
            logger.info("Result: %s chars, status=%s", len(result.main_content or ""), status)
            results.append(result)
            if i < len(urls):
                await asyncio.sleep(random.uniform(1, 2))
        return results


class SmartExtractionAgent:
    def __init__(self, agent: TrafilaturaOllamaAgent):
        self.agent = agent

    def detect_extraction_type(self, url: str) -> str:
        url_lower = url.lower()
        if "github.com" in url_lower:
            return "github"
        if any(
            kw in url_lower
            for kw in [
                "threat",
                "security",
                "malware",
                "cve",
                "attack.mitre",
                "alienvault",
                "virustotal",
                "shodan",
                "criminalip",
            ]
        ):
            return "threat_intel"
        return "general"

    async def smart_extract_batch(self, urls: list[str]) -> list[ExtractionResult]:
        extraction_types = [self.detect_extraction_type(u) for u in urls]
        from collections import Counter

        logger.info("Extraction type distribution:")
        for t, c in Counter(extraction_types).items():
            logger.info("   %s: %s", t, c)

        return await self.agent.extract_batch(urls, extraction_types)

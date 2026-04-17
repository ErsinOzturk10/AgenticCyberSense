"""
Trafilatura + Ollama Agent — SharedBrowser desteği eklendi (v2)

Değişiklikler:
  - SharedBrowser: tek bir Chromium process, çok sayıda context
    → Her URL için browser launch/close maliyeti ortadan kalkar
  - TrafilaturaOllamaAgent(shared_browser=...) parametresi eklendi
  - Geriye dönük uyumlu: shared_browser verilmezse eski davranış
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import asyncio
import os
import httpx
import re
from datetime import datetime
from playwright.async_api import async_playwright, Browser, BrowserContext
import trafilatura
import random


# ══════════════════════════════════════════════════════════════════════ #
#  SharedBrowser — tek process, paylaşımlı context havuzu               #
# ══════════════════════════════════════════════════════════════════════ #

class SharedBrowser:
    """
    Uygulama ömrü boyunca yaşayan tek bir Chromium instance.
    Her URL çağrısı yeni bir context açar, bitince kapatır.
    Browser'ın kendisi kapatılmaz → launch maliyeti sadece 1 kez ödenir.

    max_concurrent_pages: aynı anda kaç sayfa açık olabilir
    """

    def __init__(self, max_concurrent_pages: int = 5):
        self._pw             = None
        self._browser: Optional[Browser] = None
        self._semaphore      = asyncio.Semaphore(max_concurrent_pages)
        self._started        = False
        self._launch_args    = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-gpu",
        ]

    async def start(self):
        """Browser'ı başlat (main.py'de bir kez çağrılır)."""
        if self._started:
            return
        self._pw      = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True, args=self._launch_args
        )
        self._started = True
        print("🟢 SharedBrowser başlatıldı")

    async def stop(self):
        """Tüm crawl bittiğinde çağrılır."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        self._started = False
        print("🔴 SharedBrowser durduruldu")

    async def new_context(self, user_agent: str) -> BrowserContext:
        """Yeni bir izole browser context döndür."""
        ctx = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        return ctx

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def is_started(self) -> bool:
        return self._started


# ══════════════════════════════════════════════════════════════════════ #
#  ExtractionResult                                                      #
# ══════════════════════════════════════════════════════════════════════ #

@dataclass
class ExtractionResult:
    url: str
    title: Optional[str] = None
    main_content: Optional[str] = None
    metadata: Dict[str, Any] = None
    links: List[str] = None
    structured_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.links is None:
            self.links = []
        if self.structured_data is None:
            self.structured_data = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "main_content": self.main_content,
            "metadata": self.metadata,
            "links": self.links,
            "structured_data": self.structured_data,
        }


# ══════════════════════════════════════════════════════════════════════ #
#  TrafilaturaOllamaAgent                                                #
# ══════════════════════════════════════════════════════════════════════ #

class TrafilaturaOllamaAgent:
    """
    PRODUCTION AGENT:
    - Static-first (hızlı)
    - Playwright fallback (JS gereken siteler)
    - SharedBrowser ile browser yeniden kullanımı
    - Ollama IOC extraction (threat_intel tipi için)
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://localhost:11434",
        shared_browser: Optional[SharedBrowser] = None,
    ):
        self.model          = model
        self.base_url       = base_url
        self.api_url        = f"{base_url}/api/generate"
        self.shared_browser = shared_browser  # None → fallback eski davranış
        self._check_ollama()

    # ------------------------------------------------------------------ #
    # Ollama kontrolü                                                     #
    # ------------------------------------------------------------------ #

    def _check_ollama(self):
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                names = [m["name"] for m in response.json().get("models", [])]
                if self.model not in names:
                    print(f"⚠️  Model '{self.model}' bulunamadı! → ollama pull {self.model}")
                else:
                    print(f"✅ Ollama çalışıyor, model: {self.model}")
        except Exception as e:
            print(f"⚠️  Ollama bağlantısı yok (IOC extraction devre dışı): {e}")

    # ------------------------------------------------------------------ #
    # Ana extraction girişi                                               #
    # ------------------------------------------------------------------ #

    async def extract_from_url(
        self,
        url: str,
        extraction_type: str = "general",
    ) -> ExtractionResult:
        start_time = datetime.now()
        print(f"    🎯 Extracting: {url[:80]}...")

        for attempt in range(2):
            try:
                if attempt > 0:
                    print(f"    🔄 Retry {attempt + 1}/2...")
                    await asyncio.sleep(random.uniform(2, 4))

                requires_js = self._requires_javascript(url)

                # Static önce (JS gerektirmiyorsa ve ilk deneme)
                if not requires_js and attempt == 0:
                    print(f"    📄 Static fetch deneniyor...")
                    html, title, links = self._fetch_static(url)
                    if html and len(html) > 5000:
                        content = self._extract_with_trafilatura(html, url)
                        if content and len(content) > 200:
                            print(f"    ✅ Static başarılı!")
                            return self._build_result(
                                url, title, content, links,
                                extraction_type, start_time, "static",
                            )
                    print(f"    ⚠️  Static yetersiz, Playwright'a geçiliyor...")

                # Playwright (SharedBrowser varsa paylaşımlı, yoksa eski yol)
                print(f"    🌐 Playwright (stealth mode)...")
                html, title, links = await self._fetch_with_playwright(url)

                if not html:
                    if attempt == 0:
                        continue
                    return ExtractionResult(
                        url=url,
                        metadata={"error": "Retry sonrası fetch başarısız", "status": "failed"},
                    )

                content = self._extract_with_trafilatura(html, url)
                if not content or len(content) < 100:
                    content = self._fallback_extraction(html)

                if content and len(content) > 100:
                    return self._build_result(
                        url, title, content, links,
                        extraction_type, start_time, "playwright",
                    )

                if attempt == 0:
                    continue

                return ExtractionResult(
                    url=url,
                    title=title,
                    main_content=content or "",
                    links=links,
                    metadata={"error": "İçerik çok kısa", "status": "partial", "method": "playwright"},
                )

            except Exception as e:
                if attempt == 0:
                    print(f"    ⚠️  Deneme başarısız: {str(e)[:50]}")
                    continue
                print(f"    ❌ Tüm denemeler başarısız: {str(e)[:80]}")
                return ExtractionResult(
                    url=url,
                    metadata={"error": str(e)[:200], "status": "failed"},
                )

        return ExtractionResult(
            url=url,
            metadata={"error": "Max retry aşıldı", "status": "failed"},
        )

    # ------------------------------------------------------------------ #
    # Playwright — SharedBrowser veya standalone                          #
    # ------------------------------------------------------------------ #

    async def _fetch_with_playwright(
        self, url: str
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        try:
            return await asyncio.wait_for(
                self._fetch_playwright_internal(url),
                timeout=150.0,
            )
        except asyncio.TimeoutError:
            print(f"    ⏱️  Timeout (150s)")
            return None, None, []
        except Exception:
            return None, None, []

    async def _fetch_playwright_internal(
        self, url: str
    ) -> Tuple[str, Optional[str], List[str]]:

        ua = random.choice(self.USER_AGENTS)

        if self.shared_browser and self.shared_browser.is_started:
            # ── Paylaşımlı browser: sadece context aç/kapat ──────────── #
            async with self.shared_browser.semaphore:
                context = await self.shared_browser.new_context(user_agent=ua)
                try:
                    return await self._run_page(context, url)
                finally:
                    await context.close()
        else:
            # ── Geriye dönük uyumluluk: standalone browser ─────────────  #
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
                await context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )
                try:
                    return await self._run_page(context, url)
                finally:
                    await browser.close()

    async def _run_page(
        self, context: BrowserContext, url: str
    ) -> Tuple[str, Optional[str], List[str]]:
        """Verilen context üzerinde sayfayı yükle ve içeriği al."""
        page = await context.new_page()
        await self._dismiss_cookie_banners(page)

        print(f"    ⏳ Yükleniyor (90s)...")
        try:
            await page.goto(url, wait_until="networkidle", timeout=90000)
        except Exception:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception:
                await page.close()
                return "", None, []

        print(f"    ⏳ JS render bekleniyor (5s)...")
        await page.wait_for_timeout(5000)
        await self._dismiss_cookie_banners(page)
        await self._wait_for_content(page, url)

        # Lazy-load için scroll
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)

        html  = await page.content()
        title = await page.title()

        if "Loading..." in html[:1000] or len(html) < 5000:
            print(f"    ⚠️  Extra 5s bekleniyor...")
            await page.wait_for_timeout(5000)
            html = await page.content()

        print(f"    ✅ {len(html):,} bytes")
        links = await self._extract_links_with_playwright(page, url)
        await page.close()

        return html, title, links

    # ------------------------------------------------------------------ #
    # Yardımcı metodlar (değişmedi)                                       #
    # ------------------------------------------------------------------ #

    def _requires_javascript(self, url: str) -> bool:
        url_lower = url.lower()
        js_sites  = [
            "alienvault.com", "virustotal.com", "shodan.io", "greynoise.io",
            "any.run", "criminalip.io", "hudsonrock.com",
            "socradar.io", "solarwinds.com", "att.com", "disinfox.com",
            "app.", "dashboard.", "console.", "portal.",
            "/app/", "/dashboard/", "/console/", "/portal/",
        ]
        return any(ind in url_lower for ind in js_sites)

    def _fetch_static(self, url: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        try:
            ua         = random.choice(self.USER_AGENTS)
            downloaded = trafilatura.fetch_url(
                url, config={"USER_AGENT": ua}
            )
            if not downloaded:
                return None, None, []
            links    = self._extract_links_from_html(downloaded, url)
            metadata = trafilatura.extract_metadata(downloaded)
            title    = metadata.title if metadata else None
            return downloaded, title, links
        except Exception:
            return None, None, []

    def _build_result(
        self, url, title, content, links, extraction_type, start_time, method
    ) -> ExtractionResult:
        structured_data = {}
        if extraction_type == "threat_intel" and len(content) > 200:
            print(f"    🔍 IOC çıkarılıyor...")
            try:
                loop = asyncio.get_event_loop()
                structured_data = loop.run_until_complete(
                    self._extract_iocs_with_ollama(content)
                )
            except Exception:
                structured_data = self._empty_structured_data()

        result             = ExtractionResult(url=url)
        result.title       = title
        result.main_content = content
        result.links        = links
        result.structured_data = structured_data
        result.metadata     = {
            "extraction_type": extraction_type,
            "method":          method,
            "model":           self.model if structured_data else None,
            "status":          "success",
            "content_length":  len(content),
            "link_count":      len(links),
            "duration_seconds": (datetime.now() - start_time).seconds,
        }
        print(f"    ✅ {len(content):,} chars, {len(links)} link")
        return result

    async def _wait_for_content(self, page, url: str):
        url_lower = url.lower()
        selectors_map = {
            "alienvault.com": [".pulse-list", '[class*="pulse"]'],
            "virustotal.com": ['[class*="report"]', ".vt-ui-main"],
            "shodan.io":      [".search-result", ".banner"],
            "any.run":        [".report", '[class*="analysis"]'],
            "criminalip.io":  [".scan-result", '[class*="scan"]'],
            "socradar.io":    [".threat", '[class*="threat"]'],
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

    async def _dismiss_cookie_banners(self, page):
        from playwright._impl._errors import TargetClosedError
        selectors = [
            'button:has-text("Accept")', 'button:has-text("I agree")',
            'button:has-text("OK")',     'button:has-text("Allow")',
            '[id*="cookie"] button',     '[class*="cookie"] button',
        ]
        for sel in selectors:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(1000)
                    break
            except TargetClosedError:
                return  # Sayfa/context kapandı, sessizce çık
            except Exception:
                continue

    async def _extract_links_with_playwright(self, page, base_url: str) -> List[str]:
        from urllib.parse import urljoin, urlparse
        base_domain = urlparse(base_url).netloc
        links = []
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
            pass
        return links

    def _extract_links_from_html(self, html: str, base_url: str) -> List[str]:
        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup
        base_domain = urlparse(base_url).netloc
        links = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                abs_url = urljoin(base_url, a["href"])
                if urlparse(abs_url).netloc == base_domain:
                    clean = abs_url.split("#")[0].split("?")[0]
                    if clean and clean != base_url and clean not in links:
                        links.append(clean)
        except Exception:
            pass
        return links[:200]

    def _extract_with_trafilatura(self, html: str, url: str) -> str:
        try:
            return trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=False,
                url=url,
            ) or ""
        except Exception:
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
            return ""

    async def _extract_iocs_with_ollama(self, content: str) -> Dict[str, Any]:
        preview = content[:20000]
        prompt  = (
            'Extract IOCs. Return ONLY JSON:\n'
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
            pass
        return self._empty_structured_data()

    def _empty_structured_data(self) -> Dict[str, Any]:
        return {
            "iocs": {"ips": [], "domains": [], "hashes": []},
            "cves": [], "malware": [], "threat_actors": [],
        }

    async def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    self.api_url,
                    json={
                        "model":   self.model,
                        "prompt":  prompt,
                        "stream":  False,
                        "format":  "json",
                        "options": {"temperature": 0, "num_predict": max_tokens},
                    },
                )
            return resp.json().get("response", "") if resp.status_code == 200 else ""
        except Exception:
            return ""

    def _parse_json_robust(self, text: str) -> Dict[str, Any]:
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
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return {}

    async def extract_batch(
        self, urls: List[str], extraction_types: List[str]
    ) -> List[ExtractionResult]:
        results = []
        for i, (url, ext_type) in enumerate(zip(urls, extraction_types), 1):
            print(f"\n  📄 [{i}/{len(urls)}] {url[:80]}...")
            result = await self.extract_from_url(url, ext_type)
            status = "✅" if result.metadata.get("status") == "success" else "❌"
            print(f"  {status} {len(result.main_content or ''):,} chars")
            results.append(result)
            if i < len(urls):
                await asyncio.sleep(random.uniform(1, 2))
        return results


# ══════════════════════════════════════════════════════════════════════ #
#  SmartExtractionAgent (değişmedi)                                      #
# ══════════════════════════════════════════════════════════════════════ #

class SmartExtractionAgent:
    def __init__(self, agent: TrafilaturaOllamaAgent):
        self.agent = agent

    def detect_extraction_type(self, url: str) -> str:
        url_lower = url.lower()
        if "github.com" in url_lower:
            return "github"
        if any(kw in url_lower for kw in [
            "threat", "security", "malware", "cve", "attack.mitre",
            "alienvault", "virustotal", "shodan", "criminalip",
        ]):
            return "threat_intel"
        return "general"

    async def smart_extract_batch(self, urls: List[str]) -> List["ExtractionResult"]:
        extraction_types = [self.detect_extraction_type(u) for u in urls]
        from collections import Counter
        print("\n📊 Tip dağılımı:")
        for t, c in Counter(extraction_types).items():
            print(f"   {t}: {c}")
        return await self.agent.extract_batch(urls, extraction_types)
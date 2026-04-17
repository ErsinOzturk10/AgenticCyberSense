"""
Deep Crawler — Alt sayfa bazlı hash kontrolü (v3)

Değişiklikler:
  - _crawl_recursive: sayfa çekildikten sonra hash kontrolü yapılır
    → değişmemişse sonuçlara eklenmez, alt linkleri de taranmaz
    → değişmişse normal işlenir + hash güncellenir
  - history_manager artık _crawl_recursive'e de geçirilir
"""
from trafilatura_ollama_agent import TrafilaturaOllamaAgent, SmartExtractionAgent
import asyncio
from typing import List, Set, Optional
from urllib.parse import urlparse
from datetime import datetime
import random
from config import DEPTH_LIMITS


class DeepCrawler:

    def __init__(self, agent: TrafilaturaOllamaAgent, max_depth: int = 1):
        self.agent              = agent
        self.max_depth          = max_depth
        self.visited: Set[str]  = set()
        self.last_activity: datetime = datetime.now()
        # İstatistik: kaç alt sayfa atlandı
        self.skipped_subpages   = 0
        self.crawled_subpages   = 0

    def _touch(self):
        self.last_activity = datetime.now()

    async def deep_crawl(
        self,
        start_url: str,
        extraction_type: str = "general",
        history_manager=None,
    ):
        print(f"\n🌐 Deep Crawling: {start_url}")
        print(f"   Max Depth: {self.max_depth}")

        results    = []
        start_time = datetime.now()

        # ── Depth 0: Ana sayfa ──────────────────────────────────────── #
        print(f"\n  📄 [Depth 0] Ana sayfa crawl ediliyor...")
        main_result = await self.agent.extract_from_url(start_url, extraction_type)
        results.append(main_result)
        self.visited.add(start_url)

        if main_result.metadata.get("status") != "success":
            print(f"  ❌ Ana sayfa başarısız")
            return results

        self._touch()

        main_content = main_result.main_content or ""
        main_links   = main_result.links

        print(f"  ✅ Ana sayfa: {len(main_content):,} karakter, {len(main_links)} link")

        # ── Ana sayfa hash kontrolü ─────────────────────────────────── #
        if history_manager and main_content:
            should_crawl, reason = history_manager.should_deep_crawl(start_url, main_content)
            if not should_crawl:
                print(f"  ⏭️  Hash eşleşti — deep crawl atlanıyor")
                return results

        # ── Depth 1+: Alt linkler ────────────────────────────────────── #
        print(f"  🚀 Deep crawl başlıyor...")

        base_domain       = urlparse(start_url).netloc
        same_domain_links = [
            link for link in main_links
            if urlparse(link).netloc == base_domain and link not in self.visited
        ]

        print(f"  🔗 Aynı domain linkleri: {len(same_domain_links)}")

        max_links_depth_0 = DEPTH_LIMITS.get(0, 0)

        if max_links_depth_0 == 0:
            print(f"  ⚠️  Depth 0 limit = 0, alt crawl yok")
            return results

        if max_links_depth_0 is None:
            selected_links = same_domain_links
            print(f"  ✅ SINIRSIZ — {len(selected_links)} link crawl edilecek")
        else:
            selected_links = random.sample(
                same_domain_links,
                min(len(same_domain_links), max_links_depth_0),
            )
            print(f"  ✅ {len(selected_links)}/{len(same_domain_links)} link seçildi")

        for i, link in enumerate(selected_links, 1):
            print(f"\n    ↳ [{i}/{len(selected_links)}] Depth 1")
            await self._crawl_recursive(link, extraction_type, 1, results, history_manager)

        duration = (datetime.now() - start_time).seconds
        print(f"\n✅ Deep crawl tamamlandı: {len(results)} sayfa ({duration}s)")
        print(f"   Alt sayfa istatistik → Çekilen: {self.crawled_subpages} | Atlanan: {self.skipped_subpages}")

        return results

    async def _crawl_recursive(
        self,
        url: str,
        extraction_type: str,
        current_depth: int,
        results: List,
        history_manager=None,   # ← YENİ parametre
    ):
        if url in self.visited:
            return
        self.visited.add(url)

        indent = "  " * current_depth
        print(f"{indent}📄 [Depth {current_depth}] {url[:70]}...")

        result = await self.agent.extract_from_url(url, extraction_type)

        if result.metadata.get("status") != "success":
            print(f"{indent}❌ Başarısız")
            results.append(result)
            return

        content = result.main_content or ""
        self._touch()

        # ── Alt sayfa hash kontrolü ─────────────────────────────────── #
        if history_manager and content:
            should_process, reason = history_manager.should_crawl_subpage(url, content)

            if not should_process:
                # İçerik değişmemiş → atla
                print(f"{indent}⏭️  Alt sayfa değişmemiş — atlanıyor")
                self.skipped_subpages += 1
                return  # sonuçlara ekleme, alt linklere bakma

            # Değişmiş veya yeni → hash'i güncelle
            history_manager.update_subpage(url, content)

        self.crawled_subpages += 1
        results.append(result)
        print(f"{indent}✅ {len(content):,} karakter, {len(result.links)} link")

        if current_depth >= self.max_depth:
            print(f"{indent}⚠️  Max derinlik")
            return

        max_links = DEPTH_LIMITS.get(current_depth, 0)
        if max_links == 0:
            return

        base_domain = urlparse(url).netloc
        same_domain_links = [
            link for link in result.links
            if urlparse(link).netloc == base_domain and link not in self.visited
        ]

        if not same_domain_links:
            return

        if max_links is None:
            selected_links = same_domain_links
            print(f"{indent}🔗 {len(selected_links)} link crawl ediliyor")
        else:
            selected_links = random.sample(
                same_domain_links,
                min(len(same_domain_links), max_links),
            )
            print(f"{indent}🔗 {len(selected_links)}/{len(same_domain_links)} link seçildi")

        for i, link in enumerate(selected_links, 1):
            print(f"{indent}  ↳ [{i}/{len(selected_links)}]")
            await self._crawl_recursive(link, extraction_type, current_depth + 1, results, history_manager)


class SmartDeepCrawler:

    def __init__(self, agent: TrafilaturaOllamaAgent, max_depth: int = 1):
        self.crawler = DeepCrawler(agent, max_depth)

    @property
    def last_activity(self) -> datetime:
        return self.crawler.last_activity

    def detect_extraction_type(self, url: str) -> str:
        url_lower = url.lower()
        if "github.com" in url_lower:
            return "github"
        if any(kw in url_lower for kw in ["threat", "security", "malware", "attack.mitre"]):
            return "threat_intel"
        return "general"

    async def smart_deep_crawl(self, url: str, history_manager=None):
        ext_type = self.detect_extraction_type(url)
        print(f"  🔍 Type: {ext_type}")
        return await self.crawler.deep_crawl(url, ext_type, history_manager)
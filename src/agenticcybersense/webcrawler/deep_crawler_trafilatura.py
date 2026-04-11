"""Deep crawler with subpage-level hash checks."""

from __future__ import annotations

import random
from datetime import datetime

from config import DEPTH_LIMITS
from trafilatura_ollama_agent import TrafilaturaOllamaAgent


class DeepCrawler:
    def __init__(self, agent: TrafilaturaOllamaAgent, max_depth: int = 1):
        self.agent = agent
        self.max_depth = max_depth
        self.visited: set[str] = set()
        self.last_activity: datetime = datetime.now()
        self.skipped_subpages = 0
        self.crawled_subpages = 0

    def _touch(self) -> None:
        self.last_activity = datetime.now()

    async def deep_crawl(
        self,
        start_url: str,
        extraction_type: str = "general",
        history_manager=None,
    ) -> list:
        print(f"\n🌐 Deep Crawling: {start_url}")
        print(f"   Max Depth: {self.max_depth}")

        results = []
        start_time = datetime.now()

        print("\n  📄 [Depth 0] Crawling main page...")
        main_result = await self.agent.extract_from_url(start_url, extraction_type)
        results.append(main_result)
        self.visited.add(start_url)

        if main_result.metadata.get("status") != "success":
            print("  ❌ Main page failed")
            return results

        self._touch()

        main_content = main_result.main_content or ""
        main_links = main_result.links

        print(f"  ✅ Main page: {len(main_content):,} chars, {len(main_links)} links")

        if history_manager and main_content:
            should_crawl, _reason = history_manager.should_deep_crawl(start_url, main_content)
            if not should_crawl:
                print("  ⏭️  Hash matched — skipping deep crawl")
                return results

        print("  🚀 Starting deep crawl...")

        from urllib.parse import urlparse

        base_domain = urlparse(start_url).netloc
        same_domain_links = [link for link in main_links if urlparse(link).netloc == base_domain and link not in self.visited]

        print(f"  🔗 Same-domain links: {len(same_domain_links)}")

        max_links_depth_0 = DEPTH_LIMITS.get(0, 0)

        if max_links_depth_0 == 0:
            print("  ⚠️  Depth 0 limit = 0, skipping sub crawl")
            return results

        if max_links_depth_0 is None:
            selected_links = same_domain_links
            print(f"  ✅ UNLIMITED — {len(selected_links)} links will be crawled")
        else:
            selected_links = random.sample(
                same_domain_links,
                min(len(same_domain_links), max_links_depth_0),
            )
            print(f"  ✅ Selected {len(selected_links)}/{len(same_domain_links)} links")

        for i, link in enumerate(selected_links, 1):
            print(f"\n    ↳ [{i}/{len(selected_links)}] Depth 1")
            await self._crawl_recursive(link, extraction_type, 1, results, history_manager)

        duration = (datetime.now() - start_time).seconds
        print(f"\n✅ Deep crawl completed: {len(results)} pages ({duration}s)")
        print(f"   Subpage stats → Crawled: {self.crawled_subpages} | Skipped: {self.skipped_subpages}")

        return results

    async def _crawl_recursive(
        self,
        url: str,
        extraction_type: str,
        current_depth: int,
        results: list,
        history_manager=None,
    ) -> None:
        if url in self.visited:
            return
        self.visited.add(url)

        indent = "  " * current_depth
        print(f"{indent}📄 [Depth {current_depth}] {url[:70]}...")

        result = await self.agent.extract_from_url(url, extraction_type)

        if result.metadata.get("status") != "success":
            print(f"{indent}❌ Failed")
            results.append(result)
            return

        content = result.main_content or ""
        self._touch()

        if history_manager and content:
            should_process, _reason = history_manager.should_crawl_subpage(url, content)

            if not should_process:
                print(f"{indent}⏭️  Subpage unchanged — skipping")
                self.skipped_subpages += 1
                return

            history_manager.update_subpage(url, content)

        self.crawled_subpages += 1
        results.append(result)
        print(f"{indent}✅ {len(content):,} chars, {len(result.links)} links")

        if current_depth >= self.max_depth:
            print(f"{indent}⚠️  Max depth reached")
            return

        max_links = DEPTH_LIMITS.get(current_depth, 0)
        if max_links == 0:
            return

        from urllib.parse import urlparse

        base_domain = urlparse(url).netloc
        same_domain_links = [link for link in result.links if urlparse(link).netloc == base_domain and link not in self.visited]

        if not same_domain_links:
            return

        if max_links is None:
            selected_links = same_domain_links
            print(f"{indent}🔗 Crawling {len(selected_links)} links")
        else:
            selected_links = random.sample(
                same_domain_links,
                min(len(same_domain_links), max_links),
            )
            print(f"{indent}🔗 Selected {len(selected_links)}/{len(same_domain_links)} links")

        for i, link in enumerate(selected_links, 1):
            print(f"{indent}  ↳ [{i}/{len(selected_links)}]")
            await self._crawl_recursive(
                link,
                extraction_type,
                current_depth + 1,
                results,
                history_manager,
            )


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

    async def smart_deep_crawl(self, url: str, history_manager=None) -> list:
        ext_type = self.detect_extraction_type(url)
        print(f"  🔍 Type: {ext_type}")
        return await self.crawler.deep_crawl(url, ext_type, history_manager)

"""Main crawler entry point with concurrency and shared browser support."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from config import (
    BLACKLIST,
    CONCURRENT_SITES,
    ENABLE_INCREMENTAL,
    FORCE_FULL_CRAWL,
    HISTORY_FILE,
    INACTIVITY_TIMEOUT,
    OUTPUT_FILE,
)
from crawl_history_manager import CrawlHistoryManager
from deep_crawler_trafilatura import SmartDeepCrawler
from trafilatura_ollama_agent import SharedBrowser, TrafilaturaOllamaAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# How many failed sites to preview in logs
FAILED_PREVIEW_COUNT = 3


def load_urls_from_excel(excel_path: str) -> list[str]:
    try:
        df = pd.read_excel(excel_path, header=None)
        urls = [str(url).strip() for url in df.iloc[:, 0].dropna() if str(url).startswith("http")]
        logger.info("✅ %d URLs loaded", len(urls))
        return urls
    except Exception as exc:
        logger.exception("❌ Excel error: %s", exc)
        return []


def save_results(results: dict, output_path: str, lock_obj=None) -> None:
    """Write crawl results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_existing_results(output_path: str) -> dict:
    if not Path(output_path).exists():
        return {}
    try:
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_statistics(
    all_results: dict,
    total_duration: int,
    skipped_count: int,
    blacklisted_count: int,
    history_manager: CrawlHistoryManager | None = None,
) -> None:
    total_pages = sum(v["total_pages"] for v in all_results.values())
    successful = len([v for v in all_results.values() if v["total_pages"] > 0])
    total_chars = 0
    empty_count = 0

    for site_data in all_results.values():
        for page in site_data.get("pages", []):
            content = page.get("main_content", "")
            total_chars += len(content) if content else 0
            if not content:
                empty_count += 1

    logger.info("%s", "\n" + "=" * 80)
    logger.info("📊 STATISTICS")
    logger.info("%s", "=" * 80)
    logger.info("Processed sites  : %d", len(all_results))
    logger.info("Blacklisted       : %d", blacklisted_count)
    logger.info("Successful        : %d", successful)
    logger.info("Failed            : %d", len(all_results) - successful)
    logger.info("Total pages       : %d", total_pages)
    logger.info("Empty content     : %d", empty_count)
    logger.info("Total content     : %s chars", f"{total_chars:,}")
    logger.info("Avg per page      : %s chars", f"{total_chars // max(total_pages, 1):,}")
    logger.info("Duration          : %dm %ds", total_duration // 60, total_duration % 60)
    logger.info("Avg per site      : %ds", total_duration // max(len(all_results), 1))

    if history_manager and not FORCE_FULL_CRAWL:
        stats = history_manager.get_stats()
        logger.info("\n📊 Hash statistics:")
        logger.info("   Total URLs     : %s", stats["total_urls"])
        logger.info("   Main pages     : %s", stats["main_pages"])
        logger.info("   Sub pages      : %s", stats["sub_pages"])
        logger.info("   Failed         : %s", stats["failed"])
        logger.info("   Success rate   : %s", stats["success_rate"])
    logger.info("%s", "=" * 80)


async def _watchdog(crawler: SmartDeepCrawler, crawl_task: asyncio.Task, inactivity_timeout: int) -> None:
    """Cancel crawl if the crawler becomes inactive for too long."""
    check_interval = 10
    while not crawl_task.done():
        await asyncio.sleep(check_interval)
        if crawl_task.done():
            break
        elapsed = (datetime.now(tz=UTC) - crawler.last_activity).total_seconds()
        if elapsed > inactivity_timeout:
            logger.warning("⏱️ %ss inactivity — site is unresponsive, cancelling", inactivity_timeout)
            crawl_task.cancel()
            break


async def process_single_site(
    *,
    url: str,
    idx: int,
    total: int,
    shared_browser: SharedBrowser,
    history: CrawlHistoryManager,
    all_results: dict,
    results_lock: asyncio.Lock,
    save_lock: asyncio.Lock,
    site_semaphore: asyncio.Semaphore,
    ollama_model: str,
    max_depth: int,
    inactivity_timeout: int,
    crawl_mode: str,
) -> None:
    async with site_semaphore:
        logger.info("%s", "\n" + "=" * 70)
        logger.info("🌐 [%d/%d] %s", idx, total, url)
        logger.info("%s", "=" * 70)

        if url in BLACKLIST:
            logger.info("🚫 BLACKLISTED — skipped: %s", url)
            return

        site_start = datetime.now(tz=UTC)
        site_data = None

        try:
            agent = TrafilaturaOllamaAgent(
                model=ollama_model,
                shared_browser=shared_browser,
            )
            crawler = SmartDeepCrawler(agent, max_depth=max_depth)

            history_to_use = None if FORCE_FULL_CRAWL else history

            crawl_task = asyncio.create_task(crawler.smart_deep_crawl(url, history_manager=history_to_use))

            watchdog_task = asyncio.create_task(_watchdog(crawler, crawl_task, inactivity_timeout))

            try:
                results = await crawl_task
            except asyncio.CancelledError:
                results = []
            finally:
                watchdog_task.cancel()

            duration = (datetime.now(tz=UTC) - site_start).seconds

            if not results:
                site_data = {
                    "total_pages": 0,
                    "duration_seconds": duration,
                    "last_updated": datetime.now(tz=UTC).isoformat(),
                    "pages": [],
                    "error": f"inactivity_timeout ({inactivity_timeout}s)",
                }
                logger.warning("⏱️  INACTIVITY TIMEOUT [%d/%d]", idx, total)
                if ENABLE_INCREMENTAL and history:
                    history.mark_failed(url, f"inactivity_timeout ({inactivity_timeout}s)")
            else:
                site_data = {
                    "total_pages": len(results),
                    "duration_seconds": duration,
                    "last_updated": datetime.now().isoformat(),
                    "crawl_mode": crawl_mode,
                    "pages": [r.to_dict() for r in results],
                }

                if ENABLE_INCREMENTAL and history and results:
                    first = results[0]
                    if first.main_content:
                        history.update_history(
                            url,
                            first.main_content,
                            len(results),
                            {
                                "extraction_type": first.metadata.get("extraction_type"),
                                "method": first.metadata.get("method"),
                                "link_count": len(first.links),
                                "force_crawled": FORCE_FULL_CRAWL,
                            },
                        )

                successful = sum(1 for r in results if r.metadata.get("status") == "success")
                total_chars = sum(len(r.main_content or "") for r in results)
                total_links = sum(len(r.links) for r in results)

                logger.info("✅ COMPLETED [%d/%d]", idx, total)
                logger.info("   Pages       : %d", len(results))
                logger.info("   Success      : %d/%d", successful, len(results))
                logger.info("   Content      : %s chars", f"{total_chars:,}")
                logger.info("   Links        : %d", total_links)
                logger.info("   Duration     : %ds (%dm %ds)", duration, duration // 60, duration % 60)

        except Exception as exc:
            err_msg = str(exc)[:200]
            duration = (datetime.now(tz=UTC) - site_start).seconds
            site_data = {
                "total_pages": 0,
                "duration_seconds": duration,
                "last_updated": datetime.now(tz=UTC).isoformat(),
                "pages": [],
                "error": err_msg,
            }
            logger.exception("❌ ERROR [%d/%d]: %s", idx, total, err_msg)
            if ENABLE_INCREMENTAL and history:
                history.mark_failed(url, err_msg)

        async with results_lock:
            all_results[url] = site_data

        async with save_lock:
            save_results(all_results, OUTPUT_FILE)


async def main() -> None:
    logger.info("%s", "=" * 80)
    logger.info("🚀 CONCURRENT HASH-BASED CRAWLING SYSTEM v2")
    mode_label = "FORCE FULL CRAWL" if FORCE_FULL_CRAWL else "Hash-based"
    logger.info("   Mode            : %s", mode_label)
    logger.info("   Concurrent sites : %d", CONCURRENT_SITES)
    logger.info("%s", "=" * 80)

    excel_path = str(Path(__file__).parent / "config" / "sites.xlsx")
    max_depth = 1
    ollama_model = "gemma3:12b"

    logger.info("\n🔧 Settings:")
    logger.info("   Model            : %s", ollama_model)
    logger.info("   Max Depth        : %d", max_depth)
    logger.info("   Inactivity TO    : %ds", INACTIVITY_TIMEOUT)
    logger.info("   Concurrent       : %d sites", CONCURRENT_SITES)
    logger.info("   Output           : %s", OUTPUT_FILE)
    logger.info("   Incremental      : %s", ENABLE_INCREMENTAL)
    logger.info("   Force Full       : %s", FORCE_FULL_CRAWL)
    logger.info("   Blacklist        : %d sites", len(BLACKLIST))

    logger.info("\n📊 Loading URLs: %s", excel_path)
    urls = load_urls_from_excel(excel_path)
    if not urls:
        logger.error("❌ No URLs found")
        return

    history = CrawlHistoryManager(HISTORY_FILE) if ENABLE_INCREMENTAL else None
    all_results = load_existing_results(OUTPUT_FILE)

    logger.info("📂 Existing history : %d sites", len(history.history) if history else 0)
    logger.info("📂 Existing results : %d sites", len(all_results))
    logger.info("📊 URLs to process  : %d", len(urls))

    if FORCE_FULL_CRAWL:
        est = len(urls) * 10
    else:
        est = len(urls) * 3 // CONCURRENT_SITES
    logger.info("⏱️  Estimated time   : ~%d minutes (concurrent=%d)", est, CONCURRENT_SITES)

    logger.info("\n🚀 Crawl starting...")
    logger.info("%s", "=" * 80)

    shared_browser = SharedBrowser(max_concurrent_pages=CONCURRENT_SITES * 3)
    await shared_browser.start()

    site_semaphore = asyncio.Semaphore(CONCURRENT_SITES)
    results_lock = asyncio.Lock()
    save_lock = asyncio.Lock()

    crawl_mode = "FORCE_FULL_CRAWL" if FORCE_FULL_CRAWL else "hash-based"
    overall_start = datetime.now(tz=UTC)

    blacklisted_count = sum(1 for u in urls if u in BLACKLIST)

    tasks = [
        process_single_site(
            url=url,
            idx=i,
            total=len(urls),
            shared_browser=shared_browser,
            history=history,
            all_results=all_results,
            results_lock=results_lock,
            save_lock=save_lock,
            site_semaphore=site_semaphore,
            ollama_model=ollama_model,
            max_depth=max_depth,
            inactivity_timeout=INACTIVITY_TIMEOUT,
            crawl_mode=crawl_mode,
        )
        for i, url in enumerate(urls, 1)
    ]

    await asyncio.gather(*tasks)

    await shared_browser.stop()
    save_results(all_results, OUTPUT_FILE)

    total_duration = (datetime.now() - overall_start).seconds
    print_statistics(all_results, total_duration, 0, blacklisted_count, history)

    logger.info("\n📁 Results  : %s", OUTPUT_FILE)
    if history:
        logger.info("📜 History  : %s", HISTORY_FILE)

    failed_sites: dict[str, list[str]] = {}
    for url, data in all_results.items():
        if data.get("total_pages", 0) == 0 and "error" in data:
            err = data["error"]
            failed_sites.setdefault(err, []).append(url)

    if failed_sites:
        logger.warning("\n⚠️  FAILED SITES:")
        for err, sites in failed_sites.items():
            logger.warning("\n   %s (%d sites):", err, len(sites))
            for site in sites[:3]:
                logger.info("      - %s", site)
            if len(sites) > 3:
                logger.info("      ... and %d more", len(sites) - 3)

    successful_sites = len([v for v in all_results.values() if v.get("total_pages", 0) > 0])
    pct = successful_sites * 100 // len(urls) if urls else 0
    logger.info("\n✅ Crawl completed!")
    logger.info("   Success: %d/%d sites (%d%%)", successful_sites, len(urls), pct)

    if not FORCE_FULL_CRAWL:
        logger.info("\n💡 Next run will skip sites with matching hashes.")
    else:
        logger.info("\n💡 Set FORCE_FULL_CRAWL=False in config.py for hash optimization.")


if __name__ == "__main__":
    asyncio.run(main())

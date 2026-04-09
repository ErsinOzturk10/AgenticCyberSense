"""
Main — Concurrent + SharedBrowser + Hash Normalizasyonu (v2)

Değişiklikler:
  - asyncio.Semaphore ile N site eş zamanlı işleniyor (varsayılan 3)
  - SharedBrowser: tek Chromium process, tüm siteler paylaşıyor
  - Sonuçlar asyncio.Lock ile thread-safe kaydediliyor
  - İlerleme bilgisi concurrent modda da düzgün çalışıyor
"""

import sys
import asyncio
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from trafilatura_ollama_agent import TrafilaturaOllamaAgent, SharedBrowser
from deep_crawler_trafilatura import SmartDeepCrawler
from crawl_history_manager import CrawlHistoryManager
from typing import List, Dict
from config import (
    INACTIVITY_TIMEOUT,
    BLACKLIST, OUTPUT_FILE, HISTORY_FILE,
    ENABLE_INCREMENTAL, FORCE_FULL_CRAWL,
    CONCURRENT_SITES,   # YENİ — config.py'ye ekleyin (aşağıda açıklanıyor)
)


# ──────────────────────────────────────────────────────────────────── #
#  Yardımcılar                                                          #
# ──────────────────────────────────────────────────────────────────── #

def load_urls_from_excel(excel_path: str) -> List[str]:
    try:
        df   = pd.read_excel(excel_path, header=None)
        urls = [
            str(u).strip() for u in df.iloc[:, 0].dropna()
            if str(u).startswith("http")
        ]
        print(f"✅ {len(urls)} URL yüklendi")
        return urls
    except Exception as e:
        print(f"❌ Excel hatası: {e}")
        return []


def save_results(results: dict, output_path: str, lock_obj=None):
    """JSON'a yaz — lock verilmişse kullan (concurrent modda)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_existing_results(output_path: str) -> dict:
    if not Path(output_path).exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_statistics(
    all_results: dict,
    total_duration: int,
    skipped_count: int,
    blacklisted_count: int,
    history_manager: CrawlHistoryManager = None,
):
    total_pages   = sum(v["total_pages"] for v in all_results.values())
    successful    = len([v for v in all_results.values() if v["total_pages"] > 0])
    total_chars   = 0
    empty_count   = 0

    for site_data in all_results.values():
        for page in site_data.get("pages", []):
            content = page.get("main_content", "")
            total_chars += len(content) if content else 0
            if not content:
                empty_count += 1

    print("\n" + "=" * 80)
    print("📊 İSTATİSTİKLER")
    print("=" * 80)
    print(f"İşlenen site       : {len(all_results)}")
    print(f"Kara liste         : {blacklisted_count}")
    print(f"Başarılı           : {successful}")
    print(f"Başarısız          : {len(all_results) - successful}")
    print(f"Toplam sayfa       : {total_pages}")
    print(f"Boş içerik         : {empty_count}")
    print(f"Toplam içerik      : {total_chars:,} karakter")
    print(f"Sayfa başı ort.    : {total_chars // max(total_pages, 1):,} karakter")
    print(f"Süre               : {total_duration // 60}d {total_duration % 60}s")
    print(f"Site başı ort.     : {total_duration // max(len(all_results), 1)}s")

    if history_manager and not FORCE_FULL_CRAWL:
        stats = history_manager.get_stats()
        print(f"\n📊 Hash İstatistikleri:")
        print(f"   Toplam URL     : {stats['total_urls']}")
        print(f"   Ana sayfa      : {stats['main_pages']}")
        print(f"   Alt sayfa      : {stats['sub_pages']}")
        print(f"   Başarısız      : {stats['failed']}")
        print(f"   Başarı oranı   : {stats['success_rate']}")
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────── #
#  Tek site işleme (concurrent'ta çağrılır)                             #
# ──────────────────────────────────────────────────────────────────── #

async def _watchdog(crawler: SmartDeepCrawler, crawl_task: asyncio.Task, inactivity_timeout: int):
    """
    Crawler'ın last_activity'sini izler.
    INACTIVITY_TIMEOUT saniye boyunca hiç yeni sayfa çekilmezse
    crawl_task'ı iptal eder.
    Veri gelmeye devam ettiği sürece asla müdahale etmez.
    """
    CHECK_INTERVAL = 10  # Her 10 saniyede kontrol et
    while not crawl_task.done():
        await asyncio.sleep(CHECK_INTERVAL)
        if crawl_task.done():
            break
        elapsed = (datetime.now() - crawler.last_activity).total_seconds()
        if elapsed > inactivity_timeout:
            print(f"\n⏱️  {inactivity_timeout}s hareketsizlik — site yanıt vermiyor, iptal ediliyor")
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
):
    async with site_semaphore:
        print(f"\n{'='*70}")
        print(f"🌐 [{idx}/{total}] {url}")
        print(f"{'='*70}")

        if url in BLACKLIST:
            print(f"🚫 KARA LİSTE — atlanıyor")
            return

        site_start = datetime.now()
        site_data  = None

        try:
            agent   = TrafilaturaOllamaAgent(
                model=ollama_model,
                shared_browser=shared_browser,
            )
            crawler = SmartDeepCrawler(agent, max_depth=max_depth)

            history_to_use = None if FORCE_FULL_CRAWL else history

            # Crawl görevini oluştur (henüz çalıştırma)
            crawl_task = asyncio.create_task(
                crawler.smart_deep_crawl(url, history_manager=history_to_use)
            )

            # Watchdog: sadece takılırsa müdahale eder
            watchdog_task = asyncio.create_task(
                _watchdog(crawler, crawl_task, inactivity_timeout)
            )

            try:
                results = await crawl_task
            except asyncio.CancelledError:
                results = []  # watchdog iptal etti
            finally:
                watchdog_task.cancel()

            duration = (datetime.now() - site_start).seconds

            if not results:
                site_data = {
                    "total_pages":      0,
                    "duration_seconds": duration,
                    "last_updated":     datetime.now().isoformat(),
                    "pages":            [],
                    "error":            f"inactivity_timeout ({inactivity_timeout}s)",
                }
                print(f"\n⏱️  HAREKETSİZLİK TIMEOUT [{idx}/{total}]")
                if ENABLE_INCREMENTAL and history:
                    history.mark_failed(url, f"inactivity_timeout ({inactivity_timeout}s)")
            else:
                site_data = {
                    "total_pages":      len(results),
                    "duration_seconds": duration,
                    "last_updated":     datetime.now().isoformat(),
                    "crawl_mode":       crawl_mode,
                    "pages":            [r.to_dict() for r in results],
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
                                "method":          first.metadata.get("method"),
                                "link_count":      len(first.links),
                                "force_crawled":   FORCE_FULL_CRAWL,
                            },
                        )

                successful  = sum(1 for r in results if r.metadata.get("status") == "success")
                total_chars = sum(len(r.main_content or "") for r in results)
                total_links = sum(len(r.links) for r in results)

                print(f"\n✅ TAMAMLANDI [{idx}/{total}]")
                print(f"   Sayfa       : {len(results)}")
                print(f"   Başarı      : {successful}/{len(results)}")
                print(f"   İçerik      : {total_chars:,} karakter")
                print(f"   Link        : {total_links}")
                print(f"   Süre        : {duration}s ({duration//60}dk {duration%60}s)")

        except Exception as e:
            err_msg   = str(e)[:200]
            duration  = (datetime.now() - site_start).seconds
            site_data = {
                "total_pages":      0,
                "duration_seconds": duration,
                "last_updated":     datetime.now().isoformat(),
                "pages":            [],
                "error":            err_msg,
            }
            print(f"\n❌ HATA [{idx}/{total}]: {err_msg}")
            if ENABLE_INCREMENTAL and history:
                history.mark_failed(url, err_msg)

        # Thread-safe kayıt
        async with results_lock:
            all_results[url] = site_data

        async with save_lock:
            save_results(all_results, OUTPUT_FILE)



# ──────────────────────────────────────────────────────────────────── #
#  Main                                                                 #
# ──────────────────────────────────────────────────────────────────── #

async def main():
    print("=" * 80)
    print("🚀 CONCURRENT HASH-BASED CRAWLING SYSTEM v2")
    mode_label = "FORCE FULL CRAWL" if FORCE_FULL_CRAWL else "Hash-based (akıllı)"
    print(f"   Mod              : {mode_label}")
    print(f"   Eş zamanlı site  : {CONCURRENT_SITES}")
    print("=" * 80)

    EXCEL_PATH   = str(Path(__file__).parent / "config" / "sites.xlsx")
    MAX_DEPTH    = 1
    OLLAMA_MODEL = "gemma3:12b"

    print(f"\n🔧 Ayarlar:")
    print(f"   Model            : {OLLAMA_MODEL}")
    print(f"   Max Depth        : {MAX_DEPTH}")
    print(f"   Hareketsizlik TO : {INACTIVITY_TIMEOUT}s (veri gelirse sınır yok)")
    print(f"   Eş zamanlı       : {CONCURRENT_SITES} site")
    print(f"   Output           : {OUTPUT_FILE}")
    print(f"   Incremental      : {ENABLE_INCREMENTAL}")
    print(f"   Force Full       : {FORCE_FULL_CRAWL}")
    print(f"   Kara liste       : {len(BLACKLIST)} site")

    print(f"\n📊 URL'ler yükleniyor: {EXCEL_PATH}")
    urls = load_urls_from_excel(EXCEL_PATH)
    if not urls:
        print("❌ URL bulunamadı!")
        return

    history     = CrawlHistoryManager(HISTORY_FILE) if ENABLE_INCREMENTAL else None
    all_results = load_existing_results(OUTPUT_FILE)

    print(f"📂 Mevcut geçmiş  : {len(history.history) if history else 0} site")
    print(f"📂 Mevcut sonuçlar: {len(all_results)} site")
    print(f"📊 İşlenecek URL  : {len(urls)}")

    if FORCE_FULL_CRAWL:
        est = len(urls) * 10
    else:
        est = len(urls) * 3 // CONCURRENT_SITES
    print(f"⏱️  Tahmini süre   : ~{est} dakika (concurrent={CONCURRENT_SITES})")

    print(f"\n🚀 Crawl başlıyor...")
    print("=" * 80)

    # Paylaşımlı browser — tüm siteler bu instance'ı kullanır
    shared_browser = SharedBrowser(max_concurrent_pages=CONCURRENT_SITES * 3)
    await shared_browser.start()

    # Eş zamanlılık kontrolleri
    site_semaphore = asyncio.Semaphore(CONCURRENT_SITES)
    results_lock   = asyncio.Lock()
    save_lock      = asyncio.Lock()

    crawl_mode     = "FORCE_FULL_CRAWL" if FORCE_FULL_CRAWL else "hash-based"
    overall_start  = datetime.now()

    blacklisted_count = sum(1 for u in urls if u in BLACKLIST)

    # Tüm siteleri paralel olarak başlat
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
            ollama_model=OLLAMA_MODEL,
            max_depth=MAX_DEPTH,
            inactivity_timeout=INACTIVITY_TIMEOUT,
            crawl_mode=crawl_mode,
        )
        for i, url in enumerate(urls, 1)
    ]

    await asyncio.gather(*tasks)

    # Browser kapat
    await shared_browser.stop()

    # Son kayıt
    save_results(all_results, OUTPUT_FILE)

    total_duration = (datetime.now() - overall_start).seconds
    print_statistics(all_results, total_duration, 0, blacklisted_count, history)

    print(f"\n📁 Sonuçlar : {OUTPUT_FILE}")
    if history:
        print(f"📜 Geçmiş   : {HISTORY_FILE}")

    # Başarısız site analizi
    failed_sites: Dict[str, List[str]] = {}
    for url, data in all_results.items():
        if data.get("total_pages", 0) == 0 and "error" in data:
            err = data["error"]
            failed_sites.setdefault(err, []).append(url)

    if failed_sites:
        print(f"\n⚠️  BAŞARISIZ SİTELER:")
        for err, sites in failed_sites.items():
            print(f"\n   {err} ({len(sites)} site):")
            for s in sites[:3]:
                print(f"      - {s}")
            if len(sites) > 3:
                print(f"      ... ve {len(sites) - 3} tane daha")

    successful_sites = len([v for v in all_results.values() if v.get("total_pages", 0) > 0])
    pct = successful_sites * 100 // len(urls) if urls else 0
    print(f"\n✅ Crawl tamamlandı!")
    print(f"   Başarı: {successful_sites}/{len(urls)} site (%{pct})")

    if not FORCE_FULL_CRAWL:
        print(f"\n💡 Sonraki çalıştırmada hash eşleşen siteler atlanacak.")
    else:
        print(f"\n💡 Hash optimizasyonu için config.py'de FORCE_FULL_CRAWL=False yapın.")


if __name__ == "__main__":
    asyncio.run(main())
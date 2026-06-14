# ruff: noqa: RUF001, BLE001
"""API-based data collector for cybersecurity threat intelligence.

Collects data from public APIs (no API key required for initial sources)
and saves to api_results.json in the same format as crawling output.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_FILE = Path(__file__).parent / "output" / "api_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(url: str, content: str, title: str = "", links: list[str] | None = None) -> dict:
    """Create a page dict with the same schema as latest_results.json."""
    return {
        "url": url,
        "title": title,
        "main_content": content,
        "metadata": {
            "extraction_type": "api",
            "method": "api",
            "model": None,
            "status": "success",
            "content_length": len(content),
            "link_count": len(links or []),
            "duration_seconds": 0,
        },
        "links": links or [],
        "structured_data": {},
    }


def _make_site(pages: list[dict], duration_seconds: int = 0) -> dict:
    """Create a site dict with the same schema as latest_results.json."""
    return {
        "total_pages": len(pages),
        "duration_seconds": duration_seconds,
        "last_updated": datetime.now(tz=UTC).isoformat(),
        "crawl_mode": "api",
        "pages": pages,
    }


def _safe_get(url: str, **kwargs: object) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=30, **kwargs)  # type: ignore[arg-type]
        r.raise_for_status()
    except Exception:
        logger.exception("GET %s failed", url)
        return None
    else:
        return r


def _safe_post(url: str, **kwargs: object) -> requests.Response | None:
    try:
        r = requests.post(url, timeout=30, **kwargs)  # type: ignore[arg-type]
        r.raise_for_status()
    except Exception:
        logger.exception("POST %s failed", url)
        return None
    else:
        return r


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_cisa_kev() -> tuple[str, dict]:
    """CISA KEV — known exploited vulnerabilities, no API key needed."""
    logger.info("📡 CISA KEV başlatıldı...")
    api_url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

    r = _safe_get(api_url)
    if not r:
        logger.error("❌ CISA KEV: bağlantı kurulamadı")
        return "https://www.cisa.gov/known-exploited-vulnerabilities-", _make_site([])

    data = r.json()
    vulns = data.get("vulnerabilities", [])
    logger.info("📥 CISA KEV: %d CVE geldi", len(vulns))

    pages = []
    for vuln in vulns:
        content = (
            f"CVE ID: {vuln.get('cveID', '')}\n"
            f"Vendor: {vuln.get('vendorProject', '')}\n"
            f"Product: {vuln.get('product', '')}\n"
            f"Vulnerability Name: {vuln.get('vulnerabilityName', '')}\n"
            f"Description: {vuln.get('shortDescription', '')}\n"
            f"Date Added: {vuln.get('dateAdded', '')}\n"
            f"Required Action: {vuln.get('requiredAction', '')}\n"
        )
        pages.append(_make_page(
            url=f"https://www.cisa.gov/known-exploited-vulnerabilities-#{vuln.get('cveID', '')}",
            content=content,
            title=vuln.get("vulnerabilityName", ""),
        ))

    logger.info("✅ CISA KEV tamamlandı: %d sayfa", len(pages))
    return "https://www.cisa.gov/known-exploited-vulnerabilities-", _make_site(pages)


def collect_mitre_attack() -> tuple[str, dict]:
    """MITRE ATT&CK — enterprise techniques via TAXII."""
    logger.info("📡 MITRE ATT&CK Techniques başlatıldı...")
    try:
        from attackcti import attack_client  # noqa: PLC0415
        lift = attack_client()
        techniques = lift.get_enterprise_techniques()
        pages = []

        for t in techniques:
            name = t.get("name", "")
            desc = t.get("description", "")
            tid = ""
            tactic = ""

            for ref in t.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    tid = ref.get("external_id", "")

            kill_chain = t.get("kill_chain_phases", [])
            if kill_chain:
                tactic = kill_chain[0].get("phase_name", "")

            content = (
                f"Technique ID: {tid}\n"
                f"Name: {name}\n"
                f"Tactic: {tactic}\n"
                f"Description: {desc}\n"
                f"Platforms: {', '.join(t.get('x_mitre_platforms', []))}\n"
            )
            pages.append(_make_page(
                url=f"https://attack.mitre.org/techniques/{tid}/",
                content=content,
                title=f"{tid} — {name}",
            ))

        logger.info("✅ MITRE ATT&CK Techniques: %d teknik", len(pages))
        return "https://attack.mitre.org/", _make_site(pages)

    except ImportError:
        logger.warning("⚠️ attackcti yüklü değil: pip install attackcti")
        return "https://attack.mitre.org/", _make_site([])
    except Exception:
        logger.exception("❌ MITRE ATT&CK Techniques hatası")
        return "https://attack.mitre.org/", _make_site([])


def collect_mitre_groups() -> tuple[str, dict]:
    """MITRE ATT&CK — threat actor groups via TAXII."""
    logger.info("📡 MITRE ATT&CK Groups başlatıldı...")
    try:
        from attackcti import attack_client  # noqa: PLC0415
        lift = attack_client()
        groups = lift.get_groups()
        pages = []

        for g in groups:
            name = g.get("name", "")
            desc = g.get("description", "")
            gid = ""
            aliases = g.get("aliases", [])

            for ref in g.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    gid = ref.get("external_id", "")

            content = (
                f"Group ID: {gid}\n"
                f"Name: {name}\n"
                f"Aliases: {', '.join(aliases)}\n"
                f"Description: {desc}\n"
            )
            pages.append(_make_page(
                url=f"https://attack.mitre.org/groups/{gid}/",
                content=content,
                title=f"{gid} — {name}",
            ))

        logger.info("✅ MITRE ATT&CK Groups: %d grup", len(pages))
        return "https://attack.mitre.org/versions/v18/groups/", _make_site(pages)

    except ImportError:
        logger.warning("⚠️ attackcti yüklü değil: pip install attackcti")
        return "https://attack.mitre.org/versions/v18/groups/", _make_site([])
    except Exception:
        logger.exception("❌ MITRE ATT&CK Groups hatası")
        return "https://attack.mitre.org/versions/v18/groups/", _make_site([])


def collect_mitre_software() -> tuple[str, dict]:
    """MITRE ATT&CK — malware and tools via TAXII."""
    logger.info("📡 MITRE ATT&CK Software başlatıldı...")
    try:
        from attackcti import attack_client  # noqa: PLC0415
        lift = attack_client()
        software_list = lift.get_software()
        pages = []

        for s in software_list:
            name = s.get("name", "")
            desc = s.get("description", "")
            sid = ""
            sw_type = s.get("type", "")

            for ref in s.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    sid = ref.get("external_id", "")

            content = (
                f"Software ID: {sid}\n"
                f"Name: {name}\n"
                f"Type: {sw_type}\n"
                f"Platforms: {', '.join(s.get('x_mitre_platforms', []))}\n"
                f"Description: {desc}\n"
            )
            pages.append(_make_page(
                url=f"https://attack.mitre.org/software/{sid}/",
                content=content,
                title=f"{sid} — {name}",
            ))

        logger.info("✅ MITRE ATT&CK Software: %d yazılım", len(pages))
        return "https://attack.mitre.org/versions/v18/software/", _make_site(pages)

    except ImportError:
        logger.warning("⚠️ attackcti yüklü değil: pip install attackcti")
        return "https://attack.mitre.org/versions/v18/software/", _make_site([])
    except Exception:
        logger.exception("❌ MITRE ATT&CK Software hatası")
        return "https://attack.mitre.org/versions/v18/software/", _make_site([])


def collect_mitre_campaigns() -> tuple[str, dict]:
    """MITRE ATT&CK — campaigns via TAXII."""
    logger.info("📡 MITRE ATT&CK Campaigns başlatıldı...")
    try:
        from attackcti import attack_client  # noqa: PLC0415
        lift = attack_client()
        campaigns = lift.get_campaigns()
        pages = []

        for c in campaigns:
            name = c.get("name", "")
            desc = c.get("description", "")
            cid = ""
            first_seen = c.get("x_mitre_first_seen_citation", "")

            for ref in c.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    cid = ref.get("external_id", "")

            content = (
                f"Campaign ID: {cid}\n"
                f"Name: {name}\n"
                f"First Seen: {first_seen}\n"
                f"Description: {desc}\n"
            )
            pages.append(_make_page(
                url=f"https://attack.mitre.org/campaigns/{cid}/",
                content=content,
                title=f"{cid} — {name}",
            ))

        logger.info("✅ MITRE ATT&CK Campaigns: %d kampanya", len(pages))
        return "https://attack.mitre.org/versions/v18/campaigns/", _make_site(pages)

    except ImportError:
        logger.warning("⚠️ attackcti yüklü değil: pip install attackcti")
        return "https://attack.mitre.org/versions/v18/campaigns/", _make_site([])
    except Exception:
        logger.exception("❌ MITRE ATT&CK Campaigns hatası")
        return "https://attack.mitre.org/versions/v18/campaigns/", _make_site([])


def collect_github_repos() -> tuple[str, dict]:
    """GitHub — README content from threat intelligence repos."""
    logger.info("📡 GitHub Repos başlatıldı...")

    repos = [
        ("hslatman", "awesome-threat-intelligence", "https://github.com/hslatman/awesome-threat-intelligen"),
        ("crits", "crits", "https://github.com/crits"),
        ("opencti-platform", "opencti", "https://github.com/topics/opencti"),
        ("mitre", "cti", "https://github.com/mitre/cti"),
    ]

    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "AgenticCyberSense"}
    pages = []

    for owner, repo, source_url in repos:
        logger.info("  → %s/%s", owner, repo)
        r = _safe_get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=headers)
        if not r:
            logger.warning("  ⚠️ README alınamadı: %s/%s", owner, repo)
            continue

        encoded = r.json().get("content", "")
        if not encoded:
            continue

        try:
            readme_text = base64.b64decode(encoded).decode("utf-8", errors="ignore")
        except Exception:
            logger.warning("  ⚠️ README decode hatası: %s/%s", owner, repo)
            continue

        pages.append(_make_page(
            url=source_url,
            content=readme_text[:50000],
            title=f"GitHub: {owner}/{repo}",
        ))
        logger.info("  ✅ %s/%s: %d karakter", owner, repo, len(readme_text))

    logger.info("✅ GitHub Repos tamamlandı: %d repo", len(pages))
    return "https://github.com/hslatman/awesome-threat-intelligen", _make_site(pages)


def collect_huggingface_datasets() -> tuple[str, dict]:
    """HuggingFace — MITRE-related cybersecurity datasets."""
    logger.info("📡 HuggingFace Datasets başlatıldı...")

    datasets = [
        ("Zainabsa99/mitre_attack", "https://huggingface.co/datasets/Zainabsa99/mitre_atta"),
    ]

    pages = []

    for dataset_id, source_url in datasets:
        logger.info("  → %s", dataset_id)

        r = _safe_get(f"https://huggingface.co/api/datasets/{dataset_id}")
        if not r:
            logger.warning("  ⚠️ Dataset bilgisi alınamadı: %s", dataset_id)
            continue

        info = r.json()
        description = info.get("cardData", {}).get("description") or info.get("description") or ""
        tags = info.get("tags", [])
        downloads = info.get("downloads", 0)

        rows_r = _safe_get(
            f"https://datasets-server.huggingface.co/first-rows"
            f"?dataset={dataset_id}&config=default&split=train"
        )
        sample_text = ""
        if rows_r:
            for row in (rows_r.json().get("rows", []))[:5]:
                sample_text += json.dumps(row.get("row", {}), ensure_ascii=False) + "\n"

        content = (
            f"Dataset: {dataset_id}\n"
            f"Description: {description}\n"
            f"Tags: {', '.join(tags)}\n"
            f"Downloads: {downloads}\n"
            f"\nSample rows:\n{sample_text}"
        )

        pages.append(_make_page(
            url=source_url,
            content=content,
            title=f"HuggingFace Dataset: {dataset_id}",
        ))
        logger.info("  ✅ %s alındı", dataset_id)

    logger.info("✅ HuggingFace Datasets tamamlandı: %d dataset", len(pages))
    return "https://huggingface.co/datasets/Zainabsa99/mitre_atta", _make_site(pages)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLLECTORS = [
    collect_cisa_kev,
    collect_mitre_attack,
    collect_mitre_groups,
    collect_mitre_software,
    collect_mitre_campaigns,
    collect_github_repos,
    collect_huggingface_datasets,
]


def run() -> None:
    """Run all collectors, save results to disk, and ingest into ChromaDB."""
    logger.info("=" * 60)
    logger.info("🚀 API Collector başlatıldı")
    logger.info("=" * 60)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            all_results = json.load(f)
        logger.info("📂 Mevcut kayıt: %d kaynak", len(all_results))
    else:
        all_results = {}

    for collector in COLLECTORS:
        try:
            key, site_data = collector()
            all_results[key] = site_data
            logger.info("💾 Kaydedildi: %s (%d sayfa)", key, site_data["total_pages"])
        except Exception:
            logger.exception("❌ Collector hatası (%s)", collector.__name__)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_pages = sum(v["total_pages"] for v in all_results.values())
    total_chars = sum(
        len(p.get("main_content", "") or "")
        for v in all_results.values()
        for p in v.get("pages", [])
    )

    logger.info("=" * 60)
    logger.info("📊 ÖZET")
    logger.info("   Kaynak sayisi  : %d", len(all_results))
    logger.info("   Toplam sayfa   : %d", total_pages)
    logger.info("   Toplam karakter: %s", f"{total_chars:,}")
    logger.info("   Cikti          : %s", OUTPUT_FILE)
    logger.info("=" * 60)

    logger.info("\nRAG indeksi guncelleniyor...")
    try:
        from agenticcybersense.web_crawler.rag_ingest import ingest_crawler_json  # noqa: PLC0415
        rag_stats = ingest_crawler_json(str(OUTPUT_FILE))
        logger.info("RAG indeksi guncellendi: %s", rag_stats)
    except Exception:
        logger.exception("RAG ingest basarisiz — veriler diske kaydedildi")


if __name__ == "__main__":
    run()

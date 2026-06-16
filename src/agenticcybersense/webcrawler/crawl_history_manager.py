"""
Crawl History Manager — SQLite + Akıllı Hash Normalizasyonu (v3)

Değişiklikler:
  - should_crawl_subpage() eklendi: alt sayfalar da bağımsız hash kontrolüne tabi
  - update_subpage() eklendi: alt sayfa hash'ini kaydet
  - get_stats() genişletildi: toplam alt sayfa sayısını da gösterir
"""

import sqlite3
import hashlib
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple


class CrawlHistoryManager:

    # ------------------------------------------------------------------ #
    # Init / DB                                                            #
    # ------------------------------------------------------------------ #

    def __init__(self, db_file: str = "crawl_history.db"):
        self.db_file = db_file
        self._init_db()
        self._migrate_json_if_exists()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crawl_history (
                    url            TEXT PRIMARY KEY,
                    content_hash   TEXT,
                    last_crawled   TEXT,
                    last_checked   TEXT,
                    content_length INTEGER,
                    total_pages    INTEGER,
                    status         TEXT DEFAULT 'success',
                    error          TEXT,
                    metadata       TEXT,
                    -- 'main' = ana sayfa, 'sub' = alt sayfa
                    page_type      TEXT DEFAULT 'main'
                )
            """)
            # Eski tabloya page_type kolonu yoksa ekle (migration)
            try:
                conn.execute("ALTER TABLE crawl_history ADD COLUMN page_type TEXT DEFAULT 'main'")
            except Exception:
                pass  # zaten var
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON crawl_history(url)")
            conn.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # Migration: eski JSON → SQLite                                        #
    # ------------------------------------------------------------------ #

    def _migrate_json_if_exists(self):
        json_file = Path(self.db_file).with_suffix(".json")
        fallback  = Path("crawl_history.json")

        for src in [json_file, fallback]:
            if src.exists():
                try:
                    with open(src, "r", encoding="utf-8") as f:
                        old_data: Dict = json.load(f)
                    migrated = 0
                    with self._connect() as conn:
                        for url, v in old_data.items():
                            conn.execute("""
                                INSERT OR IGNORE INTO crawl_history
                                    (url, content_hash, last_crawled, last_checked,
                                     content_length, total_pages, status, metadata, page_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'main')
                            """, (
                                url,
                                v.get("content_hash"),
                                v.get("last_crawled"),
                                v.get("last_checked"),
                                v.get("content_length"),
                                v.get("total_pages"),
                                v.get("status", "success"),
                                json.dumps(v.get("metadata") or {}),
                            ))
                            migrated += 1
                    src.rename(src.with_suffix(".json.bak"))
                    print(f"✅ Migration: {migrated} kayıt JSON→SQLite taşındı ({src})")
                except Exception as e:
                    print(f"⚠️  Migration hatası ({src}): {e}")

    # ------------------------------------------------------------------ #
    # Hash — normalize et, sonra hash'le                                  #
    # ------------------------------------------------------------------ #

    _PATTERNS = [
        re.compile(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"),
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s]*"),
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
            r"\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(?:AM|PM)?\b", re.IGNORECASE),
        re.compile(
            r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d{4,}\b"),
        re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE),
    ]

    def normalize_content(self, content: str) -> str:
        text = content
        for pat in self._PATTERNS:
            text = pat.sub("", text)
        return " ".join(text.lower().split())

    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(self.normalize_content(content).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Ana sayfa kararı                                                    #
    # ------------------------------------------------------------------ #

    def should_deep_crawl(self, url: str, current_content: str) -> Tuple[bool, str]:
        current_hash = self.compute_hash(current_content)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash, status FROM crawl_history WHERE url = ?", (url,)
            ).fetchone()

        if row is None:
            print(f"    🆕 Yeni site — full deep crawl")
            return True, "new_site"

        stored_hash = row["content_hash"]
        if not stored_hash:
            print(f"    ⚠️  Hash yok — full deep crawl")
            return True, "no_hash"

        if current_hash != stored_hash:
            print(f"    🔄 Ana sayfa değişmiş — deep crawl")
            print(f"       Eski : {stored_hash[:16]}...")
            print(f"       Yeni : {current_hash[:16]}...")
            return True, "content_changed"

        print(f"    ✅ Ana sayfa değişmemiş — deep crawl atlanıyor")
        print(f"       Hash : {current_hash[:16]}...")
        self._touch_last_checked(url)
        return False, "unchanged"

    # ------------------------------------------------------------------ #
    # Alt sayfa kararı  ← YENİ                                           #
    # ------------------------------------------------------------------ #

    def should_crawl_subpage(self, url: str, current_content: str) -> Tuple[bool, str]:
        """
        Alt sayfa için hash kontrolü.
        Returns (should_process: bool, reason: str)

        Ana sayfadan farklı olarak:
        - 'new_subpage'    : daha önce görülmemiş
        - 'subpage_changed': içerik değişmiş
        - 'subpage_unchanged': aynı → atla
        """
        current_hash = self.compute_hash(current_content)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM crawl_history WHERE url = ?", (url,)
            ).fetchone()

        if row is None:
            return True, "new_subpage"

        stored_hash = row["content_hash"]
        if not stored_hash:
            return True, "no_hash"

        if current_hash != stored_hash:
            return True, "subpage_changed"

        # Aynı → son kontrol zamanını güncelle, atla
        self._touch_last_checked(url)
        return False, "subpage_unchanged"

    def update_subpage(self, url: str, content: str):
        """Alt sayfa hash'ini kaydet / güncelle."""
        content_hash = self.compute_hash(content)
        now          = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO crawl_history
                    (url, content_hash, last_crawled, last_checked,
                     content_length, status, page_type)
                VALUES (?, ?, ?, ?, ?, 'success', 'sub')
            """, (url, content_hash, now, now, len(content)))

    # ------------------------------------------------------------------ #
    # Güncelleme / Hata işaretleme                                        #
    # ------------------------------------------------------------------ #

    def _touch_last_checked(self, url: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE crawl_history SET last_checked = ? WHERE url = ?",
                (datetime.now().isoformat(), url),
            )

    def update_history(self, url: str, content: str, total_pages: int, metadata: Dict = None):
        content_hash = self.compute_hash(content)
        now          = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO crawl_history
                    (url, content_hash, last_crawled, last_checked,
                     content_length, total_pages, status, metadata, page_type)
                VALUES (?, ?, ?, ?, ?, ?, 'success', ?, 'main')
            """, (url, content_hash, now, now, len(content),
                  total_pages, json.dumps(metadata or {})))

        print(f"    💾 History güncellendi: {content_hash[:16]}... ({total_pages} sayfa)")

    def mark_failed(self, url: str, error: str):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO crawl_history
                    (url, last_checked, status, error)
                VALUES (?, ?, 'failed', ?)
            """, (url, datetime.now().isoformat(), error))

    # ------------------------------------------------------------------ #
    # İstatistik                                                          #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        with self._connect() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM crawl_history").fetchone()[0]
            main_ok  = conn.execute(
                "SELECT COUNT(*) FROM crawl_history WHERE page_type='main' AND content_hash IS NOT NULL"
            ).fetchone()[0]
            sub_ok   = conn.execute(
                "SELECT COUNT(*) FROM crawl_history WHERE page_type='sub' AND content_hash IS NOT NULL"
            ).fetchone()[0]
            failed   = conn.execute(
                "SELECT COUNT(*) FROM crawl_history WHERE status='failed'"
            ).fetchone()[0]

        return {
            "total_urls":    total,
            "main_pages":    main_ok,
            "sub_pages":     sub_ok,
            "failed":        failed,
            "success_rate":  f"{(main_ok + sub_ok) / total * 100:.1f}%" if total > 0 else "0%",
        }

    def get_cached_result(self, url: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM crawl_history WHERE url = ?", (url,)
            ).fetchone()
        return dict(row) if row else None

    @property
    def history(self) -> Dict:
        """Geriye dönük uyumluluk."""
        with self._connect() as conn:
            rows = conn.execute("SELECT url FROM crawl_history WHERE page_type='main'").fetchall()
        return {row["url"]: {} for row in rows}
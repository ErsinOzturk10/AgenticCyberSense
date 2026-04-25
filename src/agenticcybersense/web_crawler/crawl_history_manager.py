"""Crawl history manager with SQLite storage and hash-based normalization."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CrawlHistoryManager:
    """Manage crawl history records and content hashes."""

    def __init__(self, db_file: str = "crawl_history.db"):
        self.db_file = db_file
        self._init_db()
        self._migrate_json_if_exists()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
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
                    page_type      TEXT DEFAULT 'main'
                )
                """,
            )
            try:
                conn.execute("ALTER TABLE crawl_history ADD COLUMN page_type TEXT DEFAULT 'main'")
            except Exception as exc:
                logger.debug("Could not add page_type column (may already exist): %s", exc)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON crawl_history(url)")
            conn.execute("PRAGMA journal_mode=WAL")

    def _migrate_json_if_exists(self) -> None:
        json_file = Path(self.db_file).with_suffix(".json")
        fallback = Path("crawl_history.json")

        for src in [json_file, fallback]:
            if not src.exists():
                continue

            try:
                with open(src, encoding="utf-8") as f:
                    old_data: dict = json.load(f)

                migrated = 0
                with self._connect() as conn:
                    for url, value in old_data.items():
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO crawl_history
                                (url, content_hash, last_crawled, last_checked,
                                 content_length, total_pages, status, metadata, page_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'main')
                            """,
                            (
                                url,
                                value.get("content_hash"),
                                value.get("last_crawled"),
                                value.get("last_checked"),
                                value.get("content_length"),
                                value.get("total_pages"),
                                value.get("status", "success"),
                                json.dumps(value.get("metadata") or {}),
                            ),
                        )
                        migrated += 1

                src.rename(src.with_suffix(".json.bak"))
                logger.info("Migration completed: %s records migrated from JSON (%s)", migrated, src)
            except Exception as exc:
                logger.warning("Migration failed for %s: %s", src, exc)

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
        for pattern in self._PATTERNS:
            text = pattern.sub("", text)
        return " ".join(text.lower().split())

    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(self.normalize_content(content).encode("utf-8")).hexdigest()

    def should_deep_crawl(self, url: str, current_content: str) -> tuple[bool, str]:
        current_hash = self.compute_hash(current_content)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM crawl_history WHERE url = ?",
                (url,),
            ).fetchone()

        if row is None:
            logger.info("New site detected; running full deep crawl: %s", url)
            return True, "new_site"

        stored_hash = row["content_hash"]
        if not stored_hash:
            logger.info("No stored hash found; running full deep crawl: %s", url)
            return True, "no_hash"

        if current_hash != stored_hash:
            logger.info("Main page changed; running deep crawl: %s", url)
            return True, "content_changed"

        logger.info("Main page unchanged; skipping deep crawl: %s", url)
        self._touch_last_checked(url)
        return False, "unchanged"

    def should_crawl_subpage(self, url: str, current_content: str) -> tuple[bool, str]:
        current_hash = self.compute_hash(current_content)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM crawl_history WHERE url = ?",
                (url,),
            ).fetchone()

        if row is None:
            return True, "new_subpage"

        stored_hash = row["content_hash"]
        if not stored_hash:
            return True, "no_hash"

        if current_hash != stored_hash:
            return True, "subpage_changed"

        self._touch_last_checked(url)
        return False, "subpage_unchanged"

    def update_subpage(self, url: str, content: str) -> None:
        content_hash = self.compute_hash(content)
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO crawl_history
                    (url, content_hash, last_crawled, last_checked,
                     content_length, status, page_type)
                VALUES (?, ?, ?, ?, ?, 'success', 'sub')
                """,
                (url, content_hash, now, now, len(content)),
            )

    def _touch_last_checked(self, url: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE crawl_history SET last_checked = ? WHERE url = ?",
                (datetime.now().isoformat(), url),
            )

    def update_history(
        self,
        url: str,
        content: str,
        total_pages: int,
        metadata: dict | None = None,
    ) -> None:
        content_hash = self.compute_hash(content)
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO crawl_history
                    (url, content_hash, last_crawled, last_checked,
                     content_length, total_pages, status, metadata, page_type)
                VALUES (?, ?, ?, ?, ?, ?, 'success', ?, 'main')
                """,
                (
                    url,
                    content_hash,
                    now,
                    now,
                    len(content),
                    total_pages,
                    json.dumps(metadata or {}),
                ),
            )

        logger.info("History updated: %s (%s pages)", content_hash[:16], total_pages)

    def mark_failed(self, url: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO crawl_history
                    (url, last_checked, status, error)
                VALUES (?, ?, 'failed', ?)
                """,
                (url, datetime.now().isoformat(), error),
            )

    def get_stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM crawl_history").fetchone()[0]
            main_ok = conn.execute("SELECT COUNT(*) FROM crawl_history WHERE page_type='main' AND content_hash IS NOT NULL").fetchone()[0]
            sub_ok = conn.execute("SELECT COUNT(*) FROM crawl_history WHERE page_type='sub' AND content_hash IS NOT NULL").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM crawl_history WHERE status='failed'").fetchone()[0]

        return {
            "total_urls": total,
            "main_pages": main_ok,
            "sub_pages": sub_ok,
            "failed": failed,
            "success_rate": f"{(main_ok + sub_ok) / total * 100:.1f}%" if total > 0 else "0%",
        }

    def get_cached_result(self, url: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM crawl_history WHERE url = ?",
                (url,),
            ).fetchone()
        return dict(row) if row else None

    @property
    def history(self) -> dict:
        """Backward-compatible access to stored main pages."""
        with self._connect() as conn:
            rows = conn.execute("SELECT url FROM crawl_history WHERE page_type='main'").fetchall()
        return {row["url"]: {} for row in rows}

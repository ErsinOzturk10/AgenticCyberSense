"""Tests for crawl history SQLite initialization behavior."""

from pathlib import Path

from agenticcybersense.web_crawler.crawl_history_manager import CrawlHistoryManager


def test_history_manager_creates_missing_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "missing" / "nested" / "crawl_history.db"

    CrawlHistoryManager(str(db_path))

    assert db_path.exists()
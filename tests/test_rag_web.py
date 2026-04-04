"""Tests for the web-crawl RAG ingest and hybrid search modules.

All tests use mocks so they run without a live Ollama instance or a real
ChromaDB on disk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from agenticcybersense.rag.ingest import WebContentIngester
    from agenticcybersense.rag.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COUNT_TWO: int = 2

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_vectordb() -> MagicMock:
    """Return a mock Chroma instance with the methods used by ingest/retriever."""
    db = MagicMock()
    db.add_documents.return_value = None
    db.delete.return_value = None
    db.similarity_search_with_relevance_scores.return_value = []
    db.get.return_value = {"documents": [], "metadatas": []}
    return db


# ---------------------------------------------------------------------------
# WebContentIngester tests
# ---------------------------------------------------------------------------


class TestWebContentIngester:
    """Unit tests for WebContentIngester."""

    @pytest.fixture
    def ingester_and_db(self, tmp_path: Path) -> tuple[WebContentIngester, MagicMock]:
        """Return a (WebContentIngester, mock_db) pair using mocked dependencies."""
        mock_db = _make_mock_vectordb()
        with (
            patch("agenticcybersense.rag.ingest._build_embeddings", return_value=MagicMock()),
            patch("agenticcybersense.rag.ingest._build_chroma", return_value=mock_db),
            patch.object(
                __import__("agenticcybersense.rag.ingest", fromlist=["settings"]).settings,
                "web_chroma_persist_dir",
                tmp_path,
            ),
        ):
            from agenticcybersense.rag.ingest import WebContentIngester  # noqa: PLC0415

            inst = WebContentIngester()

        # inst._vectordb is mock_db; manifest_path is under tmp_path
        return inst, mock_db  # type: ignore[return-value]

    def test_ingest_text_new_url(self, ingester_and_db: tuple[WebContentIngester, MagicMock]) -> None:
        """New URL should be added and chunks should be created."""
        ingester, mock_db = ingester_and_db
        result = ingester.ingest_text(
            url="https://example.com/page",
            content="This is some threat intelligence content about ransomware.",
        )
        assert result["status"] == "added"
        assert result["chunks"] >= 1
        mock_db.add_documents.assert_called_once()

    def test_ingest_text_unchanged_skipped(self, ingester_and_db: tuple[WebContentIngester, MagicMock]) -> None:
        """Re-ingesting the same content should be skipped without DB writes."""
        ingester, mock_db = ingester_and_db
        content = "Static content that never changes."
        # First call: add
        result1 = ingester.ingest_text(url="https://example.com/static", content=content)
        assert result1["status"] == "added"
        # Second call with identical content: skip
        result2 = ingester.ingest_text(url="https://example.com/static", content=content)
        assert result2["status"] == "skipped"
        assert result2["chunks"] == 0
        # add_documents called only once (first ingest), not on skip
        assert mock_db.add_documents.call_count == 1

    def test_ingest_text_updated_url(self, ingester_and_db: tuple[WebContentIngester, MagicMock]) -> None:
        """Changed content for an existing URL should trigger delete + re-add."""
        ingester, mock_db = ingester_and_db
        old_content = "Old ransomware content."
        new_content = "New APT content that is completely different."
        # First call: add
        ingester.ingest_text(url="https://example.com/news", content=old_content)
        # Second call with different content: update
        result = ingester.ingest_text(url="https://example.com/news", content=new_content)
        assert result["status"] == "updated"
        assert result["chunks"] >= 1
        mock_db.delete.assert_called_once()
        assert mock_db.add_documents.call_count == EXPECTED_COUNT_TWO  # once for add, once for update

    def test_ingest_json_file(self, ingester_and_db: tuple[WebContentIngester, MagicMock], tmp_path: Path) -> None:
        """Ingesting a valid JSON crawl file should add all documents."""
        ingester, _ = ingester_and_db
        crawl_data = [
            {"url": "https://example.com/a", "content": "Threat intel page A with malware info."},
            {"url": "https://example.com/b", "content": "Threat intel page B with CVE details."},
        ]
        json_path = tmp_path / "crawl_result.json"
        json_path.write_text(json.dumps(crawl_data), encoding="utf-8")

        summary = ingester.ingest_json_file(json_path)
        assert summary["added"] == EXPECTED_COUNT_TWO
        assert summary["chunks"] >= EXPECTED_COUNT_TWO

    def test_ingest_json_file_missing_fields(self, ingester_and_db: tuple[WebContentIngester, MagicMock], tmp_path: Path) -> None:
        """Items without url/content should be counted as errors."""
        ingester, _ = ingester_and_db
        bad_data = [{"title": "no url or content here"}]
        json_path = tmp_path / "bad.json"
        json_path.write_text(json.dumps(bad_data), encoding="utf-8")

        summary = ingester.ingest_json_file(json_path)
        assert summary["errors"] == 1
        assert summary["added"] == 0

    def test_ingest_directory(self, ingester_and_db: tuple[WebContentIngester, MagicMock], tmp_path: Path) -> None:
        """Ingesting a directory should process all JSON files found."""
        ingester, _ = ingester_and_db
        (tmp_path / "site_a.json").write_text(
            json.dumps({"url": "https://a.com", "content": "Content for site A."}),
            encoding="utf-8",
        )
        (tmp_path / "site_b.json").write_text(
            json.dumps({"url": "https://b.com", "content": "Content for site B."}),
            encoding="utf-8",
        )

        totals = ingester.ingest_directory(tmp_path)
        assert totals["files_processed"] == EXPECTED_COUNT_TWO
        assert totals["added"] == EXPECTED_COUNT_TWO


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Unit tests for HybridRetriever."""

    @pytest.fixture
    def retriever_and_db(self) -> tuple[HybridRetriever, MagicMock]:
        """Return a (HybridRetriever, mock_db) pair using mocked dependencies."""
        mock_db = _make_mock_vectordb()
        with (
            patch("agenticcybersense.rag.retriever._build_embeddings", return_value=MagicMock()),
            patch("agenticcybersense.rag.retriever._build_chroma", return_value=mock_db),
        ):
            from agenticcybersense.rag.retriever import HybridRetriever  # noqa: PLC0415

            inst = HybridRetriever()

        return inst, mock_db  # type: ignore[return-value]

    def test_search_returns_list(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """search() should always return a list."""
        retriever, _ = retriever_and_db
        results = retriever.search("ransomware")
        assert isinstance(results, list)

    def test_vector_search_returns_formatted_results(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """Vector search results should be properly formatted dicts."""
        from langchain_core.documents import Document  # noqa: PLC0415

        retriever, mock_db = retriever_and_db
        doc = Document(page_content="Ransomware hit critical infrastructure.", metadata={"source_url": "https://cisa.gov"})
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.91)]

        results = retriever.search("ransomware")
        assert len(results) >= 1
        assert results[0]["content"] == "Ransomware hit critical infrastructure."
        assert results[0]["source_url"] == "https://cisa.gov"
        assert results[0]["score"] == pytest.approx(0.91)
        assert results[0]["search_type"] == "semantic"

    def test_keyword_search_finds_match(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """Keyword search should find documents containing all query tokens."""
        retriever, mock_db = retriever_and_db
        mock_db.get.return_value = {
            "documents": [
                "APT29 Cozy Bear campaign details.",
                "Unrelated weather forecast.",
            ],
            "metadatas": [
                {"source_url": "https://mitre.org/apt29"},
                {"source_url": "https://weather.com"},
            ],
        }

        results = retriever.search("APT29")
        assert len(results) >= 1
        assert any("APT29" in r["content"] for r in results)

    def test_keyword_search_requires_all_tokens(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """Keyword search should require ALL query tokens to be present (AND logic)."""
        retriever, mock_db = retriever_and_db
        mock_db.get.return_value = {
            "documents": [
                "APT29 uses phishing.",
                "APT28 uses spear phishing.",
            ],
            "metadatas": [{}, {}],
        }

        results = retriever.search("APT29 phishing")
        assert len(results) == 1
        assert "APT29" in results[0]["content"]

    def test_merge_deduplicates_identical_content(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """Duplicate content from semantic and keyword results should be de-duplicated."""
        from langchain_core.documents import Document  # noqa: PLC0415

        retriever, mock_db = retriever_and_db

        dup_text = "Duplicate content appearing in both semantic and keyword results."
        unique_text = "Unique keyword-only content."

        doc = Document(page_content=dup_text, metadata={"source_url": "https://x.com"})
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        mock_db.get.return_value = {
            "documents": [dup_text, unique_text],
            "metadatas": [{"source_url": "https://x.com"}, {"source_url": "https://y.com"}],
        }

        results = retriever.search("duplicate content", k=10)
        contents = [r["content"] for r in results]
        # dup_text should appear exactly once
        assert contents.count(dup_text) == 1
        assert unique_text in contents

    def test_get_relevant_documents_returns_langchain_docs(self, retriever_and_db: tuple[HybridRetriever, MagicMock]) -> None:
        """get_relevant_documents should return LangChain Document objects."""
        from langchain_core.documents import Document  # noqa: PLC0415

        retriever, mock_db = retriever_and_db
        doc = Document(page_content="Critical CVE details.", metadata={"source_url": "https://nvd.nist.gov"})
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.88)]

        docs = retriever.get_relevant_documents("CVE vulnerability")
        assert len(docs) >= 1
        assert docs[0].page_content == "Critical CVE details."


# ---------------------------------------------------------------------------
# hybrid_search convenience function
# ---------------------------------------------------------------------------


class TestHybridSearchFunction:
    """Unit tests for the hybrid_search convenience function."""

    def test_returns_no_results_message_when_empty(self) -> None:
        """Should return informative message when no results are found."""
        mock_db = _make_mock_vectordb()
        with (
            patch("agenticcybersense.rag.retriever._build_embeddings", return_value=MagicMock()),
            patch("agenticcybersense.rag.retriever._build_chroma", return_value=mock_db),
        ):
            from agenticcybersense.rag.retriever import hybrid_search  # noqa: PLC0415

            output = hybrid_search("unknown query", k=3)
            assert "No relevant information found" in output

    def test_returns_formatted_markdown_with_results(self) -> None:
        """Should return Markdown-formatted string when results are found."""
        from langchain_core.documents import Document  # noqa: PLC0415

        mock_db = _make_mock_vectordb()
        doc = Document(
            page_content="Threat actor uses ransomware.",
            metadata={"source_url": "https://example.com/threat"},
        )
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.95)]

        with (
            patch("agenticcybersense.rag.retriever._build_embeddings", return_value=MagicMock()),
            patch("agenticcybersense.rag.retriever._build_chroma", return_value=mock_db),
        ):
            from agenticcybersense.rag.retriever import hybrid_search  # noqa: PLC0415

            output = hybrid_search("ransomware", k=3)
            assert "Source:" in output
            assert "https://example.com/threat" in output

"""Module for ingesting web-crawl documents into the RAG system.

Supports incremental ingestion: only changed content (detected via SHA-256 hash)
is re-embedded and updated in ChromaDB.  Uses ``nomic-embed-text`` via Ollama for
embeddings so that no external API keys are required.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agenticcybersense.settings import settings

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: Final[str] = "web_ingested_manifest.json"
HASH_READ_CHUNK_SIZE: Final[int] = 65_536


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.nomic_embed_model,
    )


def _build_chroma(embeddings: OllamaEmbeddings) -> Chroma:
    persist_dir = str(settings.web_chroma_persist_dir)
    return Chroma(
        collection_name=settings.web_collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def _split_text(text: str, source_url: str, extra_metadata: dict[str, Any] | None = None) -> list[Document]:
    """Split *text* into overlapping chunks and attach metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.web_chunk_size,
        chunk_overlap=settings.web_chunk_overlap,
    )
    raw_chunks = splitter.split_text(text)
    metadata_base: dict[str, Any] = {"source_url": source_url}
    if extra_metadata:
        metadata_base.update(extra_metadata)

    return [Document(page_content=chunk, metadata=dict(metadata_base)) for chunk in raw_chunks]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Web manifest unreadable; resetting. Error: %s", exc)
    return {"documents": {}}


def _save_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_by_url(vectordb: Chroma, source_url: str) -> None:
    """Remove all chunks previously stored for *source_url*."""
    try:
        delete_fn = getattr(vectordb, "delete", None)
        if callable(delete_fn):
            delete_fn(where={"source_url": source_url})
        else:
            logger.warning("Chroma delete API not available; cannot remove old chunks for %s", source_url)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("Delete by URL failed (%s): %s", source_url, exc)


def _make_chunk_ids(sha256: str, chunks: list[Document]) -> list[str]:
    return [f"{sha256}:{i}" for i in range(len(chunks))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WebContentIngester:
    """Ingest web-crawl content into ChromaDB with incremental (hash-based) updates.

    Example::

        ingester = WebContentIngester()
        result = ingester.ingest_text(
            url="https://example.com/blog/post",
            content="Article body ...",
            metadata={"title": "My Post", "crawled_at": "2025-01-01T00:00:00Z"},
        )
        print(result)  # {"status": "added", "chunks": 3}
    """

    def __init__(self) -> None:
        """Initialise embeddings, vector store, and incremental manifest."""
        self._embeddings = _build_embeddings()
        self._vectordb = _build_chroma(self._embeddings)
        self._manifest_path = settings.web_chroma_persist_dir / MANIFEST_FILENAME
        self._manifest = _load_manifest(self._manifest_path)

    # ------------------------------------------------------------------
    # Single-document ingestion
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        url: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest or incrementally update a single web page.

        Args:
            url: Canonical URL used as the document identifier.
            content: Full text content of the page.
            metadata: Optional extra metadata stored alongside each chunk.

        Returns:
            A dict with keys ``status`` (``"added"``, ``"updated"``, ``"skipped"``)
            and ``chunks`` (number of chunks written).

        """
        content_sha = _sha256_text(content)
        prev = self._manifest["documents"].get(url)

        if prev and prev.get("sha256") == content_sha:
            logger.debug("Skipping unchanged URL: %s", url)
            return {"status": "skipped", "chunks": 0}

        if prev:
            logger.info("Updating changed URL: %s", url)
            _delete_by_url(self._vectordb, url)
            status = "updated"
        else:
            logger.info("Adding new URL: %s", url)
            status = "added"

        extra_meta: dict[str, Any] = {"crawled_at": _utc_now_iso()}
        if metadata:
            extra_meta.update(metadata)

        chunks = _split_text(content, source_url=url, extra_metadata=extra_meta)
        if not chunks:
            logger.warning("No chunks produced for URL: %s", url)
            return {"status": status, "chunks": 0}

        ids = _make_chunk_ids(content_sha, chunks)
        self._vectordb.add_documents(documents=chunks, ids=ids)

        self._manifest["documents"][url] = {
            "sha256": content_sha,
            "chunks": len(chunks),
            "ingested_at_utc": _utc_now_iso(),
        }
        _save_manifest(self._manifest, self._manifest_path)
        logger.info("Ingested %d chunks for URL: %s (status=%s)", len(chunks), url, status)
        return {"status": status, "chunks": len(chunks)}

    # ------------------------------------------------------------------
    # Batch ingestion from JSON crawl output files
    # ------------------------------------------------------------------

    def ingest_json_file(self, json_path: Path) -> dict[str, Any]:
        """Ingest a single JSON file produced by the web crawler.

        The JSON file should contain either:
        * A single object with ``url`` and ``content`` (or ``text``) keys, **or**
        * A list of such objects.

        Returns:
            Summary dict with counts: ``added``, ``updated``, ``skipped``, ``errors``, ``chunks``.

        """
        summary: dict[str, Any] = {"added": 0, "updated": 0, "skipped": 0, "errors": 0, "chunks": 0}
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("Cannot read JSON file %s", json_path)
            summary["errors"] += 1
            return summary

        items: list[dict[str, Any]] = raw if isinstance(raw, list) else [raw]
        for item in items:
            url = item.get("url") or item.get("link") or ""
            content = item.get("content") or item.get("text") or item.get("body") or ""
            if not url or not content:
                logger.debug("Skipping item without url/content in %s", json_path)
                summary["errors"] += 1
                continue

            extra_meta: dict[str, Any] = {
                k: v for k, v in item.items() if k not in {"url", "link", "content", "text", "body"} and isinstance(v, str | int | float | bool)
            }
            result = self.ingest_text(url=url, content=content, metadata=extra_meta)
            summary[result["status"]] = summary.get(result["status"], 0) + 1
            summary["chunks"] += result["chunks"]

        return summary

    def ingest_directory(self, directory: Path) -> dict[str, Any]:
        """Recursively ingest all ``*.json`` files under *directory*.

        Returns:
            Aggregated summary with counts across all files.

        """
        totals: dict[str, Any] = {"files_processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0, "chunks": 0}
        json_files = list(directory.rglob("*.json"))
        logger.info("Found %d JSON files in %s", len(json_files), directory)
        for json_file in json_files:
            file_summary = self.ingest_json_file(json_file)
            totals["files_processed"] += 1
            for key in ("added", "updated", "skipped", "errors", "chunks"):
                totals[key] = totals.get(key, 0) + file_summary.get(key, 0)
        return totals

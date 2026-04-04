"""Hybrid retriever combining vector similarity search and keyword search.

Searches the web-crawl ChromaDB collection built by :mod:`agenticcybersense.rag.ingest`.
Results from both strategies are merged and de-duplicated so that the most
relevant documents appear first.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Final

from agenticcybersense.rag.ingest import _build_chroma, _build_embeddings

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

DEFAULT_K: Final[int] = 5
MAX_KEYWORD_RESULTS: Final[int] = 200  # upper limit when fetching all docs for keyword scan


class HybridRetriever:
    """Perform hybrid (semantic + keyword) search over the web-crawl vector store.

    Semantic search uses ChromaDB's cosine-similarity index; keyword search
    scans stored chunk text for exact (case-insensitive) substring matches.
    Results are merged and de-duplicated by chunk content.

    Example::

        retriever = HybridRetriever()
        results = retriever.search("ransomware APT29", k=5)
        for doc in results:
            print(doc["source_url"], doc["content"][:120])
    """

    def __init__(self) -> None:
        """Initialise the retriever by connecting to the web-crawl vector store."""
        embeddings = _build_embeddings()
        self._vectordb: Chroma = _build_chroma(embeddings)

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        """Run hybrid search and return merged semantic + keyword results.

        Args:
            query: Natural-language or keyword query string.
            k: Maximum number of results to return.

        Returns:
            List of result dicts ordered by relevance, each containing:
            ``content``, ``source_url``, ``score``, and any other metadata.

        """
        semantic_results = self._vector_search(query, k=k)
        keyword_results = self._keyword_search(query, k=k)
        merged = self._merge_results(semantic_results, keyword_results, k=k)
        logger.info(
            "Hybrid search for %r: %d semantic + %d keyword -> %d merged",
            query,
            len(semantic_results),
            len(keyword_results),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # LangChain Retriever compatibility shim
    # ------------------------------------------------------------------

    def get_relevant_documents(self, query: str, *, k: int = DEFAULT_K) -> list[Any]:
        """Return LangChain :class:`Document` objects for *query*.

        This allows :class:`HybridRetriever` to be used anywhere a
        LangChain ``BaseRetriever`` is expected (e.g. inside a chain).
        """
        from langchain_core.documents import Document  # noqa: PLC0415

        results = self.search(query, k=k)
        return [
            Document(
                page_content=r["content"],
                metadata={k_: v for k_, v in r.items() if k_ != "content"},
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Run semantic similarity search via ChromaDB."""
        try:
            results = self._vectordb.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            logger.exception("Vector search failed for query: %r", query)
            return []

        output: list[dict[str, Any]] = []
        for doc, score in results:
            entry: dict[str, Any] = {
                "content": doc.page_content,
                "score": round(float(score), 4),
                "search_type": "semantic",
            }
            entry.update(doc.metadata)
            output.append(entry)
        return output

    def _keyword_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Run case-insensitive keyword scan over all stored chunks.

        Splits *query* into individual tokens and requires **all** tokens to
        appear in a chunk (AND semantics).  This mimics basic full-text search
        without requiring an additional dependency.
        """
        tokens = [t for t in re.split(r"\s+", query.strip()) if t]
        if not tokens:
            return []

        try:
            collection = self._vectordb.get(limit=MAX_KEYWORD_RESULTS, include=["documents", "metadatas"])
        except Exception:
            logger.exception("Keyword search failed: could not fetch documents")
            return []

        documents_list: list[str] = collection.get("documents") or []
        metadatas_list: list[dict[str, Any]] = collection.get("metadatas") or []

        # Pad metadatas to match documents length
        if len(metadatas_list) < len(documents_list):
            metadatas_list = metadatas_list + [{}] * (len(documents_list) - len(metadatas_list))

        output: list[dict[str, Any]] = []
        for text, meta in zip(documents_list, metadatas_list, strict=False):
            if not text:
                continue
            lower_text = text.lower()
            if all(tok.lower() in lower_text for tok in tokens):
                entry: dict[str, Any] = {
                    "content": text,
                    "score": 1.0,
                    "search_type": "keyword",
                }
                entry.update(meta or {})
                output.append(entry)
                if len(output) >= k:
                    break

        return output

    @staticmethod
    def _merge_results(
        semantic: list[dict[str, Any]],
        keyword: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Merge semantic and keyword results, de-duplicate by content, keep top-*k*."""
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []

        for result in semantic + keyword:
            fingerprint = result.get("content", "")[:200]
            if fingerprint not in seen:
                seen.add(fingerprint)
                merged.append(result)

        # Semantic results are already ranked; keyword-only additions come last
        return merged[:k]


def hybrid_search(query: str, k: int = DEFAULT_K) -> str:
    """Run a hybrid search and return a formatted Markdown string.

    Suitable for direct use as an MCP tool or CLI utility.

    Args:
        query: Search query.
        k: Number of results to return.

    Returns:
        Markdown-formatted string of results, or an informative message.

    """
    retriever = HybridRetriever()
    results = retriever.search(query, k=k)
    if not results:
        return "No relevant information found."

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        source = r.get("source_url", "unknown")
        score = r.get("score", "n/a")
        search_type = r.get("search_type", "unknown")
        content = (r.get("content") or "").strip()
        parts.append(f"**[{i}] Source:** {source} | **Score:** {score} | **Type:** {search_type}\n\n{content}")

    return "\n\n---\n\n".join(parts)

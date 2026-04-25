"""Webcrawler JSON → ChromaDB ingest pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

try:
    from agenticcybersense.settings import settings

    CHROMA_DIR = str(settings.chroma_persist_dir)
    EMBED_MODEL_NAME = getattr(settings, "embedding_model", "all-mpnet-base-v2")
except Exception as exc:
    logger.warning("Settings import failed, using defaults: %s", exc)
    CHROMA_DIR = "./data/chroma_db"
    EMBED_MODEL_NAME = "all-mpnet-base-v2"

_WEBCRAWLER_DIR = Path(__file__).parent
DEFAULT_JSON_PATH = str(_WEBCRAWLER_DIR / "output" / "latest_results.json")

COLLECTION_NAME = "webcrawler_pages"
CHUNK_SIZE = 1000

_embed_model: SentenceTransformer | None = None
_collection = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model  # noqa: PLW0603
    if _embed_model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_collection():
    global _collection  # noqa: PLW0603
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s (docs: %d)", COLLECTION_NAME, CHROMA_DIR, _collection.count())
    return _collection


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    return [model.encode(t).tolist() for t in texts]


def upsert_site(url: str, page_idx: int, title: str, content: str, metadata: dict) -> int:
    if not content or len(content.strip()) < 50:
        return 0
    chunks = chunk_text(content)
    embeddings = embed_texts(chunks)
    collection = _get_collection()
    doc_metadatas = [
        {
            "url": url,
            "page_idx": page_idx,
            "title": title or "",
            "chunk_idx": i,
            **{k: str(v) for k, v in metadata.items() if v is not None},
        }
        for i in range(len(chunks))
    ]
    ids = [f"{url}::{page_idx}::{i}" for i in range(len(chunks))]
    collection.upsert(documents=chunks, metadatas=doc_metadatas, ids=ids, embeddings=embeddings)
    return len(chunks)


def ingest_crawler_json(json_path: str | None = None) -> dict[str, int]:
    """Crawler output JSON'ını okuyup ChromaDB'ye yükler.
    json_path verilmezse web_crawler/output/latest_results.json kullanılır.
    """
    path = Path(json_path) if json_path else Path(DEFAULT_JSON_PATH)
    if not path.exists():
        logger.error("JSON file not found: %s", path)
        return {"sites": 0, "pages": 0, "chunks": 0}

    logger.info("Reading crawler JSON: %s (%.1f MB)", path, path.stat().st_size / 1_000_000)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    total_sites = total_pages = total_chunks = 0
    for site_url, site_data in data.items():
        pages = site_data.get("pages", [])
        if not pages:
            continue
        site_chunks = 0
        for page_idx, page in enumerate(pages):
            content = page.get("main_content") or ""
            title = page.get("title") or ""
            page_url = page.get("url") or site_url
            page_meta = page.get("metadata") or {}
            added = upsert_site(
                url=page_url,
                page_idx=page_idx,
                title=title,
                content=content,
                metadata={"site_url": site_url, "crawl_mode": site_data.get("crawl_mode", ""), "last_updated": site_data.get("last_updated", ""), **page_meta},
            )
            site_chunks += added
            total_pages += 1
        if site_chunks > 0:
            total_sites += 1
            total_chunks += site_chunks
            logger.info("✅ %s → %d pages, %d chunks", site_url[:60], len(pages), site_chunks)

    logger.info("Ingest complete: %d sites, %d pages, %d chunks", total_sites, total_pages, total_chunks)
    return {"sites": total_sites, "pages": total_pages, "chunks": total_chunks}


def query_webcrawler_rag(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """ChromaDB'ye semantic sorgu atar."""
    collection = _get_collection()
    if collection.count() == 0:
        logger.warning("webcrawler_pages collection is empty — run ingest first")
        return []
    model = _get_embed_model()
    query_embedding = model.encode(query).tolist()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.exception("ChromaDB query failed: %s", exc)
        return []

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        output.append(
            {
                "content": doc,
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "site_url": meta.get("site_url", ""),
                "last_updated": meta.get("last_updated", ""),
                "score": round(1 - dist, 3),
            },
        )
    return output


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    stats = ingest_crawler_json(json_file)
    logger.info("\n✅ Done: %s", stats)

"""
RAG Indexer — Incremental ChromaDB + nomic-embed-text + Hybrid Search

- Crawl sonuçlarını chunk'lara böler
- Ollama nomic-embed-text ile embed eder
- ChromaDB'ye yazar (incremental — sadece değişenler güncellenir)
- Hybrid search: keyword (SQLite FTS5) + similarity (ChromaDB) birleşimi
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from trafilatura_ollama_agent import ExtractionResult


# ──────────────────────────────────────────────────────────────────── #
#  TextChunker                                                          #
# ──────────────────────────────────────────────────────────────────── #


class TextChunker:
    """Uzun metinleri örtüşen chunk'lara böler."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Metni kelime bazlı chunk'lara böl.

        Her chunk dict şu alanları içerir:
            text, chunk_index, total_chunks, url, title
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        words = text.split()

        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        raw_chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            raw_chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step

        total = len(raw_chunks)
        return [
            {
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": total,
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
            }
            for idx, chunk_text in enumerate(raw_chunks)
        ]


# ──────────────────────────────────────────────────────────────────── #
#  RAGIndexer                                                           #
# ──────────────────────────────────────────────────────────────────── #


class RAGIndexer:
    """
    Crawl sonuçlarını ChromaDB'ye ve SQLite FTS5'e yazar.

    Parametreler
    ------------
    chroma_db_path : ChromaDB kalıcı veri dizini
    ollama_base_url: Ollama API adresi (varsayılan http://localhost:11434)
    embed_model    : Embedding modeli adı (varsayılan nomic-embed-text)
    chunk_size     : Chunk başına kelime sayısı
    chunk_overlap  : Ardışık chunk'lar arasındaki kelime örtüşmesi
    """

    COLLECTION_NAME = "cti_crawl"
    FTS_DB_NAME = "rag_fts.db"

    def __init__(
        self,
        chroma_db_path: str = "chroma_db",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.chroma_db_path = chroma_db_path
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._fts_db_path = str(Path(chroma_db_path) / self.FTS_DB_NAME)

        # ChromaDB istemcisi (lazy — ilk kullanımda başlatılır)
        self._chroma_client: Any = None
        self._collection: Any = None

        # FTS veritabanı
        Path(chroma_db_path).mkdir(parents=True, exist_ok=True)
        self._init_fts()

    # ------------------------------------------------------------------ #
    # ChromaDB                                                             #
    # ------------------------------------------------------------------ #

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "chromadb paketi yüklü değil. 'uv add chromadb' komutuyla yükleyin."
            raise ImportError(msg) from exc

        self._chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # ------------------------------------------------------------------ #
    # SQLite FTS5                                                          #
    # ------------------------------------------------------------------ #

    def _fts_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._fts_db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_fts(self) -> None:
        with self._fts_connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    url         TEXT,
                    title       TEXT,
                    chunk_index INTEGER,
                    total_chunks INTEGER,
                    content_hash TEXT,
                    text        TEXT,
                    indexed_at  TEXT
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    url,
                    title,
                    text,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)
            # FTS tetikleyicileri — chunks tablosunu güncelde FTS'i de güncelle
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai
                AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, chunk_id, url, title, text)
                    VALUES (new.rowid, new.chunk_id, new.url, new.title, new.text);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad
                AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, url, title, text)
                    VALUES ('delete', old.rowid, old.chunk_id, old.url, old.title, old.text);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au
                AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, url, title, text)
                    VALUES ('delete', old.rowid, old.chunk_id, old.url, old.title, old.text);
                    INSERT INTO chunks_fts(rowid, chunk_id, url, title, text)
                    VALUES (new.rowid, new.chunk_id, new.url, new.title, new.text);
                END
            """)

    # ------------------------------------------------------------------ #
    # Embedding                                                            #
    # ------------------------------------------------------------------ #

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Ollama API üzerinden embedding üret."""
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embedding"])
        return embeddings

    # ------------------------------------------------------------------ #
    # Hash yardımcısı                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_id(url: str, chunk_index: int) -> str:
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
        return f"{url_hash}_{chunk_index}"

    # ------------------------------------------------------------------ #
    # URL chunk silme                                                      #
    # ------------------------------------------------------------------ #

    def delete_url(self, url: str) -> None:
        """URL'ye ait tüm chunk'ları ChromaDB'den ve FTS'ten sil."""
        # ChromaDB
        try:
            col = self._get_collection()
            existing = col.get(where={"url": url})
            if existing and existing.get("ids"):
                col.delete(ids=existing["ids"])
                print(f"    🗑️  ChromaDB: {len(existing['ids'])} chunk silindi ({url[:50]})")
        except Exception as e:
            print(f"    ⚠️  ChromaDB silme hatası: {e}")

        # FTS / chunks tablosu
        with self._fts_connect() as conn:
            conn.execute("DELETE FROM chunks WHERE url = ?", (url,))

    # ------------------------------------------------------------------ #
    # Incremental güncelleme                                               #
    # ------------------------------------------------------------------ #

    def update_if_changed(self, url: str, content: str, content_hash: str) -> bool:
        """
        content_hash ile mevcut hash karşılaştır.
        Değiştiyse (veya yeniyse) True döndür.
        Değişmemişse False döndür.

        Not: Hash hesaplaması dışarıda (crawl_history_manager) yapılır;
             aynı hash iki kez hesaplanmaz.
        """
        with self._fts_connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM chunks WHERE url = ? LIMIT 1", (url,)
            ).fetchone()

        if row is None:
            return True  # Yeni URL

        return row["content_hash"] != content_hash

    # ------------------------------------------------------------------ #
    # Ana index fonksiyonu                                                 #
    # ------------------------------------------------------------------ #

    def index_crawl_results(
        self,
        results: list[ExtractionResult],
        url: str,
    ) -> int:
        """
        Crawl sonuçlarını chunk'lara böl, embed et ve DB'ye yaz.

        Returns
        -------
        int : Eklenen toplam chunk sayısı
        """
        # İçerikleri birleştir
        texts_with_meta: list[dict[str, Any]] = []
        for result in results:
            if not result.main_content or not result.main_content.strip():
                continue
            texts_with_meta.append(
                {
                    "text": result.main_content,
                    "url": result.url,
                    "title": result.title or "",
                }
            )

        if not texts_with_meta:
            print(f"    ℹ️  RAG: İçerik yok, atlanıyor ({url[:50]})")
            return 0

        # Chunk'la
        all_chunks: list[dict[str, Any]] = []
        for item in texts_with_meta:
            chunks = self.chunker.chunk(item["text"], metadata={"url": item["url"], "title": item["title"]})
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Mevcut URL verilerini temizle (incremental — eski chunk'lar silinir)
        self.delete_url(url)

        # Hash (tüm chunk içeriklerinden)
        combined_hash = self._content_hash("\n".join(c["text"] for c in all_chunks))

        # Embed et
        try:
            texts_list = [c["text"] for c in all_chunks]
            embeddings = self._embed_texts(texts_list)
        except Exception as e:
            print(f"    ❌ RAG embedding hatası: {e}")
            return 0

        # ChromaDB'ye yaz
        col = self._get_collection()
        ids = [self._chunk_id(url, c["chunk_index"]) for c in all_chunks]
        metadatas = [
            {
                "url": c["url"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
            }
            for c in all_chunks
        ]
        col.add(ids=ids, embeddings=embeddings, documents=texts_list, metadatas=metadatas)

        # FTS / chunks tablosuna yaz
        now = datetime.now().isoformat()
        with self._fts_connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, url, title, chunk_index, total_chunks, content_hash, text, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ids[i],
                        all_chunks[i]["url"],
                        all_chunks[i]["title"],
                        all_chunks[i]["chunk_index"],
                        all_chunks[i]["total_chunks"],
                        combined_hash,
                        texts_list[i],
                        now,
                    )
                    for i in range(len(all_chunks))
                ],
            )

        print(f"    ✅ RAG: {len(all_chunks)} chunk indexlendi ({url[:50]})")
        return len(all_chunks)

    # ------------------------------------------------------------------ #
    # Arama fonksiyonları                                                  #
    # ------------------------------------------------------------------ #

    def similarity_search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Sadece ChromaDB vektör benzerliği ile arama."""
        try:
            query_embedding = self._embed_texts([query])[0]
        except Exception as e:
            print(f"    ❌ Embedding hatası: {e}")
            return []

        col = self._get_collection()
        try:
            result = col.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, col.count()),
            )
        except Exception as e:
            print(f"    ❌ ChromaDB sorgu hatası: {e}")
            return []

        hits: list[dict[str, Any]] = []
        if not result or not result.get("ids"):
            return hits

        for i, chunk_id in enumerate(result["ids"][0]):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i] if result.get("distances") else None,
                    "source": "similarity",
                }
            )
        return hits

    def keyword_search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Sadece SQLite FTS5 ile keyword arama."""
        # FTS5 özel karakterlerini temizle
        safe_query = re.sub(r'[^\w\s]', " ", query)
        safe_query = " ".join(safe_query.split())
        if not safe_query:
            return []

        with self._fts_connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.url, c.title, c.text, c.chunk_index, c.total_chunks,
                           rank
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, n_results),
                ).fetchall()
            except sqlite3.OperationalError as e:
                print(f"    ⚠️  FTS hatası: {e}")
                return []

        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": {
                    "url": row["url"],
                    "title": row["title"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                },
                "distance": None,
                "source": "keyword",
            }
            for row in rows
        ]

    def hybrid_search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Keyword + Similarity sonuçlarını birleştir.

        Her kaynaktan n_results kadar sonuç alır,
        chunk_id bazında tekrarlananları kaldırır,
        similarity önce gelecek şekilde sıralar.
        """
        sim_results = self.similarity_search(query, n_results=n_results)
        kw_results = self.keyword_search(query, n_results=n_results)

        seen: set[str] = set()
        combined: list[dict[str, Any]] = []

        for hit in sim_results:
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                combined.append(hit)

        for hit in kw_results:
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                hit["source"] = "keyword"
                combined.append(hit)

        return combined[:n_results]

    # ------------------------------------------------------------------ #
    # İstatistik                                                           #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """Kaç URL, kaç chunk indexlenmiş bilgisini döndür."""
        with self._fts_connect() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_urls = conn.execute("SELECT COUNT(DISTINCT url) FROM chunks").fetchone()[0]

        chroma_count = 0
        try:
            chroma_count = self._get_collection().count()
        except Exception:
            pass

        return {
            "total_urls": total_urls,
            "total_chunks": total_chunks,
            "chroma_vectors": chroma_count,
            "embed_model": self.embed_model,
            "chroma_db_path": self.chroma_db_path,
        }

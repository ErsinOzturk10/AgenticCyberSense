"""
RAG Search Demo — Crawl edilmiş içeriklerde semantik arama

Kullanım:
    python rag_search_demo.py "ransomware campaign 2024"
    python rag_search_demo.py "kritik altyapı saldırıları" --mode hybrid
    python rag_search_demo.py "CVE-2024-1234" --mode keyword
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# rag_indexer modülü aynı dizinde olduğundan sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from config import CHROMA_DB_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K
from rag_indexer import RAGIndexer


def print_results(hits: list[dict], query: str, mode: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"🔍 Sorgu   : {query}")
    print(f"   Mod     : {mode}")
    print(f"   Sonuç   : {len(hits)} chunk")
    print("=" * 70)

    if not hits:
        print("❌ Sonuç bulunamadı.")
        return

    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        source = hit.get("source", "?")
        distance = hit.get("distance")
        url = meta.get("url", "?")
        title = meta.get("title", "")
        chunk_idx = meta.get("chunk_index", 0)
        total_chunks = meta.get("total_chunks", "?")
        text = hit.get("text", "")

        dist_str = f"{1 - distance:.4f}" if distance is not None else "N/A"

        print(f"\n[{i}] Kaynak: {source.upper()}" + (f" | Benzerlik: {dist_str}" if distance is not None else ""))
        print(f"    URL    : {url}")
        if title:
            print(f"    Başlık : {title}")
        print(f"    Chunk  : {chunk_idx + 1}/{total_chunks}")
        print(f"    Metin  :")
        # İlk 400 karakter göster
        preview = text[:400].replace("\n", " ")
        if len(text) > 400:
            preview += " ..."
        print(f"      {preview}")

    print(f"\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl edilmiş CTI içeriklerinde semantik / keyword / hybrid arama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="Arama sorgusu")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "similarity", "keyword"],
        default="hybrid",
        help="Arama modu (varsayılan: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_TOP_K,
        dest="top_k",
        help=f"Kaç sonuç gösterilsin (varsayılan: {RAG_TOP_K})",
    )
    parser.add_argument(
        "--db",
        default=CHROMA_DB_PATH,
        help=f"ChromaDB dizini (varsayılan: {CHROMA_DB_PATH})",
    )
    args = parser.parse_args()

    print(f"🧠 RAG Search Demo")
    print(f"   DB      : {args.db}")
    print(f"   Model   : {EMBED_MODEL}")
    print(f"   Top-K   : {args.top_k}")

    indexer = RAGIndexer(
        chroma_db_path=args.db,
        embed_model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # İstatistikleri göster
    stats = indexer.get_stats()
    print(f"\n📊 DB İçeriği:")
    print(f"   İndexlenen URL   : {stats['total_urls']}")
    print(f"   Toplam chunk     : {stats['total_chunks']}")
    print(f"   ChromaDB vektör  : {stats['chroma_vectors']}")

    if stats["total_chunks"] == 0:
        print("\n⚠️  Veri tabanı boş! Önce crawler'ı çalıştırın:")
        print("   python main_trafilatura.py")
        sys.exit(0)

    # Arama yap
    mode = args.mode
    if mode == "hybrid":
        hits = indexer.hybrid_search(args.query, n_results=args.top_k)
    elif mode == "similarity":
        hits = indexer.similarity_search(args.query, n_results=args.top_k)
    else:
        hits = indexer.keyword_search(args.query, n_results=args.top_k)

    print_results(hits, args.query, mode)


if __name__ == "__main__":
    main()

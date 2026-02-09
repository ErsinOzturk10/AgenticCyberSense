# rag.py

"""
RAG utilities: build/load a persistent Chroma vector store from PDFs in /data
and run similarity search with source citations.

- PDFs: ./data/*.pdf
- Vector DB: ./chroma_db_minilm384 (persisted locally, gitignored)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"

# Versioned DB dir to avoid embedding-dimension mismatches
DB_PATH = BASE_DIR / "chroma_db_minilm384"

# Track what we've ingested so we can do incremental updates safely
MANIFEST_PATH = DB_PATH / "ingested_manifest.json"

_vectordb: Optional[Chroma] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"files": {}}
    return {"files": {}}


def _save_manifest(manifest: dict) -> None:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_pdfs() -> list[Path]:
    if not DATA_PATH.exists():
        return []
    return sorted([p for p in DATA_PATH.glob("*.pdf") if p.is_file()])


def _build_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)


def _make_chunk_ids(file_sha: str, docs) -> list[str]:
    ids = []
    for i, d in enumerate(docs):
        page = d.metadata.get("page", "na")
        ids.append(f"{file_sha}:{page}:{i}")
    return ids


def _delete_by_source(vectordb: Chroma, source_path: str) -> None:
    """
    Best-effort delete all chunks for a given PDF source.
    """
    try:
        vectordb._collection.delete(where={"source": source_path})
    except Exception:
        pass


def _ingest_pdf(vectordb: Chroma, pdf_path: Path, file_sha: str) -> int:
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    docs = _split_docs(documents)

    src = str(pdf_path.resolve())
    for d in docs:
        d.metadata["source"] = src
        d.metadata["file_sha256"] = file_sha

    ids = _make_chunk_ids(file_sha, docs)
    vectordb.add_documents(documents=docs, ids=ids)
    return len(docs)


def initialize_rag(rebuild: bool = False) -> dict:
    """
    Initialize RAG once at application startup.

    Returns a small status dict:
    {
      "mode": "rebuild" | "load" | "incremental",
      "pdf_total": int,
      "added_pdfs": int,
      "updated_pdfs": int,
      "skipped_pdfs": int,
      "chunks_added": int
    }

    - If rebuild=True: rebuild DB from all PDFs (fresh).
    - Else: load DB if exists, then incrementally ingest any new/changed PDFs from ./data.
    """
    global _vectordb

    embeddings = _build_embeddings()
    pdfs = _list_pdfs()
    if not pdfs:
        raise RuntimeError("No PDF documents found in ./data folder.")

    status = {
        "mode": "load",
        "pdf_total": len(pdfs),
        "added_pdfs": 0,
        "updated_pdfs": 0,
        "skipped_pdfs": 0,
        "chunks_added": 0,
    }

    # Rebuild means "start clean"
    if rebuild and DB_PATH.exists():
        for p in DB_PATH.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
            except Exception:
                pass
        status["mode"] = "rebuild"

    # Load existing DB if present (fast)
    if DB_PATH.exists() and any(DB_PATH.iterdir()) and not rebuild:
        _vectordb = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embeddings,
        )
    else:
        # Build from all PDFs
        all_docs = []
        for pdf in pdfs:
            loader = PyPDFLoader(str(pdf))
            all_docs.extend(loader.load())

        docs = _split_docs(all_docs)
        _vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(DB_PATH),
        )
        try:
            _vectordb.persist()
        except Exception:
            pass

        manifest = {"files": {}}
        for pdf in pdfs:
            src = str(pdf.resolve())
            st = pdf.stat()
            file_sha = _sha256_file(pdf)
            manifest["files"][src] = {
                "sha256": file_sha,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "ingested_at_utc": _utc_now_iso(),
            }
        _save_manifest(manifest)

        status["mode"] = "rebuild" if rebuild else "rebuild_initial"
        status["added_pdfs"] = len(pdfs)
        return status

    # Incremental ingestion: add only new/changed PDFs
    status["mode"] = "incremental"
    manifest = _load_manifest()
    manifest.setdefault("files", {})

    for pdf in pdfs:
        src = str(pdf.resolve())
        st = pdf.stat()
        file_sha = _sha256_file(pdf)

        prev = manifest["files"].get(src)
        if prev and prev.get("sha256") == file_sha:
            status["skipped_pdfs"] += 1
            continue

        if prev and prev.get("sha256") != file_sha:
            _delete_by_source(_vectordb, src)
            status["updated_pdfs"] += 1
        else:
            status["added_pdfs"] += 1

        status["chunks_added"] += _ingest_pdf(_vectordb, pdf, file_sha)

        manifest["files"][src] = {
            "sha256": file_sha,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ingested_at_utc": _utc_now_iso(),
        }

    try:
        _vectordb.persist()
    except Exception:
        pass

    _save_manifest(manifest)
    return status


def rag_search(query: str, k: int = 4) -> str:
    global _vectordb

    if _vectordb is None:
        return "RAG is not initialized. Please initialize the vector store first."

    results = _vectordb.similarity_search(query, k=k)
    if not results:
        return "No relevant information found."

    response_parts = []
    for r in results:
        src_path = r.metadata.get("source", "unknown")
        src_name = Path(src_path).name if src_path != "unknown" else "unknown"
        page0 = r.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "N/A"

        text = (r.page_content or "").strip()
        response_parts.append(f"**Kaynak:** {src_name} | **Sayfa:** {page}\n\n{text}")

    return "\n\n---\n\n".join(response_parts)

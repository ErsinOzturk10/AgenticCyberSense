```python
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DB_PATH = BASE_DIR / "chroma_db_minilm384"
MANIFEST_PATH = DB_PATH / "ingested_manifest.json"

_vectordb: Chroma | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Manifest read/parse failed, resetting: %s", e)
            return {"files": {}}
    return {"files": {}}


def _save_manifest(manifest: dict) -> None:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _list_pdfs() -> list[Path]:
    if not DATA_PATH.exists():
        return []
    return sorted([p for p in DATA_PATH.glob("*.pdf") if p.is_file()])


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)


def _make_chunk_ids(file_sha: str, docs) -> list[str]:
    return [f"{file_sha}:{d.metadata.get('page', 'na')}:{i}" for i, d in enumerate(docs)]


def _safe_persist(db: Chroma) -> None:
    """Persist() varsa çağır; yoksa sessizce geç. Beklenen hataları dar yakala."""
    persist_fn = getattr(db, "persist", None)
    if not callable(persist_fn):
        return

    try:
        persist_fn()
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Chroma persist failed: %s", e)


def _delete_by_source(vectordb: Chroma, source_path: str) -> None:
    try:
        vectordb._collection.delete(where={"source": source_path})
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.warning("Delete by source failed (%s): %s", source_path, e)


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

    # Rebuild DB if requested
    if rebuild and DB_PATH.exists():
        try:
            shutil.rmtree(DB_PATH)
        except OSError as e:
            logger.warning("Could not remove DB directory %s: %s", DB_PATH, e)
        status["mode"] = "rebuild"

    # Load existing DB if present (fast)
    if DB_PATH.exists() and any(DB_PATH.iterdir()) and not rebuild:
        _vectordb = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embeddings,
        )
    else:
        # Fresh build from all PDFs (use same ingestion path for consistent metadata)
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _vectordb = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embeddings,
        )

        manifest = {"files": {}}
        for pdf in pdfs:
            st = pdf.stat()
            file_sha = _sha256_file(pdf)

            status["chunks_added"] += _ingest_pdf(_vectordb, pdf, file_sha)
            manifest["files"][str(pdf.resolve())] = {
                "sha256": file_sha,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "ingested_at_utc": _utc_now_iso(),
            }

        _safe_persist(_vectordb)
        _save_manifest(manifest)

        status["mode"] = "rebuild" if rebuild else "rebuild_initial"
        status["added_pdfs"] = len(pdfs)
        return status

    # Incremental ingestion
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

    _safe_persist(_vectordb)
    _save_manifest(manifest)
    return status


def rag_search(query: str, k: int = 4) -> str:
    """Performs a retrieval-augmented generation (RAG) search using the provided query."""
    if _vectordb is None:
        return "RAG is not initialized. Please initialize the vector store first."

    results = _vectordb.similarity_search(query, k=k)
    if not results:
        return "No relevant information found."

    response_parts: list[str] = []
    for r in results:
        src_path = r.metadata.get("source", "unknown")
        src_name = Path(src_path).name if src_path != "unknown" else "unknown"
        page0 = r.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "N/A"
        text = (r.page_content or "").strip()
        response_parts.append(f"**Kaynak:** {src_name} | **Sayfa:** {page}\n\n{text}")

    return "\n\n---\n\n".join(response_parts)

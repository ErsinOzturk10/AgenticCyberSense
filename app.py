# app.py
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama

from rag import initialize_rag, rag_search


st.set_page_config(page_title="RAG POC", layout="wide")
st.title("RAG POC (PDF → Chroma → Retrieval → Ollama LLM)")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "rag_runs.jsonl"

# Must match rag.py
DB_PATH = Path("chroma_db_minilm384")
MANIFEST_PATH = DB_PATH / "ingested_manifest.json"


def log_run(payload: dict) -> None:
    payload = dict(payload)
    payload["ts_utc"] = datetime.now(timezone.utc).isoformat()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"files": {}}
    return {"files": {}}


def human_mb(num_bytes: int) -> str:
    try:
        return f"{num_bytes / (1024*1024):.2f} MB"
    except Exception:
        return "N/A"


@st.cache_resource(show_spinner=True)
def _init_once():
    # Initializes and returns status
    return initialize_rag(rebuild=False)


# ---- Sidebar ----
st.sidebar.header("Settings")

# Sync button first (so you can press before asking)
if st.sidebar.button("🔄 Sync / Ingest new PDFs", use_container_width=True):
    with st.spinner("Syncing PDFs into Chroma..."):
        _init_once.clear()
        status = _init_once()
    st.sidebar.success(
        f"Sync done. mode={status.get('mode')} | "
        f"added={status.get('added_pdfs')} | updated={status.get('updated_pdfs')} | "
        f"skipped={status.get('skipped_pdfs')} | chunks_added={status.get('chunks_added')}"
    )

st.sidebar.divider()

# --- Indexed PDFs panel (A) ---
with st.sidebar.expander("📚 Indexed PDFs (from manifest)", expanded=True):
    manifest = load_manifest()
    files = manifest.get("files", {}) or {}
    if not files:
        st.write("No manifest entries yet.")
    else:
        # Show newest first by ingested_at_utc
        rows = []
        for src, meta in files.items():
            rows.append(
                {
                    "file": Path(src).name,
                    "ingested_at_utc": meta.get("ingested_at_utc", ""),
                    "size": human_mb(int(meta.get("size", 0) or 0)),
                    "sha256": (meta.get("sha256", "") or "")[:10],
                }
            )
        rows.sort(key=lambda r: r.get("ingested_at_utc", ""), reverse=True)

        for r in rows:
            st.markdown(
                f"- **{r['file']}**  \n"
                f"  - ingested: `{r['ingested_at_utc']}`  \n"
                f"  - size: `{r['size']}`  \n"
                f"  - sha: `{r['sha256']}…`"
            )

st.sidebar.divider()

model_name = st.sidebar.selectbox(
    "LLM model (Ollama)",
    ["llama3.2:1b", "llama3.2:3b", "llama3.2:8b"],
    index=0,
)
base_url = st.sidebar.text_input("Ollama base_url", value="http://127.0.0.1:11434")
k = st.sidebar.slider("Top-k (retrieval)", min_value=1, max_value=12, value=4, step=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
max_tokens = st.sidebar.slider("Max tokens", min_value=64, max_value=1024, value=400, step=32)

show_context = st.sidebar.checkbox("Show retrieved context", value=True)
write_logs = st.sidebar.checkbox("Write JSONL logs", value=True)
st.sidebar.caption(f"Logs: {LOG_FILE.resolve()}")


# ---- Init on first load ----
try:
    status0 = _init_once()
    st.success(
        f"RAG initialized (Chroma ready). mode={status0.get('mode')} | "
        f"added={status0.get('added_pdfs')} | updated={status0.get('updated_pdfs')} | "
        f"skipped={status0.get('skipped_pdfs')} | chunks_added={status0.get('chunks_added')}"
    )
except Exception as e:
    st.error(f"RAG init failed: {e}")
    st.stop()


# ---- Main UI ----
query = st.text_area(
    "Ask a question (English)",
    height=90,
    placeholder="e.g., Summarize how trending news affects CTI prioritization.",
)

ask = st.button("Ask", type="primary", use_container_width=True)

if ask:
    q = query.strip()
    if not q:
        st.warning("Please enter a question.")
        st.stop()

    # 1) Retrieval
    t0 = time.perf_counter()
    context_md = rag_search(q, k=k)
    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000

    # Build a plain-text context for the LLM
    context_text = context_md.replace("**Kaynak:**", "Source:").replace("**Sayfa:**", "Page:")

    # 2) Generation
    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )

    system_prompt = (
        "You are a careful assistant for a Retrieval-Augmented Generation (RAG) app.\n"
        "Use ONLY the provided context to answer.\n"
        "If the context does not contain the answer, say exactly: "
        "\"The provided document context does not contain that information.\" "
        "Do not guess.\n"
        "When possible, cite sources by referencing the Source and Page shown in the context."
    )

    user_prompt = f"QUESTION:\n{q}\n\nCONTEXT:\n{context_text}\n\nANSWER (grounded in context):"

    t2 = time.perf_counter()
    try:
        answer = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        ).content
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        st.stop()
    t3 = time.perf_counter()

    generation_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    # --- Output ---
    st.subheader("Answer")
    st.write(answer)

    st.caption(
        f"Model: {model_name} | retrieval: {retrieval_ms:.0f} ms | "
        f"generation: {generation_ms:.0f} ms | total: {total_ms:.0f} ms"
    )

    if show_context:
        st.subheader("Retrieved context")
        st.markdown(context_md)

    # --- Log ---
    if write_logs:
        log_run(
            {
                "query": q,
                "k": k,
                "llm_model": model_name,
                "ollama_base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
                "answer": answer,
                "context": context_text,
            }
        )
        st.info("Logged to JSONL.")

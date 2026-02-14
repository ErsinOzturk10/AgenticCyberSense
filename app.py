import json
import time
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from chatbotwith_tool import _init_rag, run_agent

st.set_page_config(page_title="RAG POC", layout="wide")
st.title("RAG POC (PDF → Chroma → Retrieval → Ollama LLM)")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "rag_runs.jsonl"


def log_run(payload: dict):
    payload["ts_utc"] = datetime.now(UTC).isoformat()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@st.cache_resource(show_spinner=True)
def _init_once():
    return _init_rag()


# ---- Sidebar ----
st.sidebar.header("Settings")
if st.sidebar.button("🔄 Sync / Reingest PDFs", use_container_width=True):
    with st.spinner("Syncing PDFs..."):
        _init_once.clear()
        status = _init_once()
    if status:
        st.sidebar.success(
            f"Sync done. mode={status.get('mode')} | added={status.get('added_pdfs')} | updated={status.get('updated_pdfs')} | skipped={status.get('skipped_pdfs')} | chunks_added={status.get('chunks_added')}",
        )
    else:
        st.sidebar.warning("No PDFs found or DB path incorrect.")

st.sidebar.divider()
model_name = st.sidebar.selectbox("LLM model (Ollama)", ["llama3.2:3b"])
base_url = st.sidebar.text_input("Ollama base_url", value="http://localhost:11434")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 64, 1024, 400, 32)
write_logs = st.sidebar.checkbox("Write JSONL logs", value=True)

# ---- Init ----
try:
    status0 = _init_once()
    if status0:
        st.success(
            f"RAG initialized. mode={status0.get('mode')} | added={status0.get('added_pdfs')} | updated={status0.get('updated_pdfs')} | skipped={status0.get('skipped_pdfs')} | chunks_added={status0.get('chunks_added')}",
        )
    else:
        st.warning("RAG init returned None. Check ./data for PDFs.")
except Exception as e:
    st.error(f"RAG init failed: {e}")
    st.stop()

# ---- Main UI ----
user_input = st.text_input("Sorunu yaz:")
if user_input:
    t0 = time.perf_counter()
    answer = run_agent(user_input)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000

    st.markdown("### 🤖 Cevap")
    st.write(answer)

    if write_logs:
        log_run(
            {
                "query": user_input,
                "llm_model": model_name,
                "ollama_base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "total_ms": round(total_ms, 2),
                "answer": answer,
            },
        )
        st.info("Logged to JSONL.")

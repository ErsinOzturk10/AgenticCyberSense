"""
# ----- eski retrieval + llm generation bloğu yorum satırıyla bırakıldı -----
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
    "Use the provided context to answer if relevant.\n"
    "If the context does not contain the answer, you must generate an answer using your own knowledge, "
    "but indicate clearly that it is NOT from the documents.\n"
    "When using the context, cite sources by referencing the Source and Page shown in the context."
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
"""
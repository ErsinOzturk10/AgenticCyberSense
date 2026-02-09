"""
A Streamlit app demonstrating a LangChain agent using tool calling + RAG.
"""

import os
import re
import json
import streamlit as st
from langchain.tools import tool

from langchain_ollama import ChatOllama

from rag import rag_search as _rag_search
from rag import initialize_rag


# ================== SYSTEM PROMPT ==================

SYSTEM_PROMPT = """
You are an intelligent assistant that can use tools when necessary.

You have access to the following tools:
- technical_document_lookup
- equipment_history
- email_vendor
- rag_search

Rules:
1. Use a tool ONLY if it is necessary to answer the question.
2. Prefer specific tools when equipment codes are mentioned.
3. Use rag_search ONLY when the answer requires information from uploaded documents.
4. If no tool is relevant, answer directly.
5. If rag_search returns insufficient context, say so.
6. Do NOT hallucinate.
7. When using rag_search, cite the source.
8. If the user mentions an EQ##### code and asks about geçmiş/servis/bakım/satın alma, you MUST call equipment_history.
9. If the user asks about teknik/özellik/specs, you MUST call technical_document_lookup.
10. The user is asking about equipment, NOT a person. Do not refuse as personal information.
11. Tool args must be a plain JSON object (e.g., {"equipment_name": "EQ12345"}), not a schema.
"""


# ================== TOOLS ==================

@tool
def rag_search(query: str) -> str:
    """
    Search relevant information from uploaded PDFs.
    """
    return _rag_search(query)


@tool
def technical_document_lookup(equipment_name: str) -> str:
    # """Retrieve technical specifications for an equipment code (EQ#####)."""
    """Retrieve technical specifications (teknik özellikler, specs) for an equipment code (EQ#####)."""
    code = equipment_name.strip().upper()
    if code in {"EQ12345", "EQ67890"}:
        return (
            f"Technical details for {code}: "
            "Model X, Power: 999 W, Dimensions: 50x50x50 cm."
        )
    return f"No technical details found for {equipment_name}."


@tool
def equipment_history(equipment_name: str) -> str:
    # """Fetch service/purchase history for an equipment code (EQ#####)."""
    """Fetch service/purchase history (geçmiş, servis, bakım, satın alma) for an equipment code (EQ#####)."""
    code = equipment_name.strip().upper()
    if code == "EQ12345":
        return (
            "History for EQ12345: Purchased on 2023-01-15, "
            "Last serviced on 2024-06-10."
        )
    return f"No history found for {equipment_name}."


@tool
def email_vendor(text: str) -> str:
    """Send an email to the vendor about an equipment code."""
    code_match = re.search(r"\bEQ\d{5}\b", text.upper())
    code = code_match.group(0) if code_match else None

    if code == "EQ12345":
        return (
            f"Email sent to vendor regarding {code}.\n"
            f"Body: Dear Vendor, regarding equipment {code}. Context: {text}"
        )
    if code:
        return f"Failed to send email for {code}. Only EQ12345 is enabled."
    return "Failed to send email. No valid equipment code found."


tools = [
    technical_document_lookup,
    equipment_history,
    email_vendor,
    rag_search,
]


# ================== AGENT (LLM + TOOLS) ==================

agent_llm = ChatOllama(
    # model="qwen2.5:7b",
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434", # Ollama'nın Docker'da çalıştığı URL ve port
    temperature=0,
    system=SYSTEM_PROMPT,
    verbose=True
).bind_tools(tools)

# ================== HELPER FUNCTION ==================
def _normalize_tool_args(raw_args):
    """Fix common malformed tool-call args produced by small models."""
    if raw_args is None:
        return {}

    # Sometimes args come as JSON string
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            return {}

    if not isinstance(raw_args, dict):
        return {}

    # Common model mistake: puts real values under "properties"
    if "properties" in raw_args and isinstance(raw_args["properties"], dict):
        props = raw_args["properties"]
        # If values are primitives (not schema dicts), assume they are the actual args
        if any(not isinstance(v, dict) for v in props.values()):
            return {k: v for k, v in props.items() if not isinstance(v, dict)}

    # Another common wrapper key
    if "arguments" in raw_args:
        return _normalize_tool_args(raw_args["arguments"])

    return raw_args


# ================== AGENT LOOP ==================

def run_agent(user_input: str) -> str:
    """
    Runs the agent. If a tool is called, executes the tool and
    feeds the result back to the agent.
    """

    response = agent_llm.invoke(user_input)

    # -------- TOOL CALL HANDLING --------
    if response.tool_calls:
        tool_map = {t.name: t for t in tools}
        tool_messages = []

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            # 🔍 TOOL LOGGING (UI)
            st.write(f"🛠️ Tool çağrıldı: `{name}`")
            st.json(args)

            tool_fn = tool_map[name]
            # tool_result = tool_fn.run(args)
            fixed_args = _normalize_tool_args(args)

            st.write("✅ Normalize args:")
            st.json(fixed_args)

            try:
                tool_result = tool_fn.invoke(fixed_args)  # invoke daha modern/sağlam
            except Exception as e:
                tool_result = f"Tool execution error ({name}): {e}"

            st.write("📤 Tool sonucu:")
            st.write(tool_result)

            tool_messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call["id"],
                }
            )

        messages = [
            {"role": "user", "content": user_input},
            response,
            *tool_messages,
        ]

        final = agent_llm.invoke(messages)
        return final.content

    # -------- NO TOOL --------
    return response.content


# ================== STREAMLIT UI ==================

# initialize_rag()
@st.cache_resource(show_spinner="📚 PDF'ler indexleniyor (ilk sefer biraz sürebilir)...")
def _init_rag():
    initialize_rag()

_init_rag()



st.title("Agentic RAG Chatbot")

user_input = st.text_input("Sorunu yaz:")

if user_input:
    answer = run_agent(user_input)
    st.markdown("### 🤖 Cevap")
    st.write(answer)

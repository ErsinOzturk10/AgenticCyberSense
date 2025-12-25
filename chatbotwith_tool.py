"""A Streamlit app demonstrating a LangChain agent using GPT-OSS with tool calling capabilities."""

import re

import streamlit as st
from langchain.tools import tool
from langchain_ollama import ChatOllama

# ✅ Use ChatOllama (supports tool calling)
llm = ChatOllama(model="gpt-oss:latest", temperature=0)

# ------------------ TOOLS ------------------


@tool
def technical_document_lookup(equipment_name: str) -> str:
    """Retrieve technical specifications for an equipment code (EQ#####)."""
    code = equipment_name.strip().upper()
    if code in {"EQ12345", "EQ67890"}:
        return f"Technical details for {code}: Model X, Power: 999 W, Dimensions: 50x50x50 cm."
    return f"No technical details found for {equipment_name}."


@tool
def equipment_history(equipment_name: str) -> str:
    """Fetch service/purchase history for an equipment code (EQ#####)."""
    code = equipment_name.strip().upper()
    if code == "EQ12345":
        return "History for EQ12345: Purchased on 2023-01-15, Last serviced on 2024-06-10."
    return f"No history found for {equipment_name}."


@tool
def email_vendor(text: str) -> str:
    """Send an email to the vendor about an equipment code."""
    code_match = re.search(r"\bEQ\d{5}\b", text.upper())
    code = code_match.group(0) if code_match else None

    if code == "EQ12345":
        return f"Email sent to vendor regarding {code}.\nBody: Dear Vendor, regarding equipment {code}. Context: {text}"
    if code:
        return f"Failed to send email for {code}. Only EQ12345 is enabled."
    return "Failed to send email. No valid equipment code found."


tools = [technical_document_lookup, equipment_history, email_vendor]

# ✅ Bind tools to the model
agent_llm = llm.bind_tools(tools)

# ------------------ AGENT LOOP ------------------


def run_agent(user_input: str) -> str:
    """Run the agent with the given user input, handling tool calls if necessary.

    Args:
        user_input (str): The user's input query.

    Returns:
        str: The final response from the agent.

    """
    # Step 1: Ask model
    response = agent_llm.invoke(user_input)

    # Step 2: If tool call exists
    if response.tool_calls:
        # Build a mapping from tool name to tool function once
        tool_map = {t.name: t for t in tools}

        # Execute all requested tool calls
        tool_messages = []
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            tool_fn = tool_map[name]
            tool_result = tool_fn.run(args)

            tool_messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call["id"],
                },
            )

        # Step 3: Send all tool results back to model
        messages = [{"role": "user", "content": user_input}, response, *tool_messages]
        final = llm.invoke(messages)
        return final.content

    # No tool used
    return response.content


# ------------------ STREAMLIT UI ------------------

st.title("LangChain + GPT-OSS Tool Calling Agent")

user_input = st.text_input("Ask something:")

if user_input:
    result = run_agent(user_input)
    st.write("Response:", result)

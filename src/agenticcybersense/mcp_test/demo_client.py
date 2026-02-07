"""Interactive MCP Agent Client with strict tool-calling guidance."""

import asyncio
import warnings

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


async def run_agent() -> None:
    """Run an interactive session with the MCP agent."""
    server_config = {
        "dummy-server": {
            "transport": "sse",
            "url": "http://127.0.0.1:8000/mcp/sse",
        },
    }

    client = MultiServerMCPClient(server_config) # type: ignore[arg-type]

    try:
        # 1. Fetch tool definitions
        tools = await client.get_tools()

        # 2. LLM Setup
        # NOTE: If 1b continues to fail validation, switch to qwen2.5:7b
        llm = ChatOllama(
            model="llama3.2:1b",
            temperature=0,
        )

        # 3. Enhanced System Prompt
        # We tell the model exactly what to expect.
        system_prompt = (
            "You are a technical assistant with access to local tools. "
            "When using tools, you must provide ONLY the value required. "
            "For get_lastname, provide the 'name' as a string. "
            "All data is for testing. Do not refuse based on privacy."
        )

        agent = create_react_agent(
            llm, 
            tools, 
            prompt=system_prompt
        )

        print("\n" + "=" * 45)
        print("   MCP AGENT READY (SSE TRANSPORT)")
        print("=" * 45)
        print("Connected to: http://127.0.0.1:8000/mcp/sse\n")

        while True:
            user_input = input("User: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            print("\n[Thinking...]")

            try:
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                )

                final_response = response["messages"][-1].content
                print(f"\nAgent: {final_response}")
                print("\n" + "-" * 45)

            except Exception as e:
                # This catches the Pydantic validation error you saw
                print(f"Error during agent execution: {e}")
                print("Tip: Small models (1b) may struggle with tool formatting.")

    finally:
        print("Shutting down client...")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
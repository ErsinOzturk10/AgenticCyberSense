"""An example of an LLM agent using MCP to access tools via a local MCP server."""

import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


async def main() -> None:
    """Test MCP server and client interaction with an LLM agent."""
    here = Path(__file__).resolve().parent
    mcp_server_path = here / "mcp_server.py"

    client = MultiServerMCPClient(
        {
            "dummy": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(mcp_server_path)],
            },
        },
    )

    tools = await client.get_tools()

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
    )

    agent = create_react_agent(llm, tools)
    message_llm = input("Enter your message for the agent: ")

    resp = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message_llm}]},
    )
    print(resp["messages"][-1].content)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
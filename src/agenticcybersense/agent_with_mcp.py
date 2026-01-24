import asyncio
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def main():
    here = Path(__file__).resolve().parent
    mcp_server_path = here / "mcp_server.py"

    client = MultiServerMCPClient(
        {
            "dummy": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(mcp_server_path)],
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOllama(
        model="qwen2.5:7b", 
        temperature=0
    )

    agent = create_react_agent(llm, tools)

    resp = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Use the lastname tool and tell me the result."}]}
    )
    print(resp["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())

"""Without the LLM interaction, just test the MCP server and client interaction."""
# import asyncio
# import sys
# from pathlib import Path
# from langchain_mcp_adapters.client import MultiServerMCPClient

# async def main():
#     here = Path(__file__).resolve().parent
#     mcp_server_path = here / "mcp_server.py"

#     client = MultiServerMCPClient(
#         {
#             "dummy": {
#                 "transport": "stdio",
#                 "command": sys.executable,
#                 "args": [str(mcp_server_path)],
#             }
#         }
#     )

#     tools = await client.get_tools()
#     print("Loaded tools:", [t.name for t in tools])

#     # Call the tool directly (LangChain tool interface)
#     lastname_tool = next(t for t in tools if t.name == "lastname")
#     result = await lastname_tool.ainvoke({})
#     print("Lastname result:", result)

# if __name__ == "__main__":
#     asyncio.run(main())

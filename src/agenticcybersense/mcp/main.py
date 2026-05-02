"""Interactive MCP Agent Client with strict tool-calling guidance."""

import asyncio
import logging
import warnings

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agenticcybersense.settings import settings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent model
model_name = settings.ollama_model
HTTP_TIMEOUT_SECONDS = 60 * 60
SSE_READ_TIMEOUT_SECONDS = 60 * 60


async def run_agent() -> None:
    """Run an interactive session with the MCP agent."""
    logger.info(
        "Configured MCP timeouts: timeout=%ss, sse_read_timeout=%ss",
        HTTP_TIMEOUT_SECONDS,
        SSE_READ_TIMEOUT_SECONDS,
    )
    logger.info("MCP target URL: %s", getattr(settings, "mcp_target_url", None))

    server_config = {
        "dummy-server": {
            "transport": "sse",
            "url": settings.mcp_target_url,  # type: ignore[attr-defined]
            "timeout": float(HTTP_TIMEOUT_SECONDS),
            "sse_read_timeout": float(SSE_READ_TIMEOUT_SECONDS),
        },
    }

    client = MultiServerMCPClient(server_config)  # type: ignore[arg-type]

    try:
        # 1) Fetch tool definitions
        tools = await client.get_tools()
        logger.info("\nTools: %s", tools)
        logger.info("Using Ollama model: %s", model_name)

        # 2) LLM Setup
        llm = ChatOllama(
            model=model_name,
            temperature=0,
        )

        # 3) System Prompt
        system_prompt = (
            "You are an intelligent assistant that can use tools when necessary.\n"
            "But rag_search is mandatory as the first tool call for every user request.\n"
            "Never call any tools before rag_search.\n"
            "After rag_search returns, use the rag_search output as context for deciding what to do next.\n"
            "After rag_search, decide whether any additional tool is required.\n"
            "Do not produce the final answer until all necessary tool calls are completed.\n"
            "You have access to the following tools: rag_search, telegram_search.\n"
            "When using tools, you must provide ONLY the value required.\n"
            "Rules for rag_search tool usage:\n"
            "- rag_search is mandatory as the first tool call for every user request.\n"
            "- Do NOT hallucinate. If you don't know, say you don't know.\n"
            "- When using rag_search, always cite the source.\n"
            "Use telegram_search for recent cyber threat chatter, CVEs, CTI, leaks, malware campaigns, APT activity or Telegram-sourced threat intel.\n"
            "When tools return information, you must base your final answer on the relevant outputs from all tools used.\n"
            "Do not ignore any relevant tool output.\n"
            "If a tool output is irrelevant, empty, insufficient, vague, or contradictory, explicitly account for that instead of guessing.\n"
            "Do not ignore tool results or replace them with generic advice.\n\n"
            "Critical rules for telegram_search:\n"
            "- After rag_search, use telegram_search for all Telegram-related prompts.\n"
            "- Treat the telegram_search output as the source of truth.\n"
            "- If the tool output is vague, say what is missing instead of guessing.\n"
        )

        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )

        logger.info("\n" + "=" * 45)  # noqa: G003
        logger.info("   MCP AGENT READY (SSE TRANSPORT)")
        logger.info("=" * 45)
        logger.info("Connected to the MCP server.\n")

        while True:
            user_input = input("User: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            logger.info("\n[Thinking...]")

            try:
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                )

                logger.info("\nAgent_response_as_a_dict: %s", response)
                logger.info("\n" + "-" * 45)  # noqa: G003

                tools_used = []
                seen_tool_names = set()

                for message in response.get("messages", []):
                    for tool_call in getattr(message, "tool_calls", []) or []:
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get("name")
                        else:
                            tool_name = getattr(tool_call, "name", None)

                        if tool_name and tool_name not in seen_tool_names:
                            tools_used.append(tool_name)
                            seen_tool_names.add(tool_name)

                    if getattr(message, "type", None) == "tool":
                        tool_name = getattr(message, "name", None)

                        if tool_name and tool_name not in seen_tool_names:
                            tools_used.append(tool_name)
                            seen_tool_names.add(tool_name)

                logger.info("Tools used in this response: %s", tools_used)
                logger.info("\n" + "-" * 45)  # noqa: G003

                chunks = response["messages"][2].artifact["structured_content"]["result"].split("---")
                for i, chunk in enumerate(chunks):
                    if "**Source:**" in chunk and "**Page:**" in chunk:
                        source = chunk.split("**Source:**")[1].split("|")[0].strip()
                        page = chunk.split("**Page:**")[1].split("\n")[0].strip()
                        logger.info("Chunk %d Source: %s", i + 1, source)
                        logger.info("Chunk %d Page: %s", i + 1, page)

                final_response = response["messages"][-1].content
                logger.info(f"\nAgent: {final_response}")  # noqa: G004
                logger.info("\n" + "-" * 45)  # noqa: G003

            except Exception:
                logger.exception("Pydantic validation failed")

    finally:
        logger.info("Shutting down client...")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

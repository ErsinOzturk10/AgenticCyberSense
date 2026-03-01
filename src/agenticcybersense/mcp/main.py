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

# Change to "qwen2.5:1b" if you want to test with a smaller model
# model_name = "qwen2.5:7b"
model_name = "llama3.2:3b"


async def run_agent() -> None:
    """Run an interactive session with the MCP agent."""
    server_config = {
        "dummy-server": {
            "transport": "sse",
            "url": settings.mcp_target_url,  # type: ignore[attr-defined
        },
    }

    client = MultiServerMCPClient(server_config)  # type: ignore[arg-type]

    try:
        # 1. Fetch tool definitions
        tools = await client.get_tools()

        # 2. LLM Setup
        llm = ChatOllama(
            model=model_name,
            temperature=0,
        )

        # 3. Enhanced System Prompt
        # We tell the model exactly what to expect.
        system_prompt = (
            "You are an intelligent assistant that can use tools when necessary. "
            "You have access to the following tools: rag_search, get_lastname, process_age, reverse_name."
            "When using tools, you must provide ONLY the value required. "
            "For get_lastname, provide the 'name' as a string. "
            "All data is for testing. Do not refuse based on privacy."
            
            """
            Rules for rag_search tool usage:
                - Use a tool ONLY if it is necessary to answer the question.
                - Use rag_search ONLY when the answer requires information from uploaded documents.
                - If no tool is relevant, answer directly.
                - If rag_search returns insufficient context, say so.
                - Do NOT hallucinate. If you don't know, say you don't know.
                - When using rag_search, always cite the source.
                - If the question is about general world knowledge (e.g., weather, geography, politics), answer directly without any tool.
                - Never call rag_search for unrelated general questions.
            """
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

                final_response = response["messages"][-1].content
                logger.info(f"\nAgent: {final_response}")  # noqa: G004
                logger.info("\n" + "-" * 45)  # noqa: G003

            except Exception:  # noqa: BLE001
                # This catches the Pydantic validation error you saw
                logger.error("Tip: Small models (1b) may struggle with tool formatting.")  # noqa: TRY400

    finally:
        logger.info("Shutting down client...")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

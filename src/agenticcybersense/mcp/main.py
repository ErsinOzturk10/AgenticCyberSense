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

# Initialize RAG vector store at startup, set rebuild=True to force re-ingestion of PDFs.
# In production, you might want to set this to False to avoid unnecessary reprocessing on every startup.

# Change to "qwen2.5:1b" if you want to test with a smaller model
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
    # The client will automatically connect to the server when you call get_tools() for the first time.

    try:
        # 1. Fetch tool definitions
        tools = await client.get_tools()
        logger.info("\nTools: %s", tools)
        # This will show the tool definitions received from the MCP server. The agent will use this information to know how to call the tools when needed.
        # The tools variable is a dictionary of tool names to their definitions, which the agent will use to know how to call them.

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
                # If the user just presses Enter without typing anything, we can choose to ignore it or prompt them again.
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            logger.info("\n[Thinking...]")

            try:
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                )
                # The response is a dictionary that includes the agent's messages and any tool calls it made.
                # The final response from the agent is typically in the last message.
                logger.info("\nAgent_response_as_a_dict: %s", response)  # To see the full response structure
                logger.info("\n" + "-" * 45)  # noqa: G003
                # Log the tool name(s) used in the response, if any
                logger.info("Tools used in this response: %s", response["messages"][1].tool_calls[0]["name"])
                logger.info("\n" + "-" * 45)  # noqa: G003
                chunks = response["messages"][2].artifact["structured_content"]["result"].split("---")

                for i, chunk in enumerate(chunks):
                    if "**Source:**" in chunk and "**Page:**" in chunk:
                        source = chunk.split("**Source:**")[1].split("|")[0].strip()
                        page = chunk.split("**Page:**")[1].split("\n")[0].strip()
                        logger.info("Chunk %d Source: %s", i + 1, source)
                        logger.info("Chunk %d Page: %s", i + 1, page)
                final_response = response["messages"][-1].content
                # The agent's final response is what it would say to the user after processing the input and any tool calls.
                # It should reflect the agent's reasoning and the results of any tools it used.
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

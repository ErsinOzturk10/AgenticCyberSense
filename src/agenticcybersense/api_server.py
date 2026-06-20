"""API Server for OpenWebUI integration."""

from __future__ import annotations

import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agenticcybersense.logging_utils import get_logger, setup_logging
from agenticcybersense.settings import settings
from agenticcybersense.web_crawler.config import SCHEDULE_HOUR, SCHEDULE_MINUTE

logger = get_logger("api_server")

# Global graph instance
_graph: Any | None = None

# Scheduler defaults
SCHED_HOUR = SCHEDULE_HOUR
SCHED_MINUTE = SCHEDULE_MINUTE

# Streaming chunk size for SSE responses
STREAM_CHUNK_SIZE = 20


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler — runs on startup and shutdown."""
    global _graph  # noqa: PLW0603

    setup_logging(settings.log_level)
    logger.info("=" * 60)
    logger.info("Starting AgenticCyberSense API Server")
    logger.info("Port: %s", settings.api_port)
    logger.info("=" * 60)

    # Initialize RAG vector store so DocumentationAgent can retrieve documents.
    try:
        from agenticcybersense.rag.rag import initialize_rag  # noqa: PLC0415

        status = initialize_rag(rebuild=False)
        logger.info("RAG initialized: %s", status)
    except Exception:
        logger.exception("RAG initialization failed (documentation agent will be limited)")

    # Build the LangGraph orchestration graph.
    try:
        from agenticcybersense.graph.build_graph import build_graph  # noqa: PLC0415

        _graph = build_graph()
        logger.info("Orchestration graph initialized")
    except Exception:
        logger.exception("Failed to initialize graph")
        _graph = None

    # Start the webcrawler scheduler.
    try:
        from agenticcybersense.web_crawler.crawler_scheduler import start_scheduler  # noqa: PLC0415

        await start_scheduler(hour=SCHED_HOUR, minute=SCHED_MINUTE)
        logger.info("Crawler scheduler started")
    except Exception:
        logger.exception("Crawler scheduler could not be started")

    try:
        yield
    finally:
        try:
            from agenticcybersense.web_crawler.crawler_scheduler import stop_scheduler  # noqa: PLC0415

            await stop_scheduler()
        except Exception:
            logger.exception("Crawler scheduler could not be stopped")

        logger.info("Shutting down API Server")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgenticCyberSense API",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow all origins so OpenWebUI running in Docker can reach this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_raw_telegram_retrieval_query(query: str) -> bool:
    """Detect user requests that should bypass orchestration and return raw Telegram output.

    These requests are not asking for a synthesized CTI report. They are asking
    for the actual Telegram messages, so passing them through the orchestrator may
    summarize or truncate the output.
    """
    q = (query or "").lower().strip()
    if not q:
        return False

    telegram_terms = [
        "telegram",
        "channel",
        "channels",
        "kanal",
        "kanallar",
    ]

    raw_retrieval_terms = [
        "all messages",
        "retrieve all",
        "get all messages",
        "fetch all messages",
        "without any missing",
        "without missing",
        "no missing parts",
        "full messages",
        "complete messages",
        "entire channel",
        "all channels",
        "all telegram channels",
        "raw messages",
        "message content",
        "messages content",
    ]

    has_telegram_context = any(term in q for term in telegram_terms) or re.search(r"@[\w\d_]+", query or "") is not None
    has_raw_retrieval_intent = any(term in q for term in raw_retrieval_terms)

    return has_telegram_context and has_raw_retrieval_intent


async def process_raw_telegram_query(query: str) -> str:
    """Run TelegramAgent directly and return its raw report without final graph synthesis."""
    from agenticcybersense.agents.telegram.telegram import TelegramAgent  # noqa: PLC0415
    from agenticcybersense.schemas.messages import AgentRequest  # noqa: PLC0415

    logger.info("Direct raw Telegram retrieval mode enabled")
    agent = TelegramAgent()
    response = await agent.process(AgentRequest(query=query))

    content = response.content or ""
    if not content.strip():
        return "Telegram agent returned an empty response."

    return content


async def process_with_agents(query: str, conversation_id: str, context: dict[str, Any] | None = None) -> str:
    """Route a query through the agent orchestration graph and return the final response."""
    if _graph is None:
        return "Orchestration graph not available"

    final_response = "Processing complete."

    try:
        initial_state = {
            "query": query,
            "conversation_id": conversation_id,
            "context": context or {},
            "current_agent": "orchestrator",
            "agents_consulted": [],
            "pending_agents": [],
            "agent_responses": {},
            "findings": [],
            "documentation_context": "",
            "final_response": "",
            "is_complete": False,
            "error": None,
        }

        logger.info("Processing query: %s", query[:100])
        result = cast("dict[str, Any]", await _graph.ainvoke(initial_state))

        final_response = str(result.get("final_response", "Processing complete."))
        agents_raw = result.get("agents_consulted", [])
        agents = [str(agent) for agent in agents_raw] if isinstance(agents_raw, list) else []
        findings_raw = result.get("findings", [])
        findings_count = len(findings_raw) if isinstance(findings_raw, list) else 0

        if agents:
            final_response += f"\n\n---\n*Agents: {', '.join(agents)} | Findings: {findings_count}*"

    except Exception:
        logger.exception("Error while composing final response")
        raise

    return final_response


async def generate_response_content(user_message: str, conversation_id: str) -> str:
    """Generate response content for OpenAI-compatible chat completion.

    Raw Telegram retrieval requests bypass the orchestration graph so the output
    is not summarized, sampled, or transformed into a synthesized CTI report.
    All other requests continue to use the normal multi-agent graph.
    """
    if is_raw_telegram_retrieval_query(user_message):
        return await process_raw_telegram_query(user_message)

    return await process_with_agents(user_message, conversation_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, Any]:
    """Health/info endpoint."""
    return {"name": "AgenticCyberSense", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Detailed health check."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> dict[str, Any]:
    """List available models — OpenAI-compatible format."""
    logger.info("GET /v1/models called")
    return {
        "object": "list",
        "data": [
            {
                "id": "agenticcybersense",
                "object": "model",
                "created": int(datetime.now(UTC).timestamp()),
                "owned_by": "local",
            },
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
@app.post("/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming SSE and non-streaming responses.
    """
    logger.info("=" * 40)
    logger.info("POST /v1/chat/completions called")

    try:
        body = await request.json()
        logger.info("Request body: %s", json.dumps(body, indent=2)[:500])

        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "No messages provided", "type": "invalid_request_error"}},
            )

        # Extract the last user message from the conversation history.
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        logger.info("User message: %s", user_message[:200] if user_message else "EMPTY")

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
            )

        stream = bool(body.get("stream", False))
        model = body.get("model", "agenticcybersense")
        conversation_id = str(uuid.uuid4())

        logger.info(
            "Active LLM provider: %s | model: %s",
            settings.normalized_llm_provider(),
            settings.active_llm_model(),
        )

        response_content = await generate_response_content(user_message, conversation_id)
        logger.info("Response generated: %d chars", len(response_content))

        # --- Streaming response ---
        if stream:

            async def generate() -> AsyncGenerator[str, None]:
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(datetime.now(UTC).timestamp())

                # Send response content in small chunks to simulate streaming.
                for i in range(0, len(response_content), STREAM_CHUNK_SIZE):
                    chunk_text = response_content[i : i + STREAM_CHUNK_SIZE]
                    chunk_data = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                final_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        # --- Non-streaming response ---
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split()),
            },
        }

        return JSONResponse(content=response_data)

    except json.JSONDecodeError:
        logger.exception("JSON decode error")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON payload", "type": "invalid_request_error"}},
        )
    except Exception as e:
        logger.exception("Error in chat_completions")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


@app.get("/v1/agents")
@app.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all registered CTI agents."""
    try:
        from agenticcybersense.agents.registry import get_registry  # noqa: PLC0415

        registry = get_registry()
        return {"agents": registry.get_agent_descriptions()}
    except Exception:
        logger.exception("Failed to list agents")
        return {"agents": {}, "error": "Failed to list agents"}


def run_server(host: str | None = None, port: int | None = None) -> None:
    """Start the API server using uvicorn."""
    import uvicorn  # noqa: PLC0415

    host = host or settings.api_host
    port = port or settings.api_port

    logger.info("Starting AgenticCyberSense API Server on %s:%s", host, port)
    uvicorn.run("agenticcybersense.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()

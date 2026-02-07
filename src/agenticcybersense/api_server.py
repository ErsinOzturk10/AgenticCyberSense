"""API Server for OpenWebUI integration."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agenticcybersense.logging_utils import get_logger, setup_logging
from agenticcybersense.settings import settings

logger = get_logger("api_server")

# Global graph instance
_graph = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    global _graph  # noqa: PLW0603
    setup_logging(settings.log_level)
    logger.info("=" * 60)
    logger.info("Starting AgenticCyberSense API Server")
    logger.info("Port: %s", settings.api_port)
    logger.info("=" * 60)

    try:
        from agenticcybersense.graph.build_graph import build_graph  # noqa: PLC0415

        _graph = build_graph()
        logger.info("✅ Orchestration graph initialized")
    except Exception:
        logger.exception("❌ Failed to initialize graph")
        _graph = None

    try:
        yield
    finally:
        logger.info("Shutting down API Server")


# Create the FastAPI app
app = FastAPI(
    title="AgenticCyberSense API",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_with_agents(query: str, conversation_id: str, context: dict[str, Any] | None = None) -> str:
    """Process a query through the agent orchestration system."""
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
        result = await _graph.ainvoke(initial_state)

        final_response = result.get("final_response", "Processing complete.")
        agents = result.get("agents_consulted", [])
        findings_count = len(result.get("findings", []))

        if agents:
            final_response += f"\n\n---\n*Agents: {', '.join(agents)} | Findings: {findings_count}*"

    except Exception:
        logger.exception("Error while composing final response")
        raise

    return final_response


# ============================================
# API Endpoints
# ============================================


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {"name": "AgenticCyberSense", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> dict[str, Any]:
    """List available models (OpenAI compatible)."""
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
    """OpenAI-compatible chat completions endpoint."""
    logger.info("=" * 40)
    logger.info("POST /v1/chat/completions called")

    try:
        # Parse request body
        body = await request.json()
        logger.info("Request body: %s", json.dumps(body, indent=2)[:500])

        # Extract messages
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "No messages provided", "type": "invalid_request_error"}},
            )

        # Find the last user message
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

        # Process the query
        stream = body.get("stream", False)
        model = body.get("model", "agenticcybersense")
        conversation_id = str(uuid.uuid4())

        response_content = await process_with_agents(user_message, conversation_id)
        logger.info("Response generated: %d chars", len(response_content))

        if stream:

            async def generate() -> AsyncGenerator[str, None]:
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(datetime.now(UTC).timestamp())

                # Send content in chunks
                for i in range(0, len(response_content), 20):
                    chunk_text = response_content[i : i + 20]
                    chunk_data = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Final chunk
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

        # Non-streaming response
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
    """List available CTI agents."""
    try:
        from agenticcybersense.agents.registry import get_registry  # noqa: PLC0415

        registry = get_registry()
        return {"agents": registry.get_agent_descriptions()}
    except Exception:
        logger.exception("Failed to list agents")
        return {"agents": {}, "error": "Failed to list agents"}


def run_server(host: str | None = None, port: int | None = None) -> None:
    """Run the API server."""
    import uvicorn  # local import is intentional  # noqa: PLC0415

    host = host or settings.api_host
    port = port or settings.api_port

    logger.info("Starting AgenticCyberSense API Server on %s:%s", host, port)
    uvicorn.run("agenticcybersense.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()

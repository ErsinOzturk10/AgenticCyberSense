"""MCP Server using FastAPI and FastMCP."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agenticcybersense.mcp.tools import mcp
from agenticcybersense.rag.rag import initialize_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared resources when the server starts."""
    logger.info("Initializing RAG in MCP server process...")
    logger.debug("FastAPI app instance: %r", app)
    try:
        status = initialize_rag(rebuild=False)
        logger.info("RAG initialized successfully: %s", status)
    except Exception:
        logger.exception("RAG initialization failed during server startup.")
        raise

    yield

    logger.info("MCP server shutting down.")


app = FastAPI(title="MCP FastAPI Server", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint to check server status."""
    return {"status": "MCP FastAPI Server running"}


# Mount the MCP SSE application
app.mount("/mcp", mcp.sse_app())


if __name__ == "__main__":
    uvicorn.run("agenticcybersense.mcp.server:app", host="0.0.0.0", port=8000, reload=False)  # noqa: S104

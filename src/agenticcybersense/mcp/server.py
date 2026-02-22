"""MCP Server using FastAPI and FastMCP."""

import logging

import uvicorn
from fastapi import FastAPI

# Import mcp and tools from tools module
from agenticcybersense.mcp.tools import mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(title="MCP FastAPI Server")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint to check server status."""
    return {"status": "MCP FastAPI Server running"}


# Mount the MCP SSE application
app.mount("/mcp", mcp.sse_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104

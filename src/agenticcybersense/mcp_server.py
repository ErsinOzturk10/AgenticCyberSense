from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dummy-mcp-server")

@mcp.tool()
def lastname() -> str:
    """Returns lastname tool."""
    return "Arslanoglu"

if __name__ == "__main__":
    # Runs as an MCP stdio server (what LangChain MCP client expects)
    mcp.run()

import asyncio
from fastmcp import Client, FastMCP

def get_mcp_client(sandbox_gateway, sandbox_uuid: str):
    config = {
        "mcpServers": {
            "sandbox": {
                # Remote HTTP/SSE server
                "transport": "http",  # or "sse" 
                "url": f"{sandbox_gateway}/mcp",
                "headers": {"X-MCP-Session-ID": sandbox_uuid},
            }
        },
        
    }
    return Client(config, timeout=60)
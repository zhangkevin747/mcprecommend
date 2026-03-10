"""MCP server client: spin up servers via npx, list tools, call tools."""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_CALL_TIMEOUT, MCP_CONNECT_TIMEOUT, SMITHERY_API_KEY

log = logging.getLogger(__name__)


class MCPServerConnection:
    """Manages a connection to a single MCP server over stdio."""

    def __init__(self, server_id: str, command: str, args: list[str], env: dict | None = None):
        self.server_id = server_id
        self.command = command
        self.args = args
        self.env = env
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
        self._exit_stack: AsyncExitStack | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the MCP server. Returns True on success."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={**os.environ, "BROWSER": "false", **(self.env or {})},
        )
        try:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            read, write = await asyncio.wait_for(
                self._exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=MCP_CONNECT_TIMEOUT,
            )
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self.session.initialize(), timeout=MCP_CONNECT_TIMEOUT)
            self._connected = True
            log.info(f"Connected to {self.server_id}")
            return True
        except Exception as e:
            log.warning(f"Failed to connect to {self.server_id}: {e}")
            await self.close()
            return False

    async def list_tools(self) -> list[dict]:
        """Get tool schemas from the server."""
        if not self._connected or not self.session:
            return []
        try:
            result = await asyncio.wait_for(
                self.session.list_tools(), timeout=MCP_CALL_TIMEOUT
            )
            self.tools = []
            for tool in result.tools:
                self.tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                    "server_id": self.server_id,
                })
            log.info(f"{self.server_id}: {len(self.tools)} tools")
            return self.tools
        except Exception as e:
            log.warning(f"Failed to list tools from {self.server_id}: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on this server. Returns {"content": ..., "error": ...}."""
        if not self._connected or not self.session:
            return {"content": None, "error": "Not connected"}
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments),
                timeout=MCP_CALL_TIMEOUT,
            )
            # Extract text content from result
            content_parts = []
            for part in result.content:
                if hasattr(part, "text"):
                    content_parts.append(part.text)
                else:
                    content_parts.append(str(part))
            return {
                "content": "\n".join(content_parts),
                "error": None,
                "is_error": getattr(result, "isError", False),
            }
        except Exception as e:
            log.warning(f"Tool call failed on {self.server_id}/{tool_name}: {e}")
            return {"content": None, "error": str(e)}

    async def close(self):
        """Clean up the connection."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
        self._connected = False
        self.session = None
        self._exit_stack = None


SMITHERY_API_BASE = "https://api.smithery.ai"
SMITHERY_NAMESPACE = "leech-AuZq"


def _parse_sse_json(text: str) -> dict:
    """Parse SSE response to extract JSON-RPC result.

    Smithery Connect returns SSE format: 'event: message\\ndata: {...}'
    """
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    # Fallback: try parsing as plain JSON
    return json.loads(text)


class SmitheryConnection:
    """Connects to a Smithery-hosted MCP server via Smithery Connect REST API.

    No OAuth, no browser popups. Uses API key auth to create a managed
    connection, then calls tools via JSON-RPC 2.0 over HTTP (SSE responses).
    """

    def __init__(self, server_id: str, mcp_url: str):
        self.server_id = server_id
        self.mcp_url = mcp_url
        self.connection_id: str | None = None
        self.tools: list[dict] = []
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(MCP_CALL_TIMEOUT, connect=10, read=MCP_CALL_TIMEOUT),
            headers={
                "Authorization": f"Bearer {SMITHERY_API_KEY}",
                "Accept": "application/json, text/event-stream",
            },
        )

    async def connect(self) -> bool:
        """Create a Smithery Connect connection. Returns True on success."""
        try:
            resp = None
            for attempt in range(4):
                resp = await self._client.post(
                    f"{SMITHERY_API_BASE}/connect/{SMITHERY_NAMESPACE}",
                    json={"mcpUrl": self.mcp_url},
                )
                if resp.status_code != 429:
                    break
                wait = 2 ** attempt
                log.debug(f"Smithery 429 for {self.server_id}, retrying in {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            if resp.status_code not in (200, 201):
                log.warning(f"Smithery connect failed for {self.server_id}: {resp.status_code} {resp.text[:200]}")
                return False
            data = resp.json()
            self.connection_id = data.get("connectionId")
            status = data.get("status", {})
            state = status.get("state", "") if isinstance(status, dict) else status
            if state == "auth_required":
                log.info(f"{self.server_id}: Smithery auth_required (needs OAuth grant), skipping")
                return False
            if not self.connection_id:
                log.warning(f"{self.server_id}: no connectionId in Smithery response")
                return False
            log.info(f"Smithery connected: {self.server_id} (conn={self.connection_id})")
            return True
        except Exception as e:
            log.warning(f"Smithery connect error for {self.server_id}: {e}")
            return False

    def _mcp_endpoint(self) -> str:
        return f"{SMITHERY_API_BASE}/connect/{SMITHERY_NAMESPACE}/{self.connection_id}/mcp"

    async def _mcp_initialize(self) -> None:
        """Send MCP initialize + initialized notification. Required by some servers before tools/list."""
        ep = self._mcp_endpoint()
        try:
            await asyncio.wait_for(self._client.post(ep, json={
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1.0"},
                },
            }), timeout=15)
            await asyncio.wait_for(self._client.post(ep, json={
                "jsonrpc": "2.0", "method": "notifications/initialized",
            }), timeout=10)
        except Exception:
            pass

    async def list_tools(self) -> list[dict]:
        """List tools via JSON-RPC 2.0 over Smithery Connect."""
        if not self.connection_id:
            return []
        await self._mcp_initialize()
        try:
            resp = await asyncio.wait_for(self._client.post(
                self._mcp_endpoint(),
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            ), timeout=MCP_CALL_TIMEOUT)
            if resp.status_code != 200:
                log.warning(f"Smithery list_tools failed for {self.server_id}: {resp.status_code}")
                return []
            data = _parse_sse_json(resp.text)
            if "error" in data:
                log.warning(f"Smithery list_tools error for {self.server_id}: {data['error']}")
                return []
            raw_tools = data.get("result", {}).get("tools", [])
            self.tools = []
            for t in raw_tools:
                self.tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("inputSchema", {}),
                    "server_id": self.server_id,
                })
            log.info(f"{self.server_id}: {len(self.tools)} tools (via Smithery Connect)")
            return self.tools
        except Exception as e:
            log.warning(f"Smithery list_tools error for {self.server_id}: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool via JSON-RPC 2.0 over Smithery Connect."""
        if not self.connection_id:
            return {"content": None, "error": "No Smithery connection"}
        try:
            resp = await self._client.post(
                self._mcp_endpoint(),
                json={
                    "jsonrpc": "2.0", "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )
            if resp.status_code != 200:
                return {"content": None, "error": f"Smithery HTTP {resp.status_code}"}
            data = _parse_sse_json(resp.text)
            if "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                return {"content": None, "error": msg}
            result = data.get("result", {})
            content_parts = []
            for part in result.get("content", []):
                if isinstance(part, dict) and "text" in part:
                    content_parts.append(part["text"])
                elif isinstance(part, str):
                    content_parts.append(part)
                else:
                    content_parts.append(str(part))
            return {
                "content": "\n".join(content_parts) if content_parts else str(result),
                "error": None,
                "is_error": result.get("isError", False),
            }
        except Exception as e:
            log.warning(f"Smithery call_tool failed on {self.server_id}/{tool_name}: {e}")
            return {"content": None, "error": str(e)}

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


def get_smithery_mcp_url(server: dict) -> str | None:
    """Extract the Smithery MCP URL from server args or derive from id."""
    args = server.get("args", [])
    for a in args:
        if isinstance(a, str) and "server.smithery.ai" in a:
            return a
    if "smithery" in server.get("sources", []):
        return f"https://server.smithery.ai/{server['id']}/mcp"
    return None


# Verified npm packages that spin up and expose tools via npx.
# Maps server_id patterns → (npm_package, extra_args, required_env_vars)
VERIFIED_NPM_PACKAGES: dict[str, dict] = {
    # Search & web
    "tavily-mcp": {"pkg": "tavily-mcp", "env_keys": ["TAVILY_API_KEY"]},
    "exa": {"pkg": "exa-mcp-server"},
    "exa-mcp-server": {"pkg": "exa-mcp-server"},
    "brave-search-mcp": {"pkg": "brave-search-mcp", "env_keys": ["BRAVE_API_KEY"]},
    "mcp-server-brave-search": {"pkg": "mcp-server-brave-search", "env_keys": ["BRAVE_API_KEY"]},
    # General purpose
    "@modelcontextprotocol/server-everything": {"pkg": "@modelcontextprotocol/server-everything"},
    "@modelcontextprotocol/server-memory": {"pkg": "@modelcontextprotocol/server-memory"},
    "@modelcontextprotocol/server-filesystem": {"pkg": "@modelcontextprotocol/server-filesystem", "extra_args": ["/tmp"]},
    "@modelcontextprotocol/server-sequential-thinking": {"pkg": "@modelcontextprotocol/server-sequential-thinking"},
    # Browser
    "@playwright/mcp": {"pkg": "@playwright/mcp"},
    # Crawling
    "firecrawl-mcp": {"pkg": "firecrawl-mcp", "env_keys": ["FIRECRAWL_API_KEY"]},
}


def derive_smithery_command(server: dict) -> tuple[str, list[str], dict] | None:
    """Derive mcp-remote command for Smithery-hosted servers.

    Uses npx mcp-remote to proxy Smithery's streamable HTTP endpoint
    via stdio. Requires OAuth (browser popup on first use, then cached).

    Returns (command, args, env) or None if not a Smithery server.
    """
    sources = server.get("sources", [])
    is_deployed = server.get("is_deployed", False)

    if "smithery" not in sources or not is_deployed:
        return None

    server_id = server["id"]
    url = f"https://server.smithery.ai/{server_id}/mcp"
    return ("npx", ["-y", "mcp-remote", url], {})


def derive_npx_command(server: dict) -> tuple[str, list[str], dict] | None:
    """Derive npx command + args + env from server metadata.

    Returns (command, args, env) or None if we can't figure it out.
    Checks verified packages first, then Smithery hosted, then heuristics.
    """
    server_id = server["id"]

    # Check verified npm packages by server_id
    if server_id in VERIFIED_NPM_PACKAGES:
        info = VERIFIED_NPM_PACKAGES[server_id]
        pkg = info["pkg"]
        extra = info.get("extra_args", [])
        env = {}
        for k in info.get("env_keys", []):
            val = os.environ.get(k, "")
            if val:
                env[k] = val
            else:
                return None  # Missing required env var
        return ("npx", ["-y", pkg] + extra, env)

    # Check by npm package name matching
    for key, info in VERIFIED_NPM_PACKAGES.items():
        if info["pkg"] == server_id or server_id.endswith(f"/{info['pkg']}"):
            extra = info.get("extra_args", [])
            env = {}
            for k in info.get("env_keys", []):
                val = os.environ.get(k, "")
                if val:
                    env[k] = val
                else:
                    return None
            return ("npx", ["-y", info["pkg"]] + extra, env)

    # Try Smithery hosted servers via mcp-remote
    smithery_cmd = derive_smithery_command(server)
    if smithery_cmd:
        return smithery_cmd

    return None


# Verified servers to always include as fallbacks
FALLBACK_SERVERS = [
    {
        "id": "__fallback_tavily",
        "name": "Tavily Search",
        "description": "Web search using Tavily AI for current information, news, and research.",
        "command": "npx",
        "args": ["-y", "tavily-mcp"],
        "env": {"TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", "")},
    },
    {
        "id": "__fallback_exa",
        "name": "Exa Search",
        "description": "AI-powered web search, company research, and code context retrieval.",
        "command": "npx",
        "args": ["-y", "exa-mcp-server"],
    },
]

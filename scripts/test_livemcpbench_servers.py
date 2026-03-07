"""Parallel test-install of LiveMCPBench MCP servers.

Downloads all_config.json, deduplicates against our verified pool,
and tests all new servers in parallel (semaphore-limited).
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp2.mcp_client import MCPServerConnection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Max concurrent server tests (each spawns a subprocess)
MAX_CONCURRENT = 8

# Servers to skip: either already in our pool (by command) or problematic
SKIP_COMMANDS = {
    "npx -y 12306-mcp",
    "npx -y @modelcontextprotocol/server-puppeteer",
    "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
    "npx -y mcp-crypto-price",
    "uvx osm-mcp-server",
}

# Servers that need filesystem access or special env — skip for now
SKIP_IDS = {
    "filesystem",        # needs path arg, security concern
    "desktop-commander", # needs desktop access
    "docker-mcp",        # needs Docker daemon
    "git",               # needs git repo path
    "basic-memory",      # needs local state
    "memory",            # needs local state
    "taskmanager",       # utility, not tool-use relevant
    "arxiv-mcp-server",  # needs --storage-path to specific dir
}


def load_livemcpbench_servers() -> list[dict]:
    """Parse LiveMCPBench all_config.json into flat server list."""
    config_path = ROOT / "data" / "raw" / "livemcpbench_all_config.json"
    with open(config_path) as f:
        entries = json.load(f)

    servers = []
    for entry in entries:
        config = entry.get("config", {}).get("mcpServers", {})
        for sid, cfg in config.items():
            cmd_str = f"{cfg['command']} {' '.join(cfg.get('args', []))}"

            if cmd_str in SKIP_COMMANDS:
                continue
            if sid in SKIP_IDS:
                continue
            # Skip servers needing env vars (API keys)
            if cfg.get("env"):
                log.info(f"Skipping {sid}: requires env vars")
                continue

            servers.append({
                "id": sid,
                "name": entry.get("name", sid),
                "command": cfg["command"],
                "args": cfg.get("args", []),
                "env": cfg.get("env"),
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "web": entry.get("web", ""),
            })

    return servers


def guess_probe_args(tool: dict) -> dict:
    """Guess minimal arguments for a tool call based on its input schema."""
    schema = tool.get("input_schema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    args = {}
    for prop_name in required:
        prop = properties.get(prop_name, {})
        prop_type = prop.get("type", "string")

        if prop_type == "string":
            name_lower = prop_name.lower()
            if "query" in name_lower or "search" in name_lower or "keyword" in name_lower:
                args[prop_name] = "bitcoin price"
            elif "url" in name_lower:
                args[prop_name] = "https://example.com"
            elif "currency" in name_lower or "from" in name_lower:
                args[prop_name] = "USD"
            elif "to" in name_lower:
                args[prop_name] = "EUR"
            elif "location" in name_lower or "city" in name_lower or "place" in name_lower or "address" in name_lower:
                args[prop_name] = "New York"
            elif "date" in name_lower:
                args[prop_name] = "2025-01-01"
            elif "symbol" in name_lower or "ticker" in name_lower or "coin" in name_lower:
                args[prop_name] = "AAPL"
            elif "path" in name_lower or "file" in name_lower:
                args[prop_name] = "/tmp/test.txt"
            elif "id" in name_lower:
                args[prop_name] = "1"
            elif "name" in name_lower:
                args[prop_name] = "test"
            elif "thought" in name_lower or "text" in name_lower or "content" in name_lower:
                args[prop_name] = "This is a test."
            else:
                args[prop_name] = "test"
        elif prop_type in ("number", "integer"):
            if "limit" in prop_name.lower() or "max" in prop_name.lower() or "count" in prop_name.lower():
                args[prop_name] = 3
            else:
                args[prop_name] = 1
        elif prop_type == "boolean":
            args[prop_name] = True
        elif prop_type == "array":
            args[prop_name] = []

    return args


async def test_server(server: dict, sem: asyncio.Semaphore) -> dict:
    """Test a single MCP server: connect, list tools, probe one tool."""
    async with sem:
        server_id = server["id"]
        result = {
            "id": server_id,
            "name": server["name"],
            "command": server["command"],
            "args": server["args"],
            "description": server["description"],
            "category": server["category"],
            "web": server["web"],
            "connect": False,
            "tools": [],
            "probe_tool": None,
            "probe_result": None,
            "probe_error": None,
            "time_connect_s": 0,
            "time_probe_s": 0,
        }

        conn = MCPServerConnection(server_id, server["command"], server["args"], server.get("env"))

        # Connect
        t0 = time.time()
        log.info(f"[{server_id}] Connecting: {server['command']} {' '.join(server['args'])}")
        ok = await conn.connect()
        result["time_connect_s"] = round(time.time() - t0, 1)

        if not ok:
            log.warning(f"[{server_id}] FAILED to connect ({result['time_connect_s']}s)")
            result["probe_error"] = "Connection failed"
            return result

        result["connect"] = True
        log.info(f"[{server_id}] Connected in {result['time_connect_s']}s")

        # List tools
        tools = await conn.list_tools()
        result["tools"] = [
            {"name": t["name"], "description": t.get("description", "")[:120],
             "input_schema": t.get("input_schema", {})}
            for t in tools
        ]
        log.info(f"[{server_id}] {len(tools)} tools: {[t['name'] for t in tools[:5]]}")

        if not tools:
            log.warning(f"[{server_id}] No tools available!")
            await conn.close()
            return result

        # Pick first tool with required args and probe it
        target_tool = tools[0]
        probe_tool_name = target_tool["name"]
        probe_args = guess_probe_args(target_tool)
        result["probe_tool"] = probe_tool_name

        log.info(f"[{server_id}] Probing: {probe_tool_name}({json.dumps(probe_args)[:80]})")

        t0 = time.time()
        try:
            call_result = await conn.call_tool(probe_tool_name, probe_args)
            result["time_probe_s"] = round(time.time() - t0, 1)

            if call_result.get("error"):
                result["probe_error"] = call_result["error"]
                log.warning(f"[{server_id}] PROBE FAILED: {call_result['error'][:100]}")
            else:
                content = call_result.get("content") or ""
                result["probe_result"] = content[:500]
                is_error = call_result.get("is_error", False)
                if is_error:
                    result["probe_error"] = f"isError: {content[:200]}"
                    log.warning(f"[{server_id}] PROBE isError: {content[:100]}")
                else:
                    log.info(f"[{server_id}] PROBE OK ({result['time_probe_s']}s): {content[:100]}")
        except Exception as e:
            result["time_probe_s"] = round(time.time() - t0, 1)
            result["probe_error"] = str(e)
            log.warning(f"[{server_id}] PROBE EXCEPTION: {e}")

        await conn.close()
        return result


async def main():
    servers = load_livemcpbench_servers()
    print(f"Testing {len(servers)} LiveMCPBench servers (max {MAX_CONCURRENT} concurrent)\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [test_server(s, sem) for s in servers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.error(f"[{servers[i]['id']}] EXCEPTION: {r}")
            final_results.append({
                "id": servers[i]["id"],
                "name": servers[i]["name"],
                "connect": False,
                "probe_error": str(r),
            })
        else:
            final_results.append(r)

    # Summary
    ok_connect = [r for r in final_results if r.get("connect")]
    ok_probe = [r for r in final_results if r.get("probe_result") and not r.get("probe_error")]
    fail = [r for r in final_results if not r.get("connect")]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(ok_probe)} fully working / {len(ok_connect)} connected / {len(final_results)} total")
    print(f"{'='*80}")
    for r in sorted(final_results, key=lambda x: (not x.get("connect"), not x.get("probe_result"), x.get("id", ""))):
        connected = r.get("connect", False)
        has_probe = bool(r.get("probe_result")) and not r.get("probe_error")
        tools_n = len(r.get("tools", []))
        status = "OK  " if has_probe else ("CONN" if connected else "FAIL")
        err = (r.get("probe_error") or "")[:60]
        t_conn = r.get("time_connect_s", 0)
        t_probe = r.get("time_probe_s", 0)
        print(f"  [{status}] {r['id']:30s}  tools={tools_n:2d}  conn={t_conn:5.1f}s  probe={t_probe:5.1f}s  {err}")

    # Save full results
    out = ROOT / "data" / "pool" / "livemcpbench_test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Save passing servers in pool-compatible format
    passing = []
    for r in final_results:
        if r.get("connect") and r.get("tools"):
            passing.append({
                "id": f"livemcpbench/{r['id']}",
                "name": r.get("name", r["id"]),
                "command": r.get("command", ""),
                "args": r.get("args", []),
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "source": "livemcpbench",
                "status": "probe_ok" if (r.get("probe_result") and not r.get("probe_error")) else "connect_only",
                "tools": r.get("tools", []),
                "tool_count": len(r.get("tools", [])),
            })

    pool_out = ROOT / "data" / "pool" / "livemcpbench_verified.json"
    with open(pool_out, "w") as f:
        json.dump(passing, f, indent=2)
    print(f"Pool-compatible: {len(passing)} servers saved to {pool_out}")


if __name__ == "__main__":
    asyncio.run(main())

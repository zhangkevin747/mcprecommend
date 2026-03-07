"""Test-install and probe 5 MCP servers to verify they work."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp2.mcp_client import MCPServerConnection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# 5 servers to test, with their install configs
TEST_SERVERS = [
    {
        "id": "ddg_search",
        "name": "DuckDuckGo Search",
        "command": "npx",
        "args": ["-y", "@oevortex/ddg_search@latest"],
        "probe_tool": "ddg_search",
        "probe_args": {"query": "weather today", "maxResults": 3},
    },
    {
        "id": "mcp-crypto-price",
        "name": "Crypto Price",
        "command": "npx",
        "args": ["-y", "mcp-crypto-price"],
        "probe_tool": "get-crypto-price",
        "probe_args": {"symbol": "BTC"},
    },
    {
        "id": "frankfurtermcp",
        "name": "Frankfurter Currency Exchange (BROKEN - FastMCP API mismatch)",
        "command": "uvx",
        "args": ["frankfurtermcp"],
        "probe_tool": None,
        "probe_args": None,
    },
    {
        "id": "mcp-server-airbnb",
        "name": "Airbnb Search",
        "command": "npx",
        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        "probe_tool": "airbnb_search",
        "probe_args": {"location": "New York", "checkin": "2025-06-01", "checkout": "2025-06-05"},
    },
    {
        "id": "playwright-mcp",
        "name": "Playwright Browser",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "probe_tool": None,
        "probe_args": None,
    },
]


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
            # Guess based on property name
            name_lower = prop_name.lower()
            if "query" in name_lower or "search" in name_lower or "keyword" in name_lower:
                args[prop_name] = "bitcoin price"
            elif "url" in name_lower:
                args[prop_name] = "https://example.com"
            elif "currency" in name_lower or "from" in name_lower:
                args[prop_name] = "USD"
            elif "to" in name_lower:
                args[prop_name] = "EUR"
            elif "location" in name_lower or "city" in name_lower or "place" in name_lower:
                args[prop_name] = "New York"
            elif "date" in name_lower:
                args[prop_name] = "2025-01-01"
            elif "symbol" in name_lower or "ticker" in name_lower or "coin" in name_lower:
                args[prop_name] = "bitcoin"
            elif "id" in name_lower:
                args[prop_name] = "1"
            else:
                args[prop_name] = "test"
        elif prop_type == "number" or prop_type == "integer":
            if "limit" in prop_name.lower() or "max" in prop_name.lower() or "count" in prop_name.lower():
                args[prop_name] = 3
            else:
                args[prop_name] = 1
        elif prop_type == "boolean":
            args[prop_name] = True
        elif prop_type == "array":
            args[prop_name] = []

    return args


async def test_server(server_config: dict) -> dict:
    """Test a single MCP server: connect, list tools, call one tool."""
    server_id = server_config["id"]
    result = {
        "id": server_id,
        "name": server_config["name"],
        "command": f"{server_config['command']} {' '.join(server_config['args'])}",
        "connect": False,
        "tools": [],
        "probe_tool": None,
        "probe_result": None,
        "probe_error": None,
        "time_connect_s": 0,
        "time_probe_s": 0,
    }

    conn = MCPServerConnection(
        server_id,
        server_config["command"],
        server_config["args"],
    )

    # Connect
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{server_id}] Connecting: {server_config['command']} {' '.join(server_config['args'])}")
    ok = await conn.connect()
    result["time_connect_s"] = round(time.time() - t0, 1)

    if not ok:
        print(f"[{server_id}] FAILED to connect ({result['time_connect_s']}s)")
        result["probe_error"] = "Connection failed"
        return result

    result["connect"] = True
    print(f"[{server_id}] Connected in {result['time_connect_s']}s")

    # List tools
    tools = await conn.list_tools()
    result["tools"] = [{"name": t["name"], "description": t["description"][:80]} for t in tools]
    print(f"[{server_id}] {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['name']}: {t['description'][:70]}")

    if not tools:
        print(f"[{server_id}] No tools available!")
        await conn.close()
        return result

    # Pick a tool to probe
    probe_tool_name = server_config.get("probe_tool")
    probe_args = server_config.get("probe_args")

    if probe_tool_name and probe_args:
        # Use explicit probe config
        target_tool = next((t for t in tools if t["name"] == probe_tool_name), None)
        if not target_tool:
            # Fall back to first tool
            target_tool = tools[0]
            probe_tool_name = target_tool["name"]
            probe_args = guess_probe_args(target_tool)
    else:
        # Pick first tool and guess args
        target_tool = tools[0]
        probe_tool_name = target_tool["name"]
        probe_args = guess_probe_args(target_tool)

    result["probe_tool"] = probe_tool_name
    print(f"[{server_id}] Probing: {probe_tool_name}({json.dumps(probe_args)[:100]})")

    t0 = time.time()
    call_result = await conn.call_tool(probe_tool_name, probe_args)
    result["time_probe_s"] = round(time.time() - t0, 1)

    if call_result["error"]:
        result["probe_error"] = call_result["error"]
        print(f"[{server_id}] PROBE FAILED: {call_result['error']}")
    else:
        content = call_result["content"] or ""
        result["probe_result"] = content[:500]
        is_error = call_result.get("is_error", False)
        if is_error:
            print(f"[{server_id}] PROBE returned isError=True: {content[:200]}")
            result["probe_error"] = f"isError: {content[:200]}"
        else:
            print(f"[{server_id}] PROBE OK ({result['time_probe_s']}s): {content[:200]}")

    await conn.close()
    return result


async def main():
    results = []
    for server_config in TEST_SERVERS:
        try:
            r = await test_server(server_config)
            results.append(r)
        except Exception as e:
            print(f"[{server_config['id']}] EXCEPTION: {e}")
            results.append({
                "id": server_config["id"],
                "name": server_config["name"],
                "connect": False,
                "probe_error": str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["connect"] and r.get("probe_result") and not r.get("probe_error") else "FAIL"
        tools_n = len(r.get("tools", []))
        err = r.get("probe_error") or ""
        print(f"  [{status:4s}] {r['id']:25s}  tools={tools_n:2d}  connect={r.get('time_connect_s',0):.1f}s  probe={r.get('time_probe_s',0):.1f}s  {err[:50]}")

    # Save results
    out = ROOT / "data" / "pool" / "test_install_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())

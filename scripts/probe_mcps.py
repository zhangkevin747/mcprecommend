"""Probe all 431 MCPs with gpt-5-nano to verify actual tool execution.

For each server:
  1. Connect (stdio or Smithery)
  2. List tools
  3. One gpt-5-nano turn: pick + call a tool
  4. Record: mount_ok, tool_count, tool_called, response_len, error, latency

Usage:
    python scripts/probe_mcps.py [--concurrency 32] [--output results/probe_results.jsonl]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.exp2.mcp_client import MCPServerConnection, SmitheryConnection, get_smithery_mcp_url

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

PROBE_MODEL = "gpt-5-nano"
MCP_CONNECT_TIMEOUT = 30
MCP_CALL_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are probing an MCP server to verify its tools work. "
    "Always call a tool — never refuse. "
    "If a tool needs a URL, use 'https://example.com'. "
    "If it needs a search query, use 'hello world'. "
    "If it needs a city, use 'New York'. "
    "If it needs a date, use '2025-01-01'. "
    "If it needs a stock symbol, use 'AAPL'. "
    "If it needs a library name, use 'react'. "
    "Generate any required inputs yourself — never leave required fields empty. "
    "If one tool errors or returns nothing, try a different tool. "
    "Keep trying until you get a response or exhaust all options."
)


def _is_mcp_remote(server: dict) -> bool:
    return any("mcp-remote" in str(a) for a in server.get("args", []))


async def _connect(server: dict):
    """Connect to server. Returns (conn, tools) or (None, [])."""
    server_id = server["id"]
    smithery_url = get_smithery_mcp_url(server)

    if _is_mcp_remote(server) and smithery_url:
        conn = SmitheryConnection(server_id, smithery_url)
        ok = await conn.connect()
        if not ok:
            await conn.close()
            return None, []
        tools = await conn.list_tools()
        if not tools:
            await conn.close()
            return None, []
        return conn, tools

    command = server.get("command")
    args = server.get("args", [])
    if not command:
        return None, []

    conn = MCPServerConnection(server_id, command, args, env=server.get("env"))
    ok = await conn.connect()
    if not ok:
        return None, []
    tools = await conn.list_tools()
    if not tools:
        await conn.close()
        return None, []
    return conn, tools


def _clean_schema(schema: dict) -> dict:
    """Recursively normalize schema for OpenAI strict mode.

    Strict mode rules (applied at every level):
    - Objects: type=object, properties={}, additionalProperties=false, all props in required
    - Arrays: must have items
    - No oneOf/anyOf/allOf/not/$ref (strip them)
    - No null type (replace with string)
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "string"}

    # Strip keywords strict mode doesn't support
    STRIP = {"$schema", "oneOf", "anyOf", "allOf", "not", "$ref", "if", "then", "else",
             "additionalProperties"}
    result = {k: v for k, v in schema.items() if k not in STRIP}

    # Strip unsupported format values (only date-time, date, time, duration, email,
    # hostname, ipv4, ipv6, uuid are allowed; uri/string/etc. are not)
    ALLOWED_FORMATS = {"date-time", "date", "time", "duration", "email",
                       "hostname", "ipv4", "ipv6", "uuid"}
    if "format" in result and result["format"] not in ALLOWED_FORMATS:
        del result["format"]

    # Normalize type — handle array types like ["string", "null"]
    t = result.get("type")
    if isinstance(t, list):
        # Pick first non-null type, or string as fallback
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else "string"
        result["type"] = t
    if t is None or t == "null":
        result["type"] = "string"
        t = "string"

    if t == "object" or "properties" in result:
        result["type"] = "object"
        result.setdefault("properties", {})
        result["additionalProperties"] = False
        # Recursively clean all property schemas
        result["properties"] = {
            k: _clean_schema(v) for k, v in result["properties"].items()
        }
        # All properties must appear in required
        all_props = list(result["properties"].keys())
        existing = result.get("required") or []
        result["required"] = list(existing) + [p for p in all_props if p not in existing]

    elif t == "array":
        if "items" not in result:
            result["items"] = {"type": "string"}
        else:
            result["items"] = _clean_schema(result["items"])

    return result


def _build_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert MCP tool schemas to OpenAI function format with strict mode."""
    result = []
    seen: dict[str, int] = {}
    for t in tools:
        base = t["name"][:60]
        if base in seen:
            seen[base] += 1
            name = f"{base[:57]}_{seen[base]}"
        else:
            seen[base] = 0
            name = base
        params = _clean_schema(t.get("input_schema") or {})
        # OpenAI requires top-level parameters to be type=object
        if params.get("type") != "object":
            params = {"type": "object", "properties": {"value": params},
                      "required": ["value"], "additionalProperties": False}
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (t.get("description") or "")[:200],
                "parameters": params,
                "strict": True,
            },
        })
    return result


async def _probe_one(server: dict, client: openai.AsyncOpenAI, semaphore: asyncio.Semaphore) -> dict:
    """Probe a single server. Returns result dict."""
    server_id = server["id"]
    t0 = time.time()
    result = {
        "id": server_id,
        "status_before": server.get("status", ""),
        "mount_ok": False,
        "tool_count": 0,
        "tool_called": None,
        "response_len": 0,
        "got_response": False,
        "got_error": False,
        "error_type": None,
        "error_msg": None,
        "latency_s": 0.0,
        "probe_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    async with semaphore:
        conn = None
        try:
            conn, tools = await asyncio.wait_for(_connect(server), timeout=MCP_CONNECT_TIMEOUT + 5)
        except asyncio.TimeoutError:
            result["error_type"] = "mount_timeout"
            result["error_msg"] = "connect timeout"
            result["latency_s"] = round(time.time() - t0, 2)
            return result
        except Exception as e:
            result["error_type"] = "mount_error"
            result["error_msg"] = str(e)[:200]
            result["latency_s"] = round(time.time() - t0, 2)
            return result

        if conn is None or not tools:
            result["error_type"] = "mount_failed" if conn is None else "no_tools"
            result["error_msg"] = "server did not mount or returned no tools"
            result["latency_s"] = round(time.time() - t0, 2)
            if conn:
                await conn.close()
            return result

        result["mount_ok"] = True
        result["tool_count"] = len(tools)

        # Build tool name → MCP name lookup (OpenAI truncates to 60 chars)
        openai_tools = _build_openai_tools(tools)[:20]
        api_to_mcp = {}
        for ot, t in zip(openai_tools, tools[:20]):
            api_to_mcp[ot["function"]["name"]] = t["name"]

        desc = server.get("description", "")[:300]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Server: {server_id}\nDescription: {desc}\n\n"
                    "Call one of the available tools to demonstrate it works. "
                    "Use simple, plausible example arguments."
                ),
            },
        ]

        try:
            # Multi-turn FC loop: up to 3 shots
            for shot in range(3):
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=PROBE_MODEL,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="required",
                        max_completion_tokens=2048,
                    ),
                    timeout=30,
                )
                choice = resp.choices[0]

                # No tool call — model gave up
                if not choice.message.tool_calls:
                    result["error_type"] = "llm_no_tool_call"
                    result["error_msg"] = choice.message.content or "no tool call made"
                    break

                # Append assistant message with tool calls
                messages.append(choice.message.model_dump())

                # Execute each tool call and collect results
                any_success = False
                tool_messages = []
                for tc in choice.message.tool_calls:
                    api_name = tc.function.name
                    mcp_tool_name = api_to_mcp.get(api_name, api_name)
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception:
                        args = {}

                    result["tool_called"] = mcp_tool_name

                    try:
                        tool_result = await asyncio.wait_for(
                            conn.call_tool(mcp_tool_name, args),
                            timeout=MCP_CALL_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        tool_result = {"content": None, "error": "timeout"}
                    except Exception as e:
                        tool_result = {"content": None, "error": str(e)[:200]}

                    if tool_result.get("error"):
                        tool_content = f"Error: {tool_result['error']}"
                        result["error_type"] = "api_error"
                        result["error_msg"] = str(tool_result["error"])[:300]
                    else:
                        content = tool_result.get("content") or ""
                        tool_content = content[:500]
                        if not tool_result.get("is_error") and content:
                            result["got_response"] = True
                            result["got_error"] = False
                            result["response_len"] = len(content)
                            result["response_preview"] = content[:300]
                            result["error_type"] = None
                            result["error_msg"] = None
                            any_success = True
                        elif tool_result.get("is_error"):
                            result["error_type"] = "tool_error"
                            result["error_msg"] = content[:300]

                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

                messages.extend(tool_messages)

                if any_success:
                    break

        except asyncio.TimeoutError:
            result["error_type"] = "llm_timeout"
            result["error_msg"] = "LLM call timed out"
        except Exception as e:
            result["error_type"] = "llm_error"
            result["error_msg"] = str(e)[:200]

        finally:
            if conn:
                try:
                    await conn.close()
                except Exception:
                    pass

    result["latency_s"] = round(time.time() - t0, 2)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/pool/combined_pool.json")
    parser.add_argument("--output", default="results/probe_results.jsonl")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already probed
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    remaining = [s for s in pool if s["id"] not in done_ids]
    print(f"Pool: {len(pool)} | Already done: {len(done_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All servers already probed.")
        return

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [_probe_one(s, client, semaphore) for s in remaining]

    completed = 0
    ok_count = 0
    error_counts: dict[str, int] = {}

    with open(out_path, "a") as f:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            f.write(json.dumps(r) + "\n")
            f.flush()
            completed += 1

            if r.get("got_response") and not r.get("got_error"):
                ok_count += 1
            elif r.get("error_type"):
                ec = r["error_type"]
                error_counts[ec] = error_counts.get(ec, 0) + 1

            if completed % 20 == 0 or completed == len(remaining):
                pct = completed / len(remaining) * 100
                print(
                    f"[{completed}/{len(remaining)} {pct:.0f}%] "
                    f"ok={ok_count} errors={dict(sorted(error_counts.items(), key=lambda x: -x[1]))}"
                )

    await client.close()
    print(f"\nDone. Results: {out_path}")
    print(f"OK (got response, no error): {ok_count}/{len(remaining)}")
    print("Error breakdown:", dict(sorted(error_counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    asyncio.run(main())

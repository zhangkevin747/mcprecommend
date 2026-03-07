"""Oracle: for each task, find a working MCP from the verified pool.

For each task:
  1. Semantic-retrieve top-K candidates from verified_pool.json
  2. GPT-5-nano picks a tool and calls it (up to 3 server attempts)
  3. Record: solvable, server_used, tool_called, response_preview, latency

Output: results/oracle_results.jsonl
  - solvable=True  → task has at least one working server in the verified pool
  - solvable=False → no server could answer the task (task quality issue or coverage gap)

Usage:
    python scripts/oracle_tasks.py [--concurrency 32] [--k 10]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.exp2.mcp_client import MCPServerConnection, SmitheryConnection, get_smithery_mcp_url
from src.exp2.retriever import precompute_pool_embeddings, precompute_query_embeddings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

PROBE_MODEL = "gpt-5-nano"
MCP_CONNECT_TIMEOUT = 30
MCP_CALL_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are testing whether an MCP tool can help answer a user's task. "
    "Always call a tool — never refuse. "
    "Extract the required inputs directly from the task — use the specific company, stock ticker, location, URL, or keyword mentioned. "
    "If the task mentions Microsoft, use MSFT. If it mentions Apple, use AAPL. If it mentions a city, use that city. "
    "If no specific input is mentioned, use a reasonable example: URL='https://example.com', query='hello world', city='New York', date='2025-01-01'. "
    "Call the most relevant tool for the task."
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a tool response actually answers a user's task. "
    "Reply with exactly one word: YES or NO. "
    "YES means the response contains information that genuinely helps answer the task. "
    "NO means the response is irrelevant, empty, an error, or about a completely different topic."
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
    if not isinstance(schema, dict) or not schema:
        return {"type": "string"}
    STRIP = {"$schema", "oneOf", "anyOf", "allOf", "not", "$ref", "if", "then", "else", "additionalProperties"}
    result = {k: v for k, v in schema.items() if k not in STRIP}
    ALLOWED_FORMATS = {"date-time", "date", "time", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
    if "format" in result and result["format"] not in ALLOWED_FORMATS:
        del result["format"]
    t = result.get("type")
    if isinstance(t, list):
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
        result["properties"] = {k: _clean_schema(v) for k, v in result["properties"].items()}
        all_props = list(result["properties"].keys())
        existing = result.get("required") or []
        result["required"] = list(existing) + [p for p in all_props if p not in existing]
    elif t == "array":
        result["items"] = _clean_schema(result.get("items", {})) if result.get("items") else {"type": "string"}
    return result


def _build_openai_tools(tools: list[dict]) -> list[dict]:
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


async def _judge_relevance(
    task_query: str,
    tool_called: str,
    response_preview: str,
    client: openai.AsyncOpenAI,
) -> bool:
    """Ask gpt-5-nano if the response actually answers the task."""
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=PROBE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Task: {task_query}\n\n"
                        f"Tool called: {tool_called}\n"
                        f"Tool response:\n{response_preview[:400]}\n\n"
                        "Does this response genuinely help answer the task? YES or NO."
                    )},
                ],
                max_completion_tokens=1024,
            ),
            timeout=30,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False  # Conservative: assume not relevant on error


async def _try_server(
    server: dict,
    task_query: str,
    client: openai.AsyncOpenAI,
) -> dict:
    """Try one server for a task. Returns result dict."""
    server_id = server["id"]
    result = {
        "server_id": server_id,
        "mount_ok": False,
        "tool_called": None,
        "solvable": False,
        "response_preview": None,
        "response_len": 0,
        "error_type": None,
        "error_msg": None,
    }

    try:
        conn, tools = await asyncio.wait_for(_connect(server), timeout=MCP_CONNECT_TIMEOUT + 5)
    except Exception as e:
        result["error_type"] = "mount_error"
        result["error_msg"] = str(e)[:200]
        return result

    if conn is None or not tools:
        result["error_type"] = "mount_failed"
        return result

    result["mount_ok"] = True

    openai_tools = _build_openai_tools(tools)[:20]
    api_to_mcp = {}
    for ot, t in zip(openai_tools, tools[:20]):
        api_to_mcp[ot["function"]["name"]] = t["name"]

    desc = server.get("description", "")[:200]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_query}\n\n"
                f"Server: {server_id}\nDescription: {desc}\n\n"
                "Call the most relevant tool for this task."
            ),
        },
    ]

    try:
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
            if not choice.message.tool_calls:
                result["error_type"] = "no_tool_call"
                break

            messages.append(choice.message.model_dump())

            for tc in choice.message.tool_calls:
                api_name = tc.function.name
                mcp_tool_name = api_to_mcp.get(api_name, api_name)
                result["tool_called"] = mcp_tool_name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}

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
                    result["error_type"] = "tool_error"
                    result["error_msg"] = str(tool_result["error"])[:200]
                elif tool_result.get("is_error"):
                    content = tool_result.get("content") or ""
                    tool_content = content[:500]
                    result["error_type"] = "tool_error"
                    result["error_msg"] = content[:200]
                else:
                    content = tool_result.get("content") or ""
                    tool_content = content[:500]
                    if content:
                        relevant = await _judge_relevance(task_query, mcp_tool_name, content, client)
                        if relevant:
                            result["solvable"] = True
                            result["response_preview"] = content[:300]
                            result["response_len"] = len(content)
                            result["error_type"] = None
                            result["error_msg"] = None
                        else:
                            result["error_type"] = "irrelevant"
                            result["error_msg"] = "response does not answer task"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

            if result["solvable"]:
                break

    except asyncio.TimeoutError:
        result["error_type"] = "llm_timeout"
    except Exception as e:
        result["error_type"] = "llm_error"
        result["error_msg"] = str(e)[:200]
    finally:
        try:
            await conn.close()
        except Exception:
            pass

    return result


async def _oracle_one(
    task: dict,
    candidates: list[dict],
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Find the best working server for a task. Tries candidates in similarity order."""
    task_id = task.get("uuid", "")
    task_query = task.get("query", task.get("question", ""))
    t0 = time.time()

    record = {
        "task_id": task_id,
        "task_query": task_query,
        "category": task.get("category", ""),
        "solvable": False,
        "server_used": None,
        "tool_called": None,
        "response_preview": None,
        "response_len": 0,
        "attempts": [],
        "latency_s": 0.0,
    }

    async with semaphore:
        for server in candidates[:3]:
            attempt = await _try_server(server, task_query, client)
            record["attempts"].append({
                "server_id": attempt["server_id"],
                "mount_ok": attempt["mount_ok"],
                "solvable": attempt["solvable"],
                "tool_called": attempt.get("tool_called"),
                "error_type": attempt.get("error_type"),
            })

            if attempt["solvable"]:
                record["solvable"] = True
                record["server_used"] = attempt["server_id"]
                record["tool_called"] = attempt["tool_called"]
                record["response_preview"] = attempt["response_preview"]
                record["response_len"] = attempt["response_len"]
                break

    record["latency_s"] = round(time.time() - t0, 2)
    return record


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/pool/verified_pool.json")
    parser.add_argument("--tasks", default="data/tasks_combined.json")
    parser.add_argument("--output", default="results/oracle_results.jsonl")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--k", type=int, default=10, help="Candidates per task")
    args = parser.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    tasks = json.loads(Path(args.tasks).read_text())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already processed
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["task_id"])
            except Exception:
                pass

    remaining = [t for t in tasks if t.get("uuid", "") not in done_ids]
    print(f"Tasks: {len(tasks)} | Done: {len(done_ids)} | Remaining: {len(remaining)}")
    print(f"Verified pool: {len(pool)} servers | K={args.k} candidates per task")

    if not remaining:
        print("All tasks already processed.")
        return

    # Precompute pool embeddings once
    print("Precomputing pool embeddings...")
    pool_emb_matrix, pool_entries = precompute_pool_embeddings(pool)

    # Precompute all query embeddings in one batch
    print("Precomputing query embeddings...")
    query_embs = precompute_query_embeddings(remaining)

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(args.concurrency)

    # Build per-task candidate lists using fast retrieval
    from src.exp2.retriever import retrieve_from_pool_fast

    task_candidates = []
    for task in remaining:
        tid = task.get("uuid", "")
        qemb = query_embs.get(tid)
        if qemb is not None:
            candidates = retrieve_from_pool_fast(qemb, pool_emb_matrix, pool_entries, top_n=args.k)
        else:
            candidates = pool[:args.k]
        task_candidates.append((task, candidates))

    coros = [_oracle_one(task, cands, client, semaphore) for task, cands in task_candidates]

    completed = 0
    solvable_count = 0

    with open(out_path, "a") as f:
        for coro in asyncio.as_completed(coros):
            r = await coro
            f.write(json.dumps(r) + "\n")
            f.flush()
            completed += 1
            if r["solvable"]:
                solvable_count += 1

            if completed % 50 == 0 or completed == len(remaining):
                pct = completed / len(remaining) * 100
                spct = solvable_count / completed * 100
                print(f"[{completed}/{len(remaining)} {pct:.0f}%] solvable={solvable_count} ({spct:.0f}%)")

    await client.close()
    print(f"\nDone. Results: {out_path}")
    print(f"Solvable: {solvable_count}/{len(remaining)} ({100*solvable_count/len(remaining):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())

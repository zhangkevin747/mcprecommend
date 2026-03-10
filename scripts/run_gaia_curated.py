"""Run 11 GAIA questions with manually curated MCP servers per question."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Q -> best servers from mountable pool
CURATED = {
    # Q1: ScienceDirect reference works stats - needs web search + scrape
    "0b260a57-3f3a-4405-9f29-6d7a1012dbfb": [
        "pulse/tavily-search", "exa", "pulse/scrapi", "pulse/youcom", "OEvortex/ddg_search",
    ],
    # Q2: Japanese baseball pitcher numbers - needs web search
    "a0c07678-e491-4bbc-8f0b-07405144218f": [
        "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom", "pulse/scrapi",
    ],
    # Q3: USGS invasive fish (Finding Nemo) zip codes - needs web search
    "17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc": [
        "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom", "pulse/scrapi",
    ],
    # Q4: EC numbers from virology paper - needs academic search + web
    "2a649bb1-795f-4a01-b3be-9a01868dae73": [
        "adamamer20/paper-search-mcp-openai", "pulse/tavily-search", "exa",
        "hamid-vakilzadeh/mcpsemanticscholar", "OEvortex/ddg_search",
    ],
    # Q5: "Pie Menus or Linear Menus" author's first paper - academic search
    "46719c30-f4c3-4cad-be07-d5cb21eee6bb": [
        "adamamer20/paper-search-mcp-openai", "pulse/tavily-search", "exa",
        "hamid-vakilzadeh/mcpsemanticscholar", "OEvortex/ddg_search",
    ],
    # Q6: Caesar cipher decryption - needs code execution
    "ded28325-3447-4c56-860f-e497d6fb3577": [
        "STUzhy/py_execute_mcp", "pulse/tavily-search", "exa",
        "EthanHenrickson/math-mcp", "OEvortex/ddg_search",
    ],
    # Q7: Girls Who Code percentage stats - web search
    "7d4a7d1d-cac6-44a8-96e8-ea9584a70825": [
        "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom", "pulse/scrapi",
    ],
    # Q8: Scikit-learn July 2017 changelog - docs + web search
    "d0633230-7067-47a9-9dbf-ee11e0a2cdd6": [
        "pulse/upstash-context7", "pulse/tavily-search", "exa",
        "OEvortex/ddg_search", "pulse/scrapi",
    ],
    # Q9: World Bank gross savings data - web search
    "0a3cd321-3e76-4622-911b-0fda2e5d6b1a": [
        "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom", "pulse/scrapi",
    ],
    # Q10: Wayback Machine restaurant menu comparison - web search + extract
    "e8cb5b03-41e0-4086-99e5-f6806cd97211": [
        "pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom",
    ],
    # Q11: YouTube GameGrumps video + Mario Kart record - YouTube + web
    "7a4a336d-dcfa-45a0-b014-824c7619e8de": [
        "sfiorini/youtube-mcp", "pulse/tavily-search", "exa",
        "OEvortex/ddg_search", "pulse/youcom",
    ],
}


async def run_one(task: dict, servers_to_mount: list[str], pool_by_id: dict, schema_cache: dict):
    """Run one GAIA task with curated servers."""
    from src.recsys.agent_client import run_agent
    from src.recsys.feedback import collect_feedback
    from src.recsys.pipeline import _try_connect
    from src.recsys.mcp_client import MCPServerConnection, SmitheryConnection

    task_id = task["uuid"]
    task_query = task["query"]
    t0 = time.time()

    connections = {}
    all_tools = []
    servers_mounted = []
    servers_failed = []
    server_by_id = {}

    for sid in servers_to_mount:
        server = pool_by_id.get(sid)
        if not server:
            logging.warning(f"Server {sid} not found in pool")
            servers_failed.append(sid)
            continue
        server_by_id[sid] = server

        cached = schema_cache.get(sid)
        if cached and cached.get("mountable") and cached.get("tools"):
            all_tools.extend(cached["tools"])
            servers_mounted.append(sid)
        else:
            servers_failed.append(sid)
            logging.warning(f"Server {sid} not mountable in schema cache")

    async def lazy_connect(server_id):
        if server_id in connections:
            return connections[server_id]
        server = server_by_id.get(server_id)
        if not server:
            return None
        logging.info(f"Lazy-connecting to {server_id}...")
        try:
            conn, tools = await _try_connect(server)
        except Exception as e:
            logging.warning(f"Lazy connect failed for {server_id}: {e}")
            conn, tools = None, []
        if tools:
            connections[server_id] = conn
            return conn
        if conn:
            await conn.close()
        if server_id in servers_mounted:
            servers_mounted.remove(server_id)
        if server_id not in servers_failed:
            servers_failed.append(server_id)
        return None

    logging.info(f"[{task_id[:8]}] {len(servers_mounted)} servers mounted, {len(all_tools)} tools")

    if not all_tools:
        return {"task_id": task_id, "task_query": task_query, "error": "No tools", "agent_answer": ""}

    agent_name = "deepseek-v3.2"
    try:
        agent_result = await run_agent(agent_name, task_query, all_tools, connections, lazy_connect_fn=lazy_connect)
    except Exception as e:
        logging.error(f"Agent failed: {e}")
        agent_result = {"answer": f"Error: {e}", "tools_selected": [], "tools_results": [],
                        "tools_abandoned": [], "tools_errored": [], "tool_used_final": None,
                        "input_tokens": 0, "output_tokens": 0}

    tools_offered = [f"{t['server_id']}:{t['name']}" for t in all_tools]
    tools_used = agent_result.get("tools_selected", [])
    try:
        feedback = await collect_feedback(agent_name, task_query, agent_result["answer"], tools_offered, tools_used)
    except Exception as e:
        feedback = {}

    for conn in connections.values():
        await conn.close()

    latency_s = round(time.time() - t0, 2)

    return {
        "task_id": task_id,
        "task_query": task_query,
        "task_category": task.get("category", ""),
        "final_answer_gt": task.get("final_answer", ""),
        "level": task.get("level", ""),
        "agent": agent_name,
        "inventory_mounted": servers_mounted,
        "inventory_failed": servers_failed,
        "tools_available": [f"{t['server_id']}:{t['name']}" for t in all_tools],
        "tools_selected": agent_result.get("tools_selected", []),
        "tools_errored": agent_result.get("tools_errored", []),
        "tools_abandoned": agent_result.get("tools_abandoned", []),
        "tool_used_final": agent_result.get("tool_used_final"),
        "agent_answer": agent_result.get("answer", ""),
        "input_tokens": agent_result.get("input_tokens", 0),
        "output_tokens": agent_result.get("output_tokens", 0),
        "feedback": feedback,
        "latency_s": latency_s,
        "recommender": "curated_oracle",
    }


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load pool
    with open(ROOT / "data" / "mountable_pool.json") as f:
        pool = json.load(f)
    pool_by_id = {s["id"]: s for s in pool}

    # Load schema cache
    cache_path = ROOT / "data" / "pool_schema_cache.json"
    schema_cache = {}
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        schema_cache = data.get("servers", {})

    # Load GAIA tasks
    with open(ROOT / "data" / "tasks_gaia.json") as f:
        all_tasks = json.load(f)
    tasks_by_id = {t["uuid"]: t for t in all_tasks}

    output_path = ROOT / "results" / "gaia_deepseek" / "curated_oracle.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check which are already done
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(r.get("task_id", ""))

    for task_id, servers in CURATED.items():
        if task_id in done:
            logging.info(f"Skipping {task_id[:8]} (already done)")
            continue

        task = tasks_by_id.get(task_id)
        if not task:
            logging.warning(f"Task {task_id} not found")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Q: {task['query'][:100]}")
        logging.info(f"Curated servers: {servers}")
        logging.info(f"Ground truth: {task.get('final_answer', 'N/A')}")

        result = await run_one(task, servers, pool_by_id, schema_cache)

        with open(output_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        logging.info(f"Answer: {result.get('agent_answer', '')[:200]}")
        logging.info(f"GT:     {task.get('final_answer', '')}")
        logging.info(f"Latency: {result.get('latency_s', 0)}s")

    logging.info("\nDone! Results at: %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())

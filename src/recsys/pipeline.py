"""Rollout orchestrator: runs one complete recommendation → execution → feedback cycle."""

import asyncio
import logging
import time

from .agent_client import run_agent
from .feedback import collect_feedback
from .mcp_client import (
    FALLBACK_SERVERS,
    MCPServerConnection,
    SmitheryConnection,
    derive_npx_command,
    get_smithery_mcp_url,
)
from .recommenders.base import BaseRecommender
from .retriever import retrieve, retrieve_from_pool, retrieve_from_pool_fast

log = logging.getLogger(__name__)


def _is_mcp_remote(server: dict) -> bool:
    """Check if server uses mcp-remote (Smithery OAuth — can't connect headlessly)."""
    args = server.get("args", [])
    return any("mcp-remote" in str(a) for a in args)


async def _try_connect(server: dict) -> tuple[MCPServerConnection | SmitheryConnection | None, dict]:
    """Try to connect to an MCP server. Returns (connection, tool_list) or (None, []).

    For Smithery servers: uses Smithery Connect REST API (no OAuth, no browser popups).
    For npx servers: spins up via stdio as before.
    """
    server_id = server["id"]

    # Smithery servers: connect via Smithery Connect REST API
    smithery_url = get_smithery_mcp_url(server)
    if _is_mcp_remote(server) and smithery_url:
        conn = SmitheryConnection(server_id, smithery_url)
        ok = await conn.connect()
        if not ok:
            await conn.close()
            return None, {}
        tools = await conn.list_tools()
        if not tools:
            log.info(f"{server_id}: Smithery connected but no tools")
            await conn.close()
            return None, {}
        return conn, tools

    # Use explicit command/args if provided (for fallback servers)
    if "command" in server:
        command, args = server["command"], server["args"]
        env = server.get("env")
    else:
        result = derive_npx_command(server)
        if not result:
            log.debug(f"Cannot derive npx command for {server_id}")
            return None, {}
        command, args, env = result

    conn = MCPServerConnection(server_id, command, args, env=env)
    ok = await conn.connect()
    if not ok:
        return None, {}

    tools = await conn.list_tools()
    if not tools:
        log.info(f"{server_id}: connected but no tools")
        await conn.close()
        return None, {}

    return conn, tools


async def run_rollout(
    rollout_id: int,
    task_query: str,
    task_id: str,
    agent_name: str,
    k: int,
    recommender: BaseRecommender,
    retrieve_n: int = 100,
    use_fallbacks: bool = True,
    pool: list[dict] | None = None,
    pool_emb_matrix=None,
    pool_entries: list[dict] | None = None,
    query_emb=None,
    task_category: str = "",
) -> dict:
    """Execute one complete rollout.

    1. Retrieve top N candidates via semantic search (pool-restricted if pool given)
    2. Recommender selects top K
    3. Spin up K MCP servers (parallel)
    4. Agent executes task with available tools
    5. Collect feedback
    6. Return rollout log
    """
    t0 = time.time()

    # Stage 1: Retrieve candidates
    if pool_emb_matrix is not None and query_emb is not None:
        log.info(f"[Rollout {rollout_id}] Fast retrieval (precomputed embeddings)...")
        candidates = retrieve_from_pool_fast(query_emb, pool_emb_matrix, pool_entries, top_n=retrieve_n)
    elif pool:
        log.info(f"[Rollout {rollout_id}] Retrieving from pool of {len(pool)} servers...")
        candidates = retrieve_from_pool(task_query, pool, top_n=retrieve_n)
    else:
        log.info(f"[Rollout {rollout_id}] Retrieving top {retrieve_n} candidates from full index...")
        candidates = retrieve(task_query, top_n=retrieve_n)

    # Stage 2: Rerank / select top K + buffer for fallback substitution.
    # Each method ranks all candidates; we try k + buffer in parallel, then
    # take the first k that successfully mount (in ranked order).
    # This ensures all methods get k working servers when available, so mount
    # failures don't unfairly advantage methods that learned to avoid broken servers.
    rerank_method = recommender.method_name
    buffer = k  # try up to 2k; each method uses its own ordering for the buffer
    n_to_try = min(k + buffer, len(candidates))
    ranked = recommender.recommend(agent_name, task_query, candidates, n_to_try, task_category=task_category, task_emb=query_emb)
    selected = ranked[:k]  # what the recommender chose as top-k (for logging)
    log.info(f"[Rollout {rollout_id}] Selected {len(selected)} servers (+ {len(ranked)-len(selected)} buffer) ({rerank_method})")

    # Add fallback servers if requested (eval disables this)
    servers_to_try = list(ranked)
    if use_fallbacks:
        for fb in FALLBACK_SERVERS:
            if not any(s["id"] == fb["id"] for s in servers_to_try):
                servers_to_try.append(fb)

    # Stage 3: Spin up servers in parallel
    log.info(f"[Rollout {rollout_id}] Connecting to {len(servers_to_try)} servers...")
    connect_tasks = [_try_connect(s) for s in servers_to_try]
    results = await asyncio.gather(*connect_tasks, return_exceptions=True)

    connections: dict[str, MCPServerConnection | SmitheryConnection] = {}
    all_tools: list[dict] = []
    servers_mounted = []
    servers_failed = []

    for server, result in zip(servers_to_try, results):
        if isinstance(result, Exception):
            log.warning(f"Exception connecting to {server['id']}: {result}")
            servers_failed.append(server["id"])
            continue
        conn, tools = result
        if not tools:
            servers_failed.append(server["id"])
            continue
        if len(servers_mounted) < k:
            # Accept this server (top-k or fallback substitute)
            if conn:
                connections[server["id"]] = conn
            all_tools.extend(tools)
            servers_mounted.append(server["id"])
        else:
            # Already have k mounted — close excess connection, don't count as failed
            if conn:
                await conn.close()

    log.info(f"[Rollout {rollout_id}] {len(servers_mounted)} servers up, {len(servers_failed)} failed, {len(all_tools)} tools available")

    pool_size = len(pool) if pool else 21134

    if not all_tools:
        log.error(f"[Rollout {rollout_id}] No tools available! All servers failed.")
        # Clean up
        for conn in connections.values():
            await conn.close()
        return _build_rollout_log(
            rollout_id, agent_name, task_id, task_query,
            pool_size, retrieve_n, len(candidates), k, selected, servers_mounted, servers_failed,
            all_tools=[], agent_result=None, feedback={},
            latency_s=round(time.time() - t0, 2), error="No tools available",
            rerank_method=rerank_method, task_category=task_category,
        )

    # Stage 4: Agent executes
    log.info(f"[Rollout {rollout_id}] Running agent {agent_name}...")
    try:
        agent_result = await run_agent(agent_name, task_query, all_tools, connections)
    except Exception as e:
        log.error(f"[Rollout {rollout_id}] Agent failed: {e}")
        agent_result = {"answer": f"Error: {e}", "tools_selected": [], "tools_results": [],
                        "tools_abandoned": [], "tools_errored": [], "tool_used_final": None,
                        "input_tokens": 0, "output_tokens": 0, "latency_s": 0}

    # Stage 5: Collect feedback
    log.info(f"[Rollout {rollout_id}] Collecting feedback...")
    tools_offered = [f"{t['server_id']}:{t['name']}" for t in all_tools]
    tools_used = agent_result.get("tools_selected", [])
    try:
        feedback = await collect_feedback(
            agent_name, task_query, agent_result["answer"],
            tools_offered, tools_used,
        )
    except Exception as e:
        log.warning(f"[Rollout {rollout_id}] Feedback collection failed: {e}")
        feedback = {}

    # Clean up all connections
    for conn in connections.values():
        await conn.close()

    latency_s = round(time.time() - t0, 2)
    log.info(f"[Rollout {rollout_id}] Done in {latency_s}s")

    return _build_rollout_log(
        rollout_id, agent_name, task_id, task_query,
        pool_size, retrieve_n, len(candidates), k, selected, servers_mounted, servers_failed,
        all_tools, agent_result, feedback, latency_s,
        rerank_method=rerank_method, task_category=task_category,
    )


def _build_rollout_log(
    rollout_id, agent_name, task_id, task_query,
    pool_size, retrieve_n, candidates_returned, k, selected,
    servers_mounted, servers_failed,
    all_tools, agent_result, feedback, latency_s, error=None,
    rerank_method="unknown", task_category="",
) -> dict:
    """Build rollout log in the JSONL format from CLAUDE.md."""
    log_entry = {
        "rollout_id": rollout_id,
        "agent": agent_name,
        "task_id": task_id,
        "task_query": task_query,
        "task_category": task_category,
        "stage_retrieve": {
            "pool_size": pool_size,
            "candidates_returned": candidates_returned,
            "method": "cosine_similarity",
        },
        "stage_rerank": {
            "candidates_in": candidates_returned,
            "method": rerank_method,
            "top_k": k,
        },
        "inventory_mounted": servers_mounted,
        "inventory_failed": servers_failed,
        "tools_available": [f"{t['server_id']}:{t['name']}" for t in all_tools] if all_tools else [],
    }

    if agent_result:
        log_entry.update({
            "tools_selected": agent_result.get("tools_selected", []),
            "tools_abandoned": agent_result.get("tools_abandoned", []),
            "tools_errored": agent_result.get("tools_errored", []),
            "tool_used_final": agent_result.get("tool_used_final"),
            "agent_answer": agent_result.get("answer", ""),
            "input_tokens": agent_result.get("input_tokens", 0),
            "output_tokens": agent_result.get("output_tokens", 0),
        })
    else:
        log_entry["error"] = error or "Unknown error"

    log_entry["feedback"] = feedback
    log_entry["latency_s"] = latency_s

    return log_entry

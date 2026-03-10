"""Rollout orchestrator: runs one complete recommendation → execution → feedback cycle."""

import asyncio
import functools
import json
import logging
import time
from pathlib import Path

from .agent_client import run_agent
from .feedback import collect_feedback
from .mcp_client import (
    FALLBACK_SERVERS,
    MCPServerConnection,
    SmitheryConnection,
    derive_npx_command,
    get_smithery_mcp_url,
)
import numpy as np

from .recommenders.base import BaseRecommender
from .retriever import embed_query, retrieve, retrieve_from_pool, retrieve_from_pool_fast

log = logging.getLogger(__name__)

MAX_TOOLS_FOR_AGENT = 50

_ROOT = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=1)
def _load_schema_cache() -> dict:
    """Load precomputed tool schema cache from data/pool_schema_cache.json."""
    cache_path = _ROOT / "data" / "pool_schema_cache.json"
    if not cache_path.exists():
        log.warning("Schema cache not found at %s — will use live connections", cache_path)
        return {}
    data = json.loads(cache_path.read_text())
    servers = data.get("servers", {})
    mountable = sum(1 for s in servers.values() if s.get("mountable"))
    log.info("Schema cache loaded: %d/%d mountable servers", mountable, len(servers))
    return servers


def _select_relevant_tools(all_tools: list[dict], query_emb: np.ndarray | None, task_query: str, max_tools: int = MAX_TOOLS_FOR_AGENT) -> list[dict]:
    """Select the most task-relevant tools by cosine similarity of tool descriptions to the task."""
    if len(all_tools) <= max_tools:
        return all_tools

    if query_emb is None:
        query_emb = embed_query(task_query)

    from .retriever import _get_client
    from .config import EMBEDDING_MODEL
    client = _get_client()

    # Embed tool descriptions
    texts = []
    for t in all_tools:
        desc = f"{t.get('name', '')}. {t.get('description', '')}"[:500]
        texts.append(desc)

    tool_embs = []
    for i in range(0, len(texts), 256):
        chunk = texts[i:i + 256]
        resp = client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        for item in sorted(resp.data, key=lambda x: x.index):
            tool_embs.append(np.array(item.embedding, dtype=np.float32))

    tool_matrix = np.array(tool_embs)
    norms = np.linalg.norm(tool_matrix, axis=1, keepdims=True) + 1e-9
    tool_matrix = tool_matrix / norms

    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    sims = tool_matrix @ q_norm
    top_indices = np.argsort(sims)[::-1][:max_tools]

    selected = [all_tools[i] for i in sorted(top_indices)]
    log.info(f"Tool selection: {len(all_tools)} → {len(selected)} (by task relevance)")
    return selected


def _is_mcp_remote(server: dict) -> bool:
    """Check if server uses mcp-remote (Smithery OAuth — can't connect headlessly)."""
    args = server.get("args", [])
    return any("mcp-remote" in str(a) for a in args)


async def _try_connect(server: dict) -> tuple[MCPServerConnection | SmitheryConnection | None, dict]:
    """Try to connect to an MCP server. Returns (connection, tool_list) or (None, []).

    Handles three connection methods:
    - smithery: Smithery Connect REST API (no OAuth)
    - remote_endpoint: mcp-remote over stdio
    - npx: direct npx command
    """
    server_id = server["id"]
    connection = server.get("connection", {})
    method = connection.get("method", "")

    # Smithery servers: connect via Smithery Connect REST API
    if method == "smithery":
        slug = connection.get("slug", "")
        smithery_url = f"https://server.smithery.ai/{slug}/mcp" if slug else get_smithery_mcp_url(server)
        if not smithery_url:
            log.debug(f"Smithery server {server_id} has no slug or URL")
            return None, {}
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

    # Remote endpoint servers: proxy through Smithery Connect (avoids OAuth hangs)
    if method == "remote_endpoint":
        endpoints = connection.get("endpoints", [])
        if not endpoints:
            log.debug(f"Remote endpoint server {server_id} has no endpoints")
            return None, {}
        conn = SmitheryConnection(server_id, endpoints[0])
        ok = await conn.connect()
        if not ok:
            await conn.close()
            return None, {}
        tools = await conn.list_tools()
        if not tools:
            log.info(f"{server_id}: Smithery-proxied but no tools")
            await conn.close()
            return None, {}
        return conn, tools

    # Legacy: check for mcp-remote in args (old format)
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
    epsilon: float = 0.0,
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

    # Stage 2: Rerank / select top K. No buffer — if a recommended server
    # fails to mount, that's a real signal about the recommender's quality.
    rerank_method = recommender.method_name
    ranked = recommender.recommend(agent_name, task_query, candidates, k, task_category=task_category, task_emb=query_emb, epsilon=epsilon)
    selected = ranked[:k]
    log.info(f"[Rollout {rollout_id}] Selected {len(selected)} servers ({rerank_method})")

    # Build server list for this rollout
    servers_to_try = list(ranked)
    if use_fallbacks:
        for fb in FALLBACK_SERVERS:
            if not any(s["id"] == fb["id"] for s in servers_to_try):
                servers_to_try.append(fb)

    # Stage 3: Build tool list from schema cache; connect lazily on first tool call
    schema_cache = _load_schema_cache()
    connections: dict[str, MCPServerConnection | SmitheryConnection] = {}
    all_tools: list[dict] = []
    servers_mounted = []
    servers_failed = []

    # Map server_id → server dict for lazy lookup
    server_by_id = {s["id"]: s for s in servers_to_try}

    for server in servers_to_try:
        sid = server["id"]
        is_fallback = sid.startswith("__fallback_")

        if is_fallback:
            # Fallback servers: eagerly connect (not in pool schema cache)
            try:
                conn, tools = await _try_connect(server)
            except Exception as e:
                log.warning(f"Exception connecting to fallback {sid}: {e}")
                servers_failed.append(sid)
                continue
            if not tools:
                servers_failed.append(sid)
                continue
            if conn:
                connections[sid] = conn
            all_tools.extend(tools)
            servers_mounted.append(sid)
        else:
            # Pool servers: use cached schemas if available, defer live connection
            cached = schema_cache.get(sid)
            if cached and cached.get("mountable") and cached.get("tools"):
                all_tools.extend(cached["tools"])
                servers_mounted.append(sid)
                log.debug(f"[Rollout {rollout_id}] {sid}: {len(cached['tools'])} tools from cache")
            else:
                # Not probed as mountable → immediate failure signal
                servers_failed.append(sid)
                log.debug(f"[Rollout {rollout_id}] {sid}: not in schema cache, skipping")

    # Lazy connect: called by agent when it first calls a tool on a pool server
    async def lazy_connect(server_id: str) -> MCPServerConnection | SmitheryConnection | None:
        if server_id in connections:
            return connections[server_id]
        server = server_by_id.get(server_id)
        if not server:
            return None
        log.info(f"[Rollout {rollout_id}] Lazy-connecting to {server_id}...")
        try:
            conn, tools = await _try_connect(server)
        except Exception as e:
            log.warning(f"[Rollout {rollout_id}] Lazy connect failed for {server_id}: {e}")
            conn, tools = None, []
        if tools:
            connections[server_id] = conn
            return conn
        # Connection failed at call time: demote from mounted → failed
        if conn:
            await conn.close()
        if server_id in servers_mounted:
            servers_mounted.remove(server_id)
        if server_id not in servers_failed:
            servers_failed.append(server_id)
        return None

    # Select most relevant tools by semantic similarity to task
    all_tools = _select_relevant_tools(all_tools, query_emb, task_query)

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
        agent_result = await run_agent(agent_name, task_query, all_tools, connections, lazy_connect_fn=lazy_connect)
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

"""Run GAIA L1 tasks with per-task curated MCP servers."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Web search bundle (general purpose)
WEB = ["pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom", "pulse/scrapi"]
# Academic search bundle
ACAD = ["adamamer20/paper-search-mcp-openai", "hamid-vakilzadeh/mcpsemanticscholar", "pulse/tavily-search", "exa", "OEvortex/ddg_search"]
# YouTube + web
YT = ["sfiorini/youtube-mcp", "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom"]
# Code execution + web
CODE = ["STUzhy/py_execute_mcp", "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom"]
# Docs + web
DOCS = ["pulse/upstash-context7", "docfork/docfork", "pulse/tavily-search", "exa", "pulse/scrapi"]

CURATED = {
    # Q1: Kipchoge marathon pace, Earth-Moon distance calc
    "e1fc63a2": CODE,
    # Q2: Mercedes Sosa studio albums 2000-2009 (Wikipedia)
    "8e867cd7": WEB,
    # Q3: Ping-pong riddle (logic/math)
    "ec09fa32": CODE,
    # Q4: University of Leicester fish bag volume (academic paper)
    "5d0080cb": ACAD,
    # Q5: YouTube video bird species count
    "a1e91b78": YT,
    # Q6: "Pie Menus or Linear Menus" paper author's first paper
    "46719c30": ACAD,
    # Q7: Doctor Who maze location in script
    "4b6bb5f7": WEB,
    # Q8: Reversed text puzzle (no tool needed really)
    "2d83110e": CODE,
    # Q9: Logic equivalence (no tool needed)
    "27d5d136": CODE,
    # Q10: Family reunion mashed potatoes math (no tool needed)
    "dc28cf18": CODE,
    # Q11: Emily Midkiff article in Fafnir journal
    "b816bfce": ["pulse/tavily-search", "exa", "pulse/scrapi", "adamamer20/paper-search-mcp-openai", "OEvortex/ddg_search"],
    # Q12: Bielefeld University BASE DDC 633
    "72e110e7": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q13: Tizin fictional language (no tool needed)
    "42576abe": CODE,
    # Q14: Nature Scientific Reports 2012 nano-compound
    "b415aba4": ACAD,
    # Q15: Wikipedia content policy "R" stands for
    "935e2cff": WEB,
    # Q16: Wikipedia Featured Article dinosaur Nov 2016
    "4fc2f1ae": WEB,
    # Q17: Merriam-Webster Word of the Day June 27 2022
    "5188369a": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q18: Abstract algebra * table (no tool needed)
    "6f37996b": CODE,
    # Q19: Instruction following "Guava" (no tool needed)
    "4b650a35": CODE,
    # Q20: Van Helsing vampire investigation (reasoning + web)
    "c714ab3a": WEB,
    # Q21: YouTube Stargate Teal'c quote
    "9d191bce": YT,
    # Q22: Chemistry textbook equine veterinarian
    "cabe07ed": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q23: Grocery list botany categorization (no tool needed)
    "3cef3a44": CODE,
    # Q24: Scikit-learn July 2017 changelog
    "d0633230": DOCS,
    # Q25: Polish Raymond actor in Magda M
    "305ac316": WEB,
    # Q26: BBC Earth YouTube silly animal moments bird
    "0383a3ee": YT,
    # Q27: BERT vs Transformer encoder layers (reasoning)
    "11af4e1a": ["pulse/tavily-search", "exa", "adamamer20/paper-search-mcp-openai", "OEvortex/ddg_search", "pulse/youcom"],
    # Q28: Game show probability (math reasoning)
    "e142056d": CODE,
    # Q29: 5x7 text block puzzle (no tool needed)
    "50ad0280": CODE,
    # Q30: Cornell Law federal rules word deleted
    "7673d772": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q31: US president birthplaces westernmost to easternmost
    "c365c1c7": WEB,
    # Q32: Girls Who Code percentage stats
    "7d4a7d1d": WEB,
    # Q33: James Beard Award book title
    "dc22a632": WEB,
    # Q34: Yankees 1977 walks + at bats
    "3f57289b": ["etweisberg/mlb-mcp", "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom"],
    # Q35: Audre Lorde poem stanza indentation
    "23dd907f": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q36: Universe Today article NASA grant number
    "840bfca7": ["pulse/tavily-search", "exa", "pulse/scrapi", "OEvortex/ddg_search", "pulse/youcom"],
    # Q37: NIH clinical trial H. pylori enrollment
    "a0068077": ["plainyogurt21/clintrials-mcp", "pulse/tavily-search", "exa", "OEvortex/ddg_search", "pulse/youcom"],
    # Q38: Vietnamese specimens Kuznetzov paper deposited city
    "bda648d7": ACAD,
    # Q39: Rubik's cube missing cube colors (reasoning)
    "50ec8903": CODE,
    # Q40: 1928 Olympics least athletes country
    "cf106601": WEB,
    # Q41: Tamai pitcher numbers (Japanese baseball)
    "a0c07678": WEB,
    # Q42: Malko Competition 20th century recipient
    "5a0c1adf": WEB,
}


async def run_one(task, servers_to_mount, pool_by_id, schema_cache):
    from src.recsys.agent_client import run_agent
    from src.recsys.feedback import collect_feedback
    from src.recsys.pipeline import _try_connect

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
            servers_failed.append(sid)
            continue
        server_by_id[sid] = server
        cached = schema_cache.get(sid)
        if cached and cached.get("mountable") and cached.get("tools"):
            all_tools.extend(cached["tools"])
            servers_mounted.append(sid)
        else:
            servers_failed.append(sid)

    async def lazy_connect(server_id):
        if server_id in connections:
            return connections[server_id]
        server = server_by_id.get(server_id)
        if not server:
            return None
        try:
            conn, tools = await _try_connect(server)
        except Exception:
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

    logging.info(f"[{task_id[:8]}] {len(servers_mounted)} mounted, {len(all_tools)} tools")

    if not all_tools:
        return {"task_id": task_id, "task_query": task_query, "error": "No tools",
                "agent_answer": "", "level": task.get("level", ""),
                "final_answer_gt": task.get("final_answer", "")}

    agent_name = "deepseek-v3.2"
    try:
        agent_result = await run_agent(agent_name, task_query, all_tools, connections, lazy_connect_fn=lazy_connect)
    except Exception as e:
        agent_result = {"answer": f"Error: {e}", "tools_selected": [], "tools_results": [],
                        "tools_abandoned": [], "tools_errored": [], "tool_used_final": None,
                        "input_tokens": 0, "output_tokens": 0}

    tools_offered = [f"{t['server_id']}:{t['name']}" for t in all_tools]
    tools_used = agent_result.get("tools_selected", [])
    try:
        feedback = await collect_feedback(agent_name, task_query, agent_result["answer"], tools_offered, tools_used)
    except Exception:
        feedback = {}

    for conn in connections.values():
        await conn.close()

    return {
        "task_id": task_id, "task_query": task_query,
        "task_category": task.get("category", ""),
        "final_answer_gt": task.get("final_answer", ""),
        "level": task.get("level", ""), "agent": agent_name,
        "inventory_mounted": servers_mounted, "inventory_failed": servers_failed,
        "tools_available": [f"{t['server_id']}:{t['name']}" for t in all_tools],
        "tools_selected": agent_result.get("tools_selected", []),
        "tools_errored": agent_result.get("tools_errored", []),
        "tools_abandoned": agent_result.get("tools_abandoned", []),
        "tool_used_final": agent_result.get("tool_used_final"),
        "agent_answer": agent_result.get("answer", ""),
        "input_tokens": agent_result.get("input_tokens", 0),
        "output_tokens": agent_result.get("output_tokens", 0),
        "feedback": feedback,
        "latency_s": round(time.time() - t0, 2),
        "recommender": "curated_oracle",
    }


def get_servers(task_id):
    for key, servers in CURATED.items():
        if task_id.startswith(key):
            return servers
    return WEB  # fallback


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(ROOT / "data" / "mountable_pool.json") as f:
        pool = json.load(f)
    pool_by_id = {s["id"]: s for s in pool}

    cache_path = ROOT / "data" / "pool_schema_cache.json"
    schema_cache = json.loads(cache_path.read_text()).get("servers", {}) if cache_path.exists() else {}

    with open(ROOT / "data" / "tasks_gaia.json") as f:
        all_tasks = json.load(f)
    l1_tasks = [t for t in all_tasks if t.get("level") == 1]
    logging.info(f"GAIA L1 tasks: {len(l1_tasks)}")

    output_path = ROOT / "results" / "gaia_deepseek" / "curated_l1.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get("task_id", ""))

    for i, task in enumerate(l1_tasks):
        if task["uuid"] in done:
            logging.info(f"[{i+1}/{len(l1_tasks)}] Skipping {task['uuid'][:8]} (done)")
            continue

        servers = get_servers(task["uuid"])
        logging.info(f"\n{'='*60}")
        logging.info(f"[{i+1}/{len(l1_tasks)}] Q: {task['query'][:100]}")
        logging.info(f"Servers: {servers}")

        result = await run_one(task, servers, pool_by_id, schema_cache)

        with open(output_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        ans = result.get("agent_answer", "")[:150]
        gt = task.get("final_answer", "")
        logging.info(f"  Answer: {ans}")
        logging.info(f"  GT: {gt}")

    # Summary
    if output_path.exists():
        with open(output_path) as f:
            results = [json.loads(l) for l in f if l.strip()]
        correct = 0
        for r in results:
            gt = r.get("final_answer_gt", "").strip().lower()
            ans = r.get("agent_answer", "").lower()
            if gt and gt in ans and ans != "(max turns reached)":
                correct += 1
        logging.info(f"\n{'='*60}")
        logging.info(f"SUMMARY: {correct}/{len(results)} correct ({correct/len(results)*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())

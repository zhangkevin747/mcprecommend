"""Build a pool of ~300 MCPs from the top-100 retrieval union.

Strategy:
1. Get the union of top-100 MCPs across all 1268 benchmark tasks
2. For each, try to determine an npx install command
3. Rank by a mix of retrieval frequency (how often they appear) and popularity
4. Select 300 with a mix of popular and obscure
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Load data
embeddings = np.load(ROOT / "data/index/embeddings.npy")
with open(ROOT / "data/index/embedding_index.json") as f:
    emb_index = json.load(f)
with open(ROOT / "data/index/mcp_server_index.json") as f:
    server_data = json.load(f)

servers = server_data["servers"]
server_by_id = {s["id"]: s for s in servers}

print(f"Embeddings: {embeddings.shape}")

# Normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings_norm = embeddings / norms

# Load all benchmark tasks
task_files = {
    "search": ROOT / "data/search/search_0725_single_v2.json",
    "browser": ROOT / "data/browser/browser_0724_single_v3.json",
    "finance": ROOT / "data/finance/finance_0724_single_v3.json",
    "map": ROOT / "data/map/map_0717_single_multi_lang_500.json",
    "pay": ROOT / "data/pay/pay_0723_single.json",
}

all_tasks = []
for domain, path in task_files.items():
    with open(path) as f:
        tasks = json.load(f)
    for t in tasks:
        q = t.get("query") or t.get("question") or t.get("instruction", "")
        if q:
            all_tasks.append({"query": q, "domain": domain})

print(f"Total tasks: {len(all_tasks)}")

# Batch embed all queries
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("Embedding queries...")
t0 = time.time()
all_query_embs = []
batch_size = 200
for i in range(0, len(all_tasks), batch_size):
    batch = [t["query"] for t in all_tasks[i : i + batch_size]]
    resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
    for item in sorted(resp.data, key=lambda x: x.index):
        all_query_embs.append(item.embedding)

query_embs = np.array(all_query_embs)
q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
q_norms[q_norms == 0] = 1
query_embs_norm = query_embs / q_norms
print(f"Embedded in {time.time() - t0:.1f}s")

# Compute similarities
print("Computing cosine similarities...")
sims = query_embs_norm @ embeddings_norm.T

# Count how often each MCP appears in top-100 across all tasks
print("Counting retrieval frequency...")
retrieval_freq = Counter()
for i in range(len(all_tasks)):
    top_idx = np.argsort(sims[i])[-100:][::-1]
    for idx in top_idx:
        sid = emb_index[idx]["id"]
        retrieval_freq[sid] += 1

print(f"Unique MCPs in top-100 union: {len(retrieval_freq)}")
print(f"Top 20 most retrieved:")
for sid, count in retrieval_freq.most_common(20):
    s = server_by_id.get(sid, {})
    print(f"  {count:5d}x  {sid:50s}  stars={s.get('stars', '?')}")


# --- Heuristic: guess npm package name from server metadata ---
def guess_npm_package(server: dict) -> str | None:
    """Try to guess the npm package name for a server."""
    sid = server["id"]
    name = server.get("name", "")
    repo = server.get("repo_url", "")

    # Known benchmark servers
    KNOWN = {
        "tavily-mcp": "tavily-mcp",
        "google-search": "@adenot/mcp-google-search",
        "playwright": "@executeautomation/playwright-mcp-server",
        "puppeteer": "@modelcontextprotocol/server-puppeteer",
        "google-maps": "@modelcontextprotocol/server-google-maps",
        "paypal": "@paypal/mcp",
    }
    if sid in KNOWN:
        return KNOWN[sid]

    # If id looks like an npm package (starts with @ or has no /)
    if sid.startswith("@") and "/" in sid:
        return sid  # scoped npm package

    # If repo_url is github, try to derive from it
    if repo and "github.com" in repo:
        # e.g. https://github.com/user/mcp-server-foo -> mcp-server-foo
        match = re.search(r"github\.com/[^/]+/([^/]+?)(?:\.git)?$", repo)
        if match:
            repo_name = match.group(1)
            # Many MCP servers have the npm package = repo name
            return repo_name

    # If the id itself looks like a package name (no special chars)
    if re.match(r"^[a-zA-Z@][a-zA-Z0-9._/-]*$", sid):
        return sid

    return None


# Build candidate pool with install info
print("\nBuilding candidate pool...")
candidates = []
for sid, freq in retrieval_freq.items():
    s = server_by_id.get(sid, {})
    if not s:
        continue

    npm_pkg = guess_npm_package(s)
    sources = s.get("sources", [])
    is_deployed = s.get("is_deployed", False)

    # Determine install method
    install_method = None
    install_cmd = None

    if npm_pkg:
        install_method = "npx"
        install_cmd = f"npx -y {npm_pkg}"
    elif "smithery" in sources and is_deployed:
        install_method = "smithery"
        install_cmd = f"npx -y mcp-remote https://server.smithery.ai/{sid}/mcp"

    candidates.append({
        "id": sid,
        "name": s.get("name", ""),
        "description": (s.get("description", "") or "")[:200],
        "sources": sources,
        "stars": s.get("stars"),
        "tool_count": s.get("tool_count", 0),
        "is_deployed": is_deployed,
        "retrieval_freq": freq,
        "npm_package": npm_pkg,
        "install_method": install_method,
        "install_cmd": install_cmd,
    })

# Sort by installability then frequency
npx_candidates = [c for c in candidates if c["install_method"] == "npx"]
smithery_candidates = [c for c in candidates if c["install_method"] == "smithery"]
no_install = [c for c in candidates if c["install_method"] is None]

print(f"\nInstallable via npx: {len(npx_candidates)}")
print(f"Installable via Smithery: {len(smithery_candidates)}")
print(f"No install method: {len(no_install)}")

# Select 300: prioritize npx, mix popular and obscure
# Sort npx candidates by retrieval frequency
npx_candidates.sort(key=lambda c: c["retrieval_freq"], reverse=True)

# Take all npx candidates (up to 300)
selected = npx_candidates[:300]

# If we need more, add Smithery
if len(selected) < 300:
    smithery_candidates.sort(key=lambda c: c["retrieval_freq"], reverse=True)
    remaining = 300 - len(selected)
    selected.extend(smithery_candidates[:remaining])

print(f"\nSelected {len(selected)} MCPs for pool")
print(f"  npx: {sum(1 for c in selected if c['install_method'] == 'npx')}")
print(f"  smithery: {sum(1 for c in selected if c['install_method'] == 'smithery')}")

# Show distribution
freq_values = [c["retrieval_freq"] for c in selected]
print(f"\nRetrieval frequency distribution:")
print(f"  max: {max(freq_values)}, min: {min(freq_values)}, median: {sorted(freq_values)[len(freq_values)//2]}")

# Stars distribution
starred = [c["stars"] for c in selected if c["stars"] and c["stars"] > 0]
if starred:
    print(f"Stars distribution (of {len(starred)} with stars):")
    print(f"  max: {max(starred)}, min: {min(starred)}, median: {sorted(starred)[len(starred)//2]}")

# Show top 30
print(f"\nTop 30 selected:")
for c in selected[:30]:
    print(f"  freq={c['retrieval_freq']:4d}  stars={str(c['stars'] or '?'):>6s}  {c['install_method']:8s}  {c['id'][:50]}")

# Show bottom 30 (most obscure)
print(f"\nBottom 30 (most obscure):")
for c in selected[-30:]:
    print(f"  freq={c['retrieval_freq']:4d}  stars={str(c['stars'] or '?'):>6s}  {c['install_method']:8s}  {c['id'][:50]}")

# Save the pool
output_path = ROOT / "data" / "pool" / "mcp_pool_300.json"
with open(output_path, "w") as f:
    json.dump(selected, f, indent=2)
print(f"\nSaved to {output_path}")

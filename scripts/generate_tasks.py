"""
Task Generation Pipeline for MCP Tool Recommender Experiment.

Steps:
1. Embed 766 MCP descriptions (combined Smithery + PulseMCP pool)
2. Cluster into 15 capability clusters
3. Generate tasks proportional to cluster size
4. Validate each task has retrievable MCPs
5. Deduplicate by embedding similarity
"""

import json
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_DIR = Path("data")
MCP_FILE = DATA_DIR / "combined_server_pool.json"
EMBEDDINGS_FILE = DATA_DIR / "mcp_embeddings.npz"
CLUSTERS_FILE = DATA_DIR / "mcp_clusters.json"
TASKS_RAW_FILE = DATA_DIR / "tasks_raw.json"
TASKS_FINAL_FILE = DATA_DIR / "tasks.json"

EMBEDDING_MODEL = "text-embedding-3-small"
TASK_GEN_MODEL = "gpt-5-mini"
N_CLUSTERS = 15
TARGET_TASKS = 2500
MIN_TASKS_PER_CLUSTER = 10


# ── Step 1: Embed MCP descriptions ──────────────────────────────────────────

def load_mcps():
    with open(MCP_FILE) as f:
        data = json.load(f)
    return data["servers"]


def embed_descriptions(mcps, batch_size=100):
    """Embed MCP descriptions. Cache to disk."""
    if EMBEDDINGS_FILE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_FILE}")
        npz = np.load(EMBEDDINGS_FILE)
        return npz["embeddings"], npz["ids"]

    texts = []
    ids = []
    for m in mcps:
        name = m.get("name", m.get("displayName", ""))
        desc = f"{name}. {m.get('description', '')}"
        texts.append(desc[:8000])
        ids.append(m.get("id", m.get("qualifiedName", name)))

    print(f"Embedding {len(texts)} MCP descriptions...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embs = [e.embedding for e in resp.data]
        all_embeddings.extend(batch_embs)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.1)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    ids = np.array(ids)
    np.savez(EMBEDDINGS_FILE, embeddings=embeddings, ids=ids)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")
    return embeddings, ids


# ── Step 2: Cluster ─────────────────────────────────────────────────────────

def cluster_mcps(embeddings, ids, mcps, n_clusters=N_CLUSTERS):
    """K-means clustering of MCP embeddings."""
    from sklearn.cluster import KMeans

    if CLUSTERS_FILE.exists():
        print(f"Loading cached clusters from {CLUSTERS_FILE}")
        with open(CLUSTERS_FILE) as f:
            return json.load(f)

    print(f"Clustering {len(embeddings)} MCPs into {n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Build cluster info
    mcp_lookup = {}
    for m in mcps:
        key = m.get("id", m.get("qualifiedName", m.get("name", "")))
        mcp_lookup[key] = m
    clusters = {}
    for i, (mcp_id, label) in enumerate(zip(ids, labels)):
        label = int(label)
        if label not in clusters:
            clusters[label] = {"mcps": [], "descriptions": []}
        m = mcp_lookup.get(mcp_id, {})
        name = m.get("name", m.get("displayName", mcp_id))
        clusters[label]["mcps"].append(str(mcp_id))
        clusters[label]["descriptions"].append(
            f"{name}: {m.get('description', 'N/A')[:200]}"
        )

    # Summarize each cluster
    for label, info in clusters.items():
        info["size"] = len(info["mcps"])

    with open(CLUSTERS_FILE, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"Saved {n_clusters} clusters to {CLUSTERS_FILE}")

    sizes = sorted([c["size"] for c in clusters.values()], reverse=True)
    print(f"Cluster sizes: max={sizes[0]}, min={sizes[-1]}, median={sizes[len(sizes)//2]}")
    return clusters


# ── Step 3: Generate tasks ──────────────────────────────────────────────────

def compute_tasks_per_cluster(clusters, target_total=TARGET_TASKS, min_per=MIN_TASKS_PER_CLUSTER):
    """Allocate tasks proportional to cluster size with a minimum floor."""
    total_mcps = sum(c["size"] for c in clusters.values())
    allocations = {}
    for label, info in clusters.items():
        raw = (info["size"] / total_mcps) * target_total
        allocations[label] = max(min_per, round(raw))

    # Scale to hit target
    current_total = sum(allocations.values())
    if current_total > target_total * 1.2:
        # Reduce proportionally from large clusters
        excess = current_total - target_total
        large = {k: v for k, v in allocations.items() if v > min_per}
        large_total = sum(large.values())
        for k in large:
            reduction = round((large[k] / large_total) * excess)
            allocations[k] = max(min_per, allocations[k] - reduction)

    print(f"Task allocation: {sum(allocations.values())} tasks across {len(allocations)} clusters")
    return allocations


def generate_tasks_for_cluster(cluster_label, cluster_info, num_tasks):
    """Generate tasks for a single cluster using LLM."""
    # Pick representative descriptions (up to 20)
    descs = cluster_info["descriptions"][:20]
    desc_block = "\n".join(f"- {d}" for d in descs)

    prompt = f"""You are generating evaluation tasks for an AI tool recommender system.

These are MCP tool servers in a capability cluster:

{desc_block}

Generate exactly {num_tasks} tasks. Each task is a natural language request (1-2 sentences) that an AI agent would need an external tool to answer.

STRICT RULES — every task MUST satisfy ALL of these:

1. REQUIRES A TOOL: The task needs live/external data that an LLM cannot answer from memory. Good: "What is the current price of Bitcoin?" Bad: "Explain how Bitcoin mining works."

2. SINGLE ROLLOUT: Completable with one tool call. No multi-step chains, no "first do X then Y."

3. NO PERSONAL STATE: Do not assume the user has accounts, files, repos, databases, projects, teams, calendars, inboxes, or any pre-existing data. Do not use "my", "our", or assume authenticated access. Good: "Find the top trending GitHub repos today." Bad: "List the open issues in my GitHub repo."

4. TOOL-AGNOSTIC: Do not name specific tools, APIs, or services. The task should describe WHAT the user wants, not HOW to get it.

5. CONCRETE & SPECIFIC: Include specific entities, dates, locations, or parameters. Good: "What's the weather forecast for Tokyo this weekend?" Bad: "Get some weather data."

6. NO FABRICATED REFERENCES: Only reference real, publicly accessible entities (real companies, real cities, real GitHub repos, real stock tickers). NEVER invent URLs, IDs, template names, project names, or endpoints that don't exist. The agent will actually try to use these — fake references cause false errors.

7. DIVERSE: Vary the topic, complexity, and phrasing. Mix questions, commands, and requests.

Return a JSON array of strings. No other text."""

    resp = client.chat.completions.create(
        model=TASK_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        # Handle {"tasks": [...]} or just [...]
        if isinstance(parsed, dict):
            tasks = parsed.get("tasks", parsed.get("questions", list(parsed.values())[0]))
        else:
            tasks = parsed
        return [t for t in tasks if isinstance(t, str)]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Failed to parse response for cluster {cluster_label}: {e}")
        return []


CONCURRENCY = 32
MAX_PER_CALL = 40


def generate_all_tasks(clusters, allocations):
    """Generate tasks for all clusters with concurrent LLM calls."""
    if TASKS_RAW_FILE.exists():
        print(f"Loading cached raw tasks from {TASKS_RAW_FILE}")
        with open(TASKS_RAW_FILE) as f:
            return json.load(f)

    # Build all work items: (cluster_label, batch_count)
    work_items = []
    for label, num_tasks in allocations.items():
        remaining = num_tasks
        while remaining > 0:
            batch_count = min(remaining, MAX_PER_CALL)
            work_items.append((label, batch_count))
            remaining -= batch_count

    print(f"  {len(work_items)} LLM calls across {len(allocations)} clusters (concurrency={CONCURRENCY})")

    all_tasks = []
    completed = 0

    def do_batch(item):
        label, batch_count = item
        tasks = generate_tasks_for_cluster(label, clusters[label], batch_count)
        return label, tasks

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(do_batch, item): item for item in work_items}
        for future in as_completed(futures):
            label, tasks = future.result()
            for t in tasks:
                all_tasks.append({
                    "query": t,
                    "cluster_id": int(label),
                    "cluster_size": clusters[label]["size"],
                })
            completed += 1
            if completed % 5 == 0 or completed == len(work_items):
                print(f"  {completed}/{len(work_items)} batches done ({len(all_tasks)} tasks)")

    print(f"Generated {len(all_tasks)} raw tasks")
    with open(TASKS_RAW_FILE, "w") as f:
        json.dump(all_tasks, f, indent=2)
    return all_tasks


# ── Step 4: Validate retrievability ─────────────────────────────────────────

def validate_tasks(tasks, embeddings, ids, top_k=10, min_similarity=0.3):
    """Check each task has at least one retrievable MCP above similarity threshold."""
    print(f"Validating {len(tasks)} tasks against MCP pool...")

    # Normalize MCP embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / norms

    # Embed all task queries
    queries = [t["query"] for t in tasks]
    task_embeddings = []
    for i in range(0, len(queries), 100):
        batch = queries[i:i + 100]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        task_embeddings.extend([e.embedding for e in resp.data])
        time.sleep(0.1)

    task_emb = np.array(task_embeddings, dtype=np.float32)
    task_norms = np.linalg.norm(task_emb, axis=1, keepdims=True)
    task_emb_normed = task_emb / task_norms

    # Cosine similarity: tasks × MCPs
    sims = task_emb_normed @ emb_normed.T  # (n_tasks, n_mcps)

    valid_tasks = []
    dropped = 0
    for i, task in enumerate(tasks):
        top_sims = np.sort(sims[i])[-top_k:]
        if top_sims[-1] >= min_similarity:
            task["top_mcp_similarity"] = float(top_sims[-1])
            task["embedding_index"] = i
            valid_tasks.append(task)
        else:
            dropped += 1

    print(f"Valid: {len(valid_tasks)}, Dropped: {dropped} (max sim < {min_similarity})")
    return valid_tasks, task_emb


# ── Step 5: Deduplicate ─────────────────────────────────────────────────────

def deduplicate_tasks(tasks, task_embeddings, similarity_threshold=0.92):
    """Remove near-duplicate tasks by embedding similarity."""
    print(f"Deduplicating {len(tasks)} tasks (threshold={similarity_threshold})...")

    indices = [t["embedding_index"] for t in tasks]
    embs = task_embeddings[indices]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / norms

    # Pairwise similarity (chunked to avoid memory issues)
    keep = set(range(len(tasks)))
    for i in range(len(tasks)):
        if i not in keep:
            continue
        sims = embs_normed[i] @ embs_normed.T
        for j in range(i + 1, len(tasks)):
            if j in keep and sims[j] > similarity_threshold:
                keep.discard(j)

    deduped = [tasks[i] for i in sorted(keep)]
    print(f"After dedup: {len(deduped)} tasks (removed {len(tasks) - len(deduped)} duplicates)")
    return deduped


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task Generation Pipeline")
    print("=" * 60)

    # Step 1
    print("\n── Step 1: Embed MCP descriptions ──")
    mcps = load_mcps()
    embeddings, ids = embed_descriptions(mcps)

    # Step 2
    print("\n── Step 2: Cluster MCPs ──")
    clusters = cluster_mcps(embeddings, ids, mcps, n_clusters=N_CLUSTERS)

    # Step 3
    print("\n── Step 3: Generate tasks ──")
    allocations = compute_tasks_per_cluster(clusters)
    tasks = generate_all_tasks(clusters, allocations)

    # Step 4
    print("\n── Step 4: Validate retrievability ──")
    valid_tasks, task_emb = validate_tasks(tasks, embeddings, ids)

    # Step 5
    print("\n── Step 5: Deduplicate ──")
    final_tasks = deduplicate_tasks(valid_tasks, task_emb)

    # Clean up and save
    for i, t in enumerate(final_tasks):
        t["task_id"] = f"task_{i:04d}"
        t.pop("embedding_index", None)

    with open(TASKS_FINAL_FILE, "w") as f:
        json.dump(final_tasks, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Final: {len(final_tasks)} tasks saved to {TASKS_FINAL_FILE}")
    print(f"Cluster coverage: {len(set(t['cluster_id'] for t in final_tasks))}/{len(clusters)} clusters")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

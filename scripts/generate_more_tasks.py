"""Generate 200 more tasks (spread across clusters) and merge with existing tasks.json."""
import json
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_DIR = Path("data")
CLUSTERS_FILE = DATA_DIR / "mcp_clusters.json"
EMBEDDINGS_FILE = DATA_DIR / "mcp_embeddings.npz"
TASKS_FILE = DATA_DIR / "tasks.json"
EMBEDDING_MODEL = "text-embedding-3-small"
TASK_GEN_MODEL = "gpt-5-mini"
TASKS_PER_CLUSTER = 14  # ~14 * 15 = 210 raw tasks


def generate_tasks_for_cluster(cluster_label, cluster_info, num_tasks, existing_examples):
    descs = cluster_info["descriptions"][:20]
    desc_block = "\n".join(f"- {d}" for d in descs)

    # Show a few existing tasks so the model avoids duplicates
    avoid_block = ""
    if existing_examples:
        avoid_block = "\n\nDo NOT generate tasks similar to these (already in the dataset):\n"
        avoid_block += "\n".join(f"- {e}" for e in existing_examples[:15])

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
{avoid_block}

Return a JSON array of strings. No other text."""

    resp = client.chat.completions.create(
        model=TASK_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            tasks = parsed.get("tasks", parsed.get("questions", list(parsed.values())[0]))
        else:
            tasks = parsed
        return [t for t in tasks if isinstance(t, str)]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Failed to parse response for cluster {cluster_label}: {e}")
        return []


def main():
    with open(CLUSTERS_FILE) as f:
        clusters = json.load(f)
    with open(TASKS_FILE) as f:
        existing_tasks = json.load(f)

    npz = np.load(EMBEDDINGS_FILE)
    embeddings, ids = npz["embeddings"], npz["ids"]

    print(f"Existing tasks: {len(existing_tasks)}")

    # Group existing tasks by cluster for the avoid-duplication prompt
    existing_by_cluster = {}
    for t in existing_tasks:
        cid = str(t.get("cluster_id", ""))
        existing_by_cluster.setdefault(cid, []).append(t["query"])

    # Generate new tasks for each cluster
    print(f"Generating {TASKS_PER_CLUSTER} tasks per cluster ({len(clusters)} clusters)...")
    new_tasks = []

    def do_cluster(label):
        examples = existing_by_cluster.get(label, [])
        tasks = generate_tasks_for_cluster(label, clusters[label], TASKS_PER_CLUSTER, examples)
        return label, tasks

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(do_cluster, label): label for label in clusters}
        for future in as_completed(futures):
            label, tasks = future.result()
            for t in tasks:
                new_tasks.append({
                    "query": t,
                    "cluster_id": int(label),
                    "cluster_size": clusters[label]["size"],
                })
            print(f"  Cluster {label}: {len(tasks)} tasks")

    print(f"\nGenerated {len(new_tasks)} new raw tasks")

    # Validate against MCP pool
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / norms

    queries = [t["query"] for t in new_tasks]
    task_embeddings = []
    for i in range(0, len(queries), 100):
        batch = queries[i:i + 100]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        task_embeddings.extend([e.embedding for e in resp.data])
    new_emb = np.array(task_embeddings, dtype=np.float32)
    new_norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
    new_emb_normed = new_emb / new_norms

    sims = new_emb_normed @ emb_normed.T
    valid = []
    for i, task in enumerate(new_tasks):
        if np.max(sims[i]) >= 0.3:
            task["top_mcp_similarity"] = float(np.max(sims[i]))
            task["_emb_idx"] = i
            valid.append(task)
    print(f"Valid: {len(valid)}, Dropped: {len(new_tasks) - len(valid)}")

    # Deduplicate against existing tasks
    existing_queries = [t["query"] for t in existing_tasks]
    existing_embs = []
    for i in range(0, len(existing_queries), 100):
        batch = existing_queries[i:i + 100]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        existing_embs.extend([e.embedding for e in resp.data])
    existing_emb = np.array(existing_embs, dtype=np.float32)
    existing_norms = np.linalg.norm(existing_emb, axis=1, keepdims=True)
    existing_emb_normed = existing_emb / existing_norms

    # Check each new task against all existing
    kept = []
    for task in valid:
        idx = task["_emb_idx"]
        sim_to_existing = new_emb_normed[idx] @ existing_emb_normed.T
        if np.max(sim_to_existing) < 0.88:
            kept.append(task)

    # Also dedup within new tasks
    final_new = []
    for i, task in enumerate(kept):
        is_dup = False
        for prev in final_new:
            sim = new_emb_normed[task["_emb_idx"]] @ new_emb_normed[prev["_emb_idx"]]
            if sim > 0.88:
                is_dup = True
                break
        if not is_dup:
            final_new.append(task)

    print(f"After dedup (vs existing + internal): {len(final_new)} new tasks")

    # Merge and save
    for task in final_new:
        task.pop("_emb_idx", None)

    all_tasks = existing_tasks + final_new
    # Re-number task IDs
    for i, t in enumerate(all_tasks):
        t["task_id"] = f"task_{i:04d}"

    with open(TASKS_FILE, "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"\nFinal total: {len(all_tasks)} tasks saved to {TASKS_FILE}")
    print(f"Cluster coverage: {len(set(t['cluster_id'] for t in all_tasks))}/{len(clusters)} clusters")


if __name__ == "__main__":
    main()

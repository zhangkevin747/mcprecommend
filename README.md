# MCP Tool Recommender — Experiment 2: Agent Feedback Loop

Can a recommender system learn to navigate a messy, real-world tool ecosystem purely from agent feedback?

## Research Question

Agents degrade with too many tools. Someone must filter thousands of MCPs down to a small inventory per task. We treat this as a recommendation problem: agents are users, MCP servers are items, and agent feedback is the only training signal. No ground truth. No human curation. The recommender learns quality filtering from experience.

## Method

### 1. Tool Pool Construction

**Sources**: Smithery registry and PulseMCP directory

We built a combined pool of 766 remotely-accessible MCP servers from two registries:

| Source | Servers | Connection Method |
|--------|---------|-------------------|
| Smithery | 372 | Smithery Connect REST API (slug-based) |
| PulseMCP | 394 | Remote HTTP endpoints (proxied through Smithery Connect) |
| **Total** | **766** | |

**Smithery**: Scraped the full registry via paginated API (`/servers?page=N&pageSize=50`). Filtered to deployed, remote servers. Each server connects via Smithery Connect — a managed REST API that proxies JSON-RPC 2.0 calls over HTTP, avoiding OAuth browser popups.

**PulseMCP**: Scraped server listings with their published remote endpoints (e.g., `https://mcp.context7.com/mcp`). These are standalone HTTP servers that speak MCP natively. We proxy them through Smithery Connect for uniform connection handling.

**Fields collected per server**: id, name, description, connection config, use count, popularity percentile. PulseMCP servers additionally have GitHub stars and weekly use counts.

No pre-filtering by quality — broken, flaky, and auth-gated tools are all included. The recommender learns to avoid them through agent experience.

**Schema cache**: We probed all 766 servers at concurrency=1 to build `data/pool_schema_cache.json`. Of the 766 servers, 214 (27.9%) successfully mounted and returned tool schemas. The remaining servers failed due to auth requirements, broken deploys, or no MCP endpoint. This cache enables lazy connection: agents see tool descriptions without upfront connections.

**Data files**:
- `data/combined_server_pool.json` — 766 servers with metadata and connection configs
- `data/pool_schema_cache.json` — Tool schemas for 214 mountable servers (2,427 tools)

**Scripts**: `scripts/scrape_smithery.py`, `scripts/scrape_pulsemcp.py`, `scripts/probe_pool.py`

### 2. Task Generation

**Goal**: ~1,200 diverse tasks covering the full capability space of the 766-server MCP pool.

**Pipeline** (5 steps):

1. **Embed MCP descriptions** — Each server's name + description is embedded with `text-embedding-3-small`. Cached to `data/mcp_embeddings.npz`.

2. **Cluster MCPs** — K-means (k=15) over the embeddings groups servers by capability. Cluster sizes range from 13 to 204 (median 53), covering domains like search/web, finance, developer tools, social media, code security, and more.

3. **Generate tasks per cluster** — GPT-5-mini generates tasks proportional to cluster size (minimum 10 per cluster). Each prompt includes up to 15 representative server descriptions from the cluster. Tasks are tool-agnostic natural language requests with varied complexity and specificity. Temperature=0.9 for diversity. Two clusters required handwritten tasks due to repeated LLM generation failures — 30 tasks were manually authored. Produced 3,810 raw tasks total.

4. **Validate retrievability** — Each task is embedded and checked against the MCP pool via cosine similarity. Tasks whose best-matching MCP falls below a similarity threshold of 0.3 are dropped, ensuring every task has at least one plausibly relevant tool.

5. **Deduplicate** — Near-duplicate tasks (embedding cosine similarity > 0.88) are removed.

| Stage | Count |
|-------|-------|
| Raw generated | 3,810 |
| After validation + dedup | 1,191 |
| Clusters covered | 15/15 |
| Train split | 954 (80%) |
| Test split | 237 (20%) |

Train/test split is stratified by cluster to ensure proportional coverage across all 15 capability domains.

**Final task similarity to MCP pool**: min=0.300, median=0.407, max=0.823.

Each task record includes `query`, `cluster_id`, `cluster_size`, `top_mcp_similarity`, and a unique `uuid`.

**Data files**:
- `data/mcp_embeddings.npz` — MCP description embeddings (766 servers)
- `data/mcp_clusters.json` — 15 capability clusters
- `data/tasks_raw.json` — 3,810 pre-validation tasks
- `data/tasks.json` — 1,191 final tasks
- `data/tasks_train.json` — 954 training tasks
- `data/tasks_test.json` — 237 held-out test tasks

**Script**: `scripts/generate_tasks.py`

### 3. Recommendation Loop

Each rollout executes one complete recommendation-execution-feedback cycle:

```
For each rollout:
  1. Sample a task from the task pool
  2. Sample an agent from the 5 agents
  3. Retrieve -> Rerank -> Select (766 -> 100 -> K=5):
     a. RETRIEVE: Embed task, cosine similarity against 766 MCP embeddings
        -> top 100 candidates (coarse semantic filter)
     b. RERANK: Recommender scores 100 candidates for this (agent, task) pair
     c. SELECT: Take top K=5 from reranked list
  4. Load tool schemas from cache (no upfront connections)
  5. Inject tool schemas into agent prompt + the task
  6. Agent executes (up to 5 tool-use turns):
     - Reads tool descriptions, selects tools, calls them
     - On first call to a server, lazy connection fires (Smithery Connect)
     - If connection fails: server demoted to inventory_failed, -3.0 signal
     - If error/bad result, may retry with another tool
     - Produces an answer
  7. Feedback collection (second LLM call):
     - If tools were used: agent rates each tool (liked/neutral/disliked)
     - If no tools used: agent reports whether offered tools were relevant
  8. Extract training signals from rollout
  9. Update recommender (online SGD)
  10. Log everything to JSONL
```

#### Lazy Connection Architecture

Servers in the schema cache have their tool descriptions injected into the agent prompt without making a live connection. A connection fires only when the agent actually calls a tool from that server. This avoids thousands of upfront connections per rollout and provides cleaner failure attribution: if a server fails at call time, it is demoted from `inventory_mounted` to `inventory_failed` with a strong negative signal.

Fallback servers (Tavily search, Exa search) are always eagerly connected since they are not in the pool cache.

#### Agents (5 models across 3 providers)

| Agent | Provider | Model ID |
|-------|----------|----------|
| Llama 4 Maverick | OpenRouter | `meta-llama/llama-4-maverick` |
| GPT-4o-mini | OpenAI | `gpt-4o-mini` |
| GPT-5-mini | OpenAI | `gpt-5-mini` |
| Gemini 2.5 Flash Lite | OpenRouter | `google/gemini-2.5-flash-lite` |
| Grok 4 Fast | OpenRouter | `x-ai/grok-4-fast` |

Different model families = different "user personas" with different tool-handling strategies, preferences, and feedback quality.

#### Signal Hierarchy

Feedback is converted to per-server training signals:

| Signal | Value | Source |
|--------|-------|--------|
| Liked | +1.0 | Explicit feedback |
| Neutral | +0.2 | Explicit feedback |
| Selected (no rating) | +0.1 | CTR (agent chose the tool) |
| Mounted, not selected | -0.1 | Weak negative CTR |
| Tools irrelevant | -0.3 | Relevance check |
| Abandoned | -0.4 | Agent tried then switched |
| Errored | -0.5 | Tool call returned error |
| Disliked | -1.0 | Explicit feedback |
| Mount failed | -3.0 | Server failed lazy connect |

When multiple tools from the same server are used in one rollout, signals are mean-aggregated per server before the SGD update.

#### Training Schedule

- **2 epochs** over 954 training tasks × 5 agents = **9,540 total rollouts**
- Epoch 1: model is cold, mostly semantic retrieval + exploration
- Epoch 2: model is warm, routes around known-bad servers, epsilon-greedy exploration (10%) surfaces missed servers
- Online SGD updates after each rollout
- Task order is shuffled per epoch; agent is sampled independently per task

### 4. Recommender Models

#### Baselines

**Semantic**: Ranks candidates by cosine similarity between task embedding and server description embedding, with popularity tiebreaking. This is what MCP marketplaces do today.

**Semantic + Popularity**: Weighted blend of cosine similarity and log-normalized popularity.
```
score = 0.7 * semantic_sim + 0.3 * pop_score
pop_score = avg(log1p(use_count) / log1p(2M), log1p(stars) / log1p(50K))
```

#### Latent Factor Model (ours)

```
score(agent_a, server_s, task_t) = beta_s + gamma_a . gamma_s + (P . task_emb_t) . epsilon_s
```

- **beta_s**: server bias (scalar per server) — captures "this server is broken/good"
- **gamma_a · gamma_s**: agent-server affinity (16-dim latent) — agent preferences
- **P**: shared projection matrix (16 × 1536) — compresses task embedding to latent space
- **epsilon_s**: server task-affinity (16-dim per server) — "good for this type of task"

Task embedding is frozen (`text-embedding-3-small`). Only beta, gamma, epsilon, P are learned via SGD.

**Cold start blending**: The latent factor score is blended with the semantic similarity score, with the collaborative signal ramping up as observations accumulate:
```
final_score = semantic_sim + alpha * clamp(predicted, -1, 1)
alpha = min(n_observations / 200, 0.6)
```

**SGD updates** for each observation `(agent_a, server_s, task_t, target_y)`:
```
task_proj = P . task_emb_t
predicted = beta_s + gamma_a . gamma_s + task_proj . epsilon_s
error     = y - predicted

beta_s    += lr * (error - reg * beta_s)
gamma_a   += lr * (error * gamma_s - reg * gamma_a)
gamma_s   += lr * (error * gamma_a - reg * gamma_s)
epsilon_s += lr * (error * task_proj - reg * epsilon_s)
P         += lr * (error * outer(epsilon_s, task_emb_t) - reg_P * P)
```

Hyperparameters: `latent_dim=16, lr=0.01, reg=0.001, reg_P=0.0001`.

**Model state after training**: 4,372 observations across 748 servers and 5 agents.

### 5. Evaluation

**Test set**: 150 sampled held-out tasks from `data/tasks_test.json` (stratified from the 237 test tasks).

**Protocol**: Each test task is run with all 5 agents across all recommender methods. The recommender is frozen during evaluation — no online updates.

**Methods compared**:
1. Semantic search only
2. Semantic + Popularity blend
3. Latent Factor (frozen, trained model)

**Ablation**: Same 3 methods run with the pool filtered to the 214 confirmed-mountable servers from `data/pool_schema_cache.json`. Measures recommender value when the ecosystem noise (broken tools) is removed.

**Metrics** (all derived from agent behavior, no ground truth):
- **Like rate**: fraction of rollouts with at least one "liked" tool
- **Dislike rate**: fraction of rollouts with at least one "disliked" tool
- **Mount success rate**: fraction of recommended servers that successfully connect
- **Tool use rate**: fraction of rollouts where the agent used at least one tool
- **Retry rate**: how often the agent abandons first tool choice

## Training Results

After ~4,900 training rollouts (epoch 1 + partial epoch 2):

| Metric | Overall | Recent 500 |
|--------|---------|------------|
| Like rate | 35.4% | 37.6% |
| Dislike rate | 19.4% | 20.8% |
| Mount failed rate | 48.1% | 16.8% |

Mount failure rate dropping from ~48% overall to ~17% in recent rollouts shows the recommender is actively learning to route around broken servers. The like rate ceiling (~40%) reflects the real-world ecosystem: only 214/766 servers (27.9%) are actually mountable, and even mountable servers may be auth-gated or task-mismatched at runtime.

## Directory Structure

```
MCPToolBenchPP/
├── CLAUDE.md                        # Experiment design doc
├── README.md                        # This file
├── data/
│   ├── combined_server_pool.json    # 766 servers with full metadata
│   ├── pool_schema_cache.json       # Tool schemas for 214 mountable servers
│   ├── mcp_embeddings.npz           # Server description embeddings (766 servers)
│   ├── mcp_clusters.json            # 15 capability clusters
│   ├── tasks_raw.json               # 3,810 pre-validation tasks
│   ├── tasks.json                   # 1,191 final tasks
│   ├── tasks_train.json             # 954 training tasks
│   ├── tasks_test.json              # 237 held-out test tasks
│   └── tasks_smoke.json             # Small smoke-test subset
├── src/recsys/
│   ├── config.py                    # API keys, agent configs, paths
│   ├── pipeline.py                  # Rollout orchestrator (lazy connection)
│   ├── agent_client.py              # Anthropic/OpenAI/OpenRouter agent wrapper
│   ├── feedback.py                  # Second-prompt feedback collection
│   ├── mcp_client.py                # Smithery Connect + MCP client (429 retry)
│   ├── retriever.py                 # Embedding-based semantic retrieval
│   ├── run_train.py                 # Training loop runner
│   ├── run_eval.py                  # Held-out evaluation runner
│   └── recommenders/
│       ├── base.py                  # Abstract recommender interface
│       ├── random_baseline.py       # Random from candidates
│       ├── popularity.py            # Rank by use_count + stars
│       ├── semantic.py              # Cosine similarity + popularity tiebreak
│       ├── semantic_popularity.py   # Weighted semantic + popularity blend
│       └── latent_factor.py         # Latent factor with projected task embeddings
├── scripts/
│   ├── scrape_smithery.py           # Smithery registry scraper
│   ├── scrape_pulsemcp.py           # PulseMCP directory scraper
│   ├── probe_pool.py                # Probe all servers and cache tool schemas
│   └── generate_tasks.py            # Task generation pipeline
└── results/
    ├── train/
    │   ├── train_rollouts.jsonl     # All training rollouts with feedback
    │   └── model_checkpoint.json    # Saved latent factor model state
    └── eval/                        # Evaluation JSONL files per method
```

## Rollout Log Format

```jsonl
{
  "rollout_id": 1042,
  "agent": "gpt-5-mini",
  "task_id": "uuid-...",
  "task_query": "Find the most cited paper on attention mechanisms from 2023",
  "task_category": "search_web",
  "epoch": 1,

  "stage_retrieve": {"pool_size": 766, "candidates_returned": 100, "method": "cosine_similarity"},
  "stage_rerank": {"candidates_in": 100, "method": "latent_factor", "top_k": 5},

  "inventory_mounted": ["server-A", "server-B"],
  "inventory_failed": ["server-C", "server-D", "server-E"],

  "tools_selected": ["server-A:search"],
  "tools_abandoned": [],
  "tools_errored": [],
  "tool_used_final": "server-A:search",

  "agent_answer": "...",

  "feedback": {
    "server-A:search": {"rating": "liked", "reason": "Returned structured citation data"}
  },

  "latency_s": 4.2,
  "input_tokens": 1400,
  "output_tokens": 700,
  "cost_usd": 0.006
}
```

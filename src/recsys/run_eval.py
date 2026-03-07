"""Held-out evaluation: run all recommender baselines on test tasks.

Usage:
    python -m src.exp2.run_eval --concurrency 8 --budget 30

Runs 4 recommenders (random, popularity, semantic, latent_factor) on 162 test tasks × 3 agents.
Latent factor model is loaded from training checkpoint (frozen — no online updates).
Outputs separate JSONL per recommender + comparison summary.
"""

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from .config import AGENTS, DEFAULT_K, DEFAULT_RETRIEVE_N
from .pipeline import run_rollout
from .recommenders.latent_factor import LatentFactorRecommender
from .recommenders.popularity import PopularityRecommender
from .recommenders.random_baseline import RandomRecommender
from .recommenders.semantic import SemanticRecommender
from .recommenders.semantic_popularity import SemanticPopularityRecommender
from .recommenders.tucker import TuckerRecommender
from .retriever import precompute_pool_embeddings, precompute_query_embeddings

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent

COST_TABLE = {
    "haiku-4.5": {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


def estimate_cost(rollout: dict) -> float:
    agent = rollout.get("agent", "")
    rates = COST_TABLE.get(agent, {"input": 1.0, "output": 5.0})
    inp = rollout.get("input_tokens", 0) or 0
    out = rollout.get("output_tokens", 0) or 0
    return (inp * rates["input"] + out * rates["output"]) / 1_000_000


def build_eval_schedule(tasks: list[dict], agents: list[str], seed: int) -> list[tuple[dict, str]]:
    """Every task × every agent = one evaluation rollout."""
    schedule = []
    for task in tasks:
        for agent in agents:
            schedule.append((task, agent))
    # Shuffle for even agent distribution during concurrent runs
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


class EvalRunner:
    def __init__(self, pool, tasks, k, retrieve_n, concurrency, budget, output_dir, seed, agents=None, methods=None):
        self.pool = pool
        self.tasks = tasks
        self.k = k
        self.retrieve_n = retrieve_n
        self.concurrency = concurrency
        self.budget = budget
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.agents = agents    # None = all agents
        self.methods = methods  # None = all methods

        # Shared state
        self.total_cost = 0.0
        self._lock = asyncio.Lock()
        self._budget_exceeded = False

        # Precomputed embeddings (set in run())
        self.pool_emb_matrix = None
        self.pool_entries = None
        self.query_embs = None

    def _build_recommenders(self) -> dict:
        """Build all recommender instances."""
        recs = {
            "random": RandomRecommender(seed=self.seed),
            "popularity": PopularityRecommender(),
            "semantic": SemanticRecommender(),
            "semantic_popularity": SemanticPopularityRecommender(),
            "latent_factor": self._load_trained_model(),
            "tucker": self._load_tucker_model(),
        }
        return recs

    def _load_trained_model(self) -> LatentFactorRecommender:
        """Load trained latent factor model from checkpoint."""
        # Prefer CTR-penalized version
        model_path = ROOT / "results" / "latent_factor_ctr_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v10_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v9_scaled.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v8_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v7_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v5_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v4_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v3_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_v2_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "latent_factor_trained.json"
        model = LatentFactorRecommender(seed=self.seed)
        model.load(str(model_path))
        return model

    def _load_tucker_model(self) -> TuckerRecommender:
        """Load trained Tucker model from checkpoint. Prefers CTR-penalized version."""
        model_path = ROOT / "results" / "tucker_ctr_trained.json"
        if not model_path.exists():
            model_path = ROOT / "results" / "tucker_trained.json"
        model = TuckerRecommender(seed=self.seed)
        if model_path.exists():
            model.load(str(model_path))
        else:
            log.warning(f"Tucker model not found at {model_path} — using untrained model")
        return model

    def _count_existing(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _load_done_pairs(self, path: Path) -> set:
        """Return set of (task_id, agent) tuples already in the output file."""
        done = set()
        if not path.exists():
            return done
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        tid = r.get("task_id", "")
                        agent = r.get("agent", "")
                        if tid and agent:
                            done.add((tid, agent))
                    except Exception:
                        pass
        return done

    async def _run_one(
        self, idx, task, agent, recommender, output_path, sem, file_lock, rec_name
    ):
        if self._budget_exceeded:
            return

        async with sem:
            if self._budget_exceeded:
                return

            task_query = task.get("query", task.get("question", ""))
            task_id = task.get("uuid", f"task_{idx}")
            task_category = task.get("category", "")
            query_emb = self.query_embs.get(task_id) if self.query_embs else None

            try:
                rollout = await run_rollout(
                    rollout_id=idx,
                    task_query=task_query,
                    task_id=task_id,
                    agent_name=agent,
                    k=self.k,
                    recommender=recommender,
                    retrieve_n=self.retrieve_n,
                    use_fallbacks=False,
                    pool=self.pool,
                    pool_emb_matrix=self.pool_emb_matrix,
                    pool_entries=self.pool_entries,
                    query_emb=query_emb,
                    task_category=task_category,
                )
            except Exception as e:
                log.error(f"[{rec_name} #{idx}] Fatal: {e}")
                rollout = {
                    "rollout_id": idx, "agent": agent,
                    "task_id": task_id, "task_query": task_query,
                    "error": str(e), "latency_s": 0,
                }

            cost = estimate_cost(rollout)
            rollout["cost_usd"] = round(cost, 6)
            rollout["recommender"] = rec_name

            async with self._lock:
                self.total_cost += cost
                if self.total_cost >= self.budget:
                    self._budget_exceeded = True
                    log.warning(f"Budget exceeded: ${self.total_cost:.2f}")

            # NO online update — evaluation is frozen
            async with file_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(rollout, default=str) + "\n")

    async def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Precompute embeddings once
        log.info("Precomputing pool embeddings...")
        self.pool_emb_matrix, self.pool_entries = precompute_pool_embeddings(self.pool)
        log.info("Precomputing query embeddings...")
        self.query_embs = precompute_query_embeddings(self.tasks)

        agents = self.agents if self.agents else list(AGENTS.keys())
        schedule = build_eval_schedule(self.tasks, agents, self.seed)
        log.info(f"Eval schedule: {len(schedule)} rollouts per recommender "
                 f"({len(self.tasks)} tasks × {len(agents)} agents)")

        recommenders = self._build_recommenders()
        if self.methods:
            recommenders = {k: v for k, v in recommenders.items() if k in self.methods}
            log.info(f"Running methods: {list(recommenders.keys())}")

        for rec_name, recommender in recommenders.items():
            if self._budget_exceeded:
                log.warning(f"Skipping {rec_name} — budget exceeded")
                break

            output_path = self.output_dir / f"eval_{rec_name}.jsonl"
            done_pairs = self._load_done_pairs(output_path)
            existing = len(done_pairs) if done_pairs else self._count_existing(output_path)

            remaining = [
                (i, task, agent) for i, (task, agent) in enumerate(schedule)
                if (task.get("uuid", f"task_{i}"), agent) not in done_pairs
            ]
            if not remaining:
                log.info(f"[{rec_name}] Already complete ({existing} rollouts)")
                continue

            log.info(f"[{rec_name}] Running {len(remaining)} eval rollouts "
                     f"(existing={existing}, method={recommender.method_name})")

            sem = asyncio.Semaphore(self.concurrency)
            file_lock = asyncio.Lock()
            t0 = time.time()

            tasks = [
                self._run_one(
                    orig_i, task, agent, recommender,
                    output_path, sem, file_lock, rec_name,
                )
                for orig_i, task, agent in remaining
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = time.time() - t0
            count = self._count_existing(output_path)
            log.info(f"[{rec_name}] Done: {count} rollouts in {elapsed/60:.1f}min")

        log.info(f"\nEval complete. Total cost: ${self.total_cost:.2f}")
        log.info("Run analysis: python -m src.exp2.analyze_eval")


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation runner")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieve-n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default="results/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent names, e.g. 'gpt-5-mini'. Defaults to all agents.")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated recommender names, e.g. 'random,semantic_popularity,latent_factor'.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load pool and filter to functional servers only
    # This ensures fair comparison: baselines don't get penalized for
    # recommending servers that always fail (missing API keys, broken deploys).
    # The latent_factor model learned to avoid these, giving it an unfair edge.
    pool_path = Path(args.pool) if args.pool else ROOT / "data" / "pool" / "combined_pool.json"
    with open(pool_path) as f:
        pool = json.load(f)

    functional_path = ROOT / "data" / "functional_servers.json"
    if functional_path.exists():
        with open(functional_path) as f:
            functional_ids = set(json.load(f))
        original = len(pool)
        pool = [s for s in pool if s["id"] in functional_ids]
        log.info(f"Pool: {len(pool)} functional servers (filtered from {original}, removed {original - len(pool)} broken)")
    else:
        log.info(f"Pool: {len(pool)} servers (no functional filter)")

    # Load test tasks: prefer tasks_combined.json filtered by test_task_uuids.json
    if args.tasks:
        tasks_path = Path(args.tasks)
        with open(tasks_path) as f:
            tasks = json.load(f)
    else:
        combined_path = ROOT / "data" / "tasks_combined.json"
        uuids_path = ROOT / "data" / "test_task_uuids.json"
        if combined_path.exists() and uuids_path.exists():
            with open(combined_path) as f:
                all_tasks = json.load(f)
            with open(uuids_path) as f:
                test_uuids = set(json.load(f))
            tasks = [t for t in all_tasks if t.get("uuid") in test_uuids]
            log.info(f"Loaded {len(tasks)} test tasks from tasks_combined.json (filtered by test_task_uuids.json)")
        else:
            with open(ROOT / "data" / "tasks_test.json") as f:
                tasks = json.load(f)
    log.info(f"Test tasks: {len(tasks)}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    agents = [a.strip() for a in args.agents.split(",")] if args.agents else None
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    runner = EvalRunner(
        pool=pool, tasks=tasks, k=args.k, retrieve_n=args.retrieve_n,
        concurrency=args.concurrency, budget=args.budget,
        output_dir=output_dir, seed=args.seed, agents=agents, methods=methods,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()

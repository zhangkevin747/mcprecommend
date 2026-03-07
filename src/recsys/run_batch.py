"""Batch experiment runner: execute N rollouts with parallelism, cost tracking, and crash recovery.

Usage:
    python -m src.exp2.run_batch --total 1000 --k 5 --concurrency 4 \
        --budget 50.0 --output results/rollouts_train.jsonl \
        --recommender semantic --seed 42
"""

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from .config import AGENTS, DEFAULT_K, DEFAULT_RETRIEVE_N, MCP_POOL_PATH
from .pipeline import run_rollout
from .recommenders.latent_factor import LatentFactorRecommender
from .recommenders.popularity import PopularityRecommender
from .recommenders.random_baseline import RandomRecommender
from .recommenders.semantic import SemanticRecommender
from .retriever import precompute_pool_embeddings, precompute_query_embeddings

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# Cost per 1M tokens (USD)
COST_TABLE = {
    "haiku-4.5":  {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
}

RECOMMENDERS = {
    "semantic": SemanticRecommender,
    "random": RandomRecommender,
    "popularity": PopularityRecommender,
    "latent_factor": LatentFactorRecommender,
}


def estimate_cost(rollout: dict) -> float:
    """Estimate USD cost from token counts."""
    agent = rollout.get("agent", "")
    rates = COST_TABLE.get(agent, {"input": 1.0, "output": 5.0})
    inp = rollout.get("input_tokens", 0) or 0
    out = rollout.get("output_tokens", 0) or 0
    feedback_tokens = rollout.get("feedback", {})
    # Feedback tokens are included in input/output already
    return (inp * rates["input"] + out * rates["output"]) / 1_000_000


def build_schedule(
    tasks: list[dict],
    agents: list[str],
    total: int,
    seed: int,
) -> list[tuple[dict, str]]:
    """Build deterministic (task, agent) schedule.

    Round-robin agents, random task sampling without replacement
    (reshuffles and cycles after exhausting all tasks).
    """
    rng = random.Random(seed)
    task_indices = list(range(len(tasks)))
    rng.shuffle(task_indices)

    schedule = []
    task_pos = 0
    for i in range(total):
        if task_pos >= len(task_indices):
            rng.shuffle(task_indices)
            task_pos = 0
        task = tasks[task_indices[task_pos]]
        agent = agents[i % len(agents)]
        schedule.append((task, agent))
        task_pos += 1

    return schedule


class BatchRunner:
    def __init__(
        self,
        pool: list[dict],
        tasks: list[dict],
        total: int,
        k: int,
        retrieve_n: int,
        concurrency: int,
        budget: float,
        output_path: Path,
        recommender_name: str,
        seed: int,
        use_fallbacks: bool = False,
    ):
        self.pool = pool
        self.tasks = tasks
        self.total = total
        self.k = k
        self.retrieve_n = retrieve_n
        self.concurrency = concurrency
        self.budget = budget
        self.output_path = output_path
        self.seed = seed
        self.use_fallbacks = use_fallbacks

        # Build recommender
        rec_cls = RECOMMENDERS.get(recommender_name)
        if rec_cls is None:
            raise ValueError(f"Unknown recommender: {recommender_name}. Options: {list(RECOMMENDERS)}")
        if recommender_name in ("random", "latent_factor"):
            self.recommender = rec_cls(seed=seed)
        else:
            self.recommender = rec_cls()

        # State
        self.total_cost = 0.0
        self.completed = 0
        self.errors = 0
        self._lock = asyncio.Lock()
        self._file_lock = asyncio.Lock()
        self._budget_exceeded = False

        # Precomputed embeddings (set in run())
        self.pool_emb_matrix = None
        self.pool_entries = None
        self.query_embs = None

    def _count_existing(self) -> int:
        """Count lines in existing output file for crash recovery."""
        if not self.output_path.exists():
            return 0
        count = 0
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
        return count

    async def _write_result(self, rollout: dict):
        """Append one rollout to JSONL output."""
        async with self._file_lock:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(rollout, default=str) + "\n")

    async def _run_one(self, idx: int, task: dict, agent: str, sem: asyncio.Semaphore):
        """Run a single rollout under semaphore."""
        if self._budget_exceeded:
            return

        async with sem:
            if self._budget_exceeded:
                return

            task_query = task.get("query", task.get("question", ""))
            task_id = task.get("uuid", f"task_{idx}")
            task_category = task.get("category", "")

            # Get precomputed query embedding
            query_emb = self.query_embs.get(task_id) if self.query_embs else None

            try:
                rollout = await run_rollout(
                    rollout_id=idx,
                    task_query=task_query,
                    task_id=task_id,
                    agent_name=agent,
                    k=self.k,
                    recommender=self.recommender,
                    retrieve_n=self.retrieve_n,
                    use_fallbacks=self.use_fallbacks,
                    pool=self.pool,
                    pool_emb_matrix=self.pool_emb_matrix,
                    pool_entries=self.pool_entries,
                    query_emb=query_emb,
                    task_category=task_category,
                )
            except Exception as e:
                log.error(f"[Rollout {idx}] Fatal error: {e}")
                rollout = {
                    "rollout_id": idx,
                    "agent": agent,
                    "task_id": task_id,
                    "task_query": task_query,
                    "error": str(e),
                    "latency_s": 0,
                }

            # Cost tracking
            cost = estimate_cost(rollout)
            rollout["cost_usd"] = round(cost, 6)

            async with self._lock:
                self.total_cost += cost
                self.completed += 1
                if rollout.get("error"):
                    self.errors += 1

                if self.total_cost >= self.budget:
                    self._budget_exceeded = True
                    log.warning(f"Budget exceeded: ${self.total_cost:.2f} >= ${self.budget:.2f}")

            await self._write_result(rollout)

            # Online learning: update recommender with rollout feedback
            async with self._lock:
                self.recommender.update(rollout, task_emb=query_emb)

            # Progress logging every 10 rollouts
            if self.completed % 10 == 0 or self.completed == self.total:
                elapsed = time.time() - self._start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.completed) / rate if rate > 0 else 0
                log.info(
                    f"Progress: {self.completed}/{self.total} "
                    f"({100 * self.completed / self.total:.0f}%) | "
                    f"Cost: ${self.total_cost:.2f}/${self.budget:.0f} | "
                    f"Errors: {self.errors} | "
                    f"Rate: {rate:.1f}/min | "
                    f"ETA: {remaining / 60:.0f}min"
                )

    async def run(self):
        """Execute the full batch."""
        # Crash recovery
        existing = self._count_existing()
        if existing > 0:
            log.info(f"Crash recovery: found {existing} existing rollouts, resuming from {existing}")
            # Replay existing rollouts through recommender for model warm-up
            with open(self.output_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.recommender.update(json.loads(line))
            log.info(f"Model warm-up: replayed {existing} rollouts")

        # Build schedule
        agents = list(AGENTS.keys())
        schedule = build_schedule(self.tasks, agents, self.total, self.seed)

        # Skip already completed
        schedule = schedule[existing:]
        if not schedule:
            log.info("All rollouts already completed!")
            return

        log.info(f"Batch: {len(schedule)} rollouts to run (total={self.total}, existing={existing})")
        log.info(f"Config: k={self.k}, retrieve_n={self.retrieve_n}, concurrency={self.concurrency}")
        log.info(f"Budget: ${self.budget:.0f}, Recommender: {type(self.recommender).__name__}")

        # Precompute embeddings
        log.info("Precomputing pool embeddings...")
        self.pool_emb_matrix, self.pool_entries = precompute_pool_embeddings(self.pool)

        log.info("Precomputing query embeddings...")
        self.query_embs = precompute_query_embeddings(self.tasks)

        # Run
        sem = asyncio.Semaphore(self.concurrency)
        self._start_time = time.time()

        tasks = [
            self._run_one(existing + i, task, agent, sem)
            for i, (task, agent) in enumerate(schedule)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - self._start_time
        log.info(
            f"\nBatch complete: {self.completed} rollouts in {elapsed / 60:.1f}min | "
            f"Cost: ${self.total_cost:.2f} | Errors: {self.errors}"
        )

        # Write server stats
        self._write_server_stats()

    def _write_server_stats(self):
        """Analyze rollouts and write per-server success/failure stats."""
        if not self.output_path.exists():
            return

        server_stats: dict[str, dict] = {}
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for sid in r.get("inventory_mounted", []):
                    stats = server_stats.setdefault(sid, {"mounted": 0, "selected": 0, "errored": 0, "liked": 0, "disliked": 0})
                    stats["mounted"] += 1
                for sid in r.get("inventory_failed", []):
                    stats = server_stats.setdefault(sid, {"mounted": 0, "selected": 0, "errored": 0, "liked": 0, "disliked": 0})
                    stats["errored"] += 1
                for tool in r.get("tools_selected", []):
                    sid = tool.split(":")[0] if ":" in tool else tool
                    if sid in server_stats:
                        server_stats[sid]["selected"] += 1
                for tool, fb in r.get("feedback", {}).items():
                    sid = tool.split(":")[0] if ":" in tool else tool
                    if sid not in server_stats:
                        continue
                    rating = fb.get("rating", "") if isinstance(fb, dict) else ""
                    if rating == "liked":
                        server_stats[sid]["liked"] += 1
                    elif rating == "disliked":
                        server_stats[sid]["disliked"] += 1

        stats_path = self.output_path.parent / "server_stats.json"
        with open(stats_path, "w") as f:
            json.dump(server_stats, f, indent=2, sort_keys=True)
        log.info(f"Server stats: {len(server_stats)} servers → {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--total", type=int, default=1000, help="Total rollouts to run")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Servers to mount per rollout")
    parser.add_argument("--retrieve-n", type=int, default=DEFAULT_RETRIEVE_N, help="Retrieval candidates")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel rollouts")
    parser.add_argument("--budget", type=float, default=50.0, help="Max USD spend")
    parser.add_argument("--output", type=str, default="results/rollouts_train.jsonl", help="Output JSONL path")
    parser.add_argument("--recommender", type=str, default="semantic", choices=list(RECOMMENDERS), help="Recommender strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pool", type=str, default=None, help="Pool JSON path (default: config MCP_POOL_PATH)")
    parser.add_argument("--tasks", type=str, default=None, help="Tasks JSON path")
    parser.add_argument("--use-fallbacks", action="store_true", help="Include fallback servers")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load pool
    pool_path = Path(args.pool) if args.pool else MCP_POOL_PATH
    if not pool_path.exists():
        # Try combined pool
        pool_path = ROOT / "data" / "pool" / "combined_pool.json"
    with open(pool_path) as f:
        pool = json.load(f)
    log.info(f"Pool: {len(pool)} servers from {pool_path}")

    # Load tasks
    if args.tasks:
        tasks_path = Path(args.tasks)
    else:
        tasks_path = ROOT / "data" / "tasks_combined.json"
    with open(tasks_path) as f:
        tasks = json.load(f)
    log.info(f"Tasks: {len(tasks)} from {tasks_path}")

    # Output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runner = BatchRunner(
        pool=pool,
        tasks=tasks,
        total=args.total,
        k=args.k,
        retrieve_n=args.retrieve_n,
        concurrency=args.concurrency,
        budget=args.budget,
        output_path=output_path,
        recommender_name=args.recommender,
        seed=args.seed,
        use_fallbacks=args.use_fallbacks,
    )

    asyncio.run(runner.run())


if __name__ == "__main__":
    main()

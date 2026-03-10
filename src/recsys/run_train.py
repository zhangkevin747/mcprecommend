"""Training runner: online SGD over training tasks with epsilon-greedy exploration.

Runs 2 epochs over 954 training tasks × 5 agents = 9,540 rollouts.
After each rollout, updates the latent factor model with the observed signals.
Saves a checkpoint after each epoch.

Usage:
    python -m src.recsys.run_train --concurrency 4 --epsilon 0.1
    python -m src.recsys.run_train --concurrency 4 --epsilon 0.1 --epochs 2
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
from .recommenders.latent_factor_mtl import LatentFactorMTLRecommender
from .retriever import precompute_pool_embeddings, precompute_query_embeddings

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent

COST_TABLE = {
    "llama-4-maverick": {"input": 0.80, "output": 4.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "grok-4-fast": {"input": 0.20, "output": 0.50},
    "deepseek-v3.2": {"input": 0.30, "output": 0.88},
}


def estimate_cost(rollout: dict) -> float:
    agent = rollout.get("agent", "")
    rates = COST_TABLE.get(agent, {"input": 1.0, "output": 5.0})
    inp = rollout.get("input_tokens", 0) or 0
    out = rollout.get("output_tokens", 0) or 0
    return (inp * rates["input"] + out * rates["output"]) / 1_000_000


def build_schedule(tasks: list[dict], agents: list[str], seed: int) -> list[tuple[dict, str]]:
    """Every task × every agent = one training rollout."""
    schedule = []
    for task in tasks:
        for agent in agents:
            schedule.append((task, agent))
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


class TrainingRunner:
    def __init__(self, pool, tasks, k, retrieve_n, concurrency, budget, output_dir,
                 seed, epochs, epsilon, agents=None, model_type="latent_factor"):
        self.pool = pool
        self.tasks = tasks
        self.k = k
        self.retrieve_n = retrieve_n
        self.concurrency = concurrency
        self.budget = budget
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.epochs = epochs
        self.epsilon = epsilon
        self.agents = agents or list(AGENTS.keys())
        self.model_type = model_type

        if model_type == "latent_factor_mtl":
            self.recommender = LatentFactorMTLRecommender(seed=seed)
        else:
            self.recommender = LatentFactorRecommender(seed=seed)
        self.total_cost = 0.0
        self._lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()
        self._budget_exceeded = False
        self._completed = 0
        self._liked = 0
        self._feedback_count = 0
        self._mount_success = 0
        self._mount_total = 0
        self._progress_start = None

        self.pool_emb_matrix = None
        self.pool_entries = None
        self.query_embs = None

    def _load_done_ids(self, path: Path) -> set[str]:
        """Return set of (task_id, agent, epoch) strings already logged."""
        done = set()
        if not path.exists():
            return done
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = f"{r.get('task_id')}|{r.get('agent')}|{r.get('epoch', 0)}"
                    done.add(key)
                except Exception:
                    pass
        return done

    async def _run_one(self, rollout_id, task, agent, epoch, output_path, sem, file_lock):
        if self._budget_exceeded:
            return

        async with sem:
            if self._budget_exceeded:
                return

            task_query = task.get("query", task.get("question", ""))
            task_id = task.get("uuid") or task.get("task_id", f"task_{rollout_id}")
            task_category = task.get("category", "")
            query_emb = self.query_embs.get(task_id) if self.query_embs else None

            try:
                rollout = await run_rollout(
                    rollout_id=rollout_id,
                    task_query=task_query,
                    task_id=task_id,
                    agent_name=agent,
                    k=self.k,
                    recommender=self.recommender,
                    retrieve_n=self.retrieve_n,
                    use_fallbacks=False,
                    pool=self.pool,
                    pool_emb_matrix=self.pool_emb_matrix,
                    pool_entries=self.pool_entries,
                    query_emb=query_emb,
                    task_category=task_category,
                    epsilon=self.epsilon,
                )
            except Exception as e:
                log.error(f"[Train #{rollout_id}] Fatal: {e}")
                rollout = {
                    "rollout_id": rollout_id, "agent": agent,
                    "task_id": task_id, "task_query": task_query,
                    "error": str(e), "latency_s": 0,
                }

            rollout["epoch"] = epoch
            rollout["cost_usd"] = round(estimate_cost(rollout), 6)

            # Online update — serialized to avoid concurrent SGD writes
            async with self._update_lock:
                self.recommender.update(rollout, task_emb=query_emb)

            async with self._lock:
                self.total_cost += rollout["cost_usd"]
                if self.total_cost >= self.budget:
                    self._budget_exceeded = True
                    log.warning(f"Budget exceeded: ${self.total_cost:.2f}")

                # Track progress stats
                self._completed += 1
                fb = rollout.get("feedback", {})
                if fb and "tools_relevant" not in fb:
                    self._feedback_count += 1
                    for v in fb.values():
                        if isinstance(v, dict) and v.get("rating") == "liked":
                            self._liked += 1
                mounted = len(rollout.get("inventory_mounted", []))
                total = mounted + len(rollout.get("inventory_failed", []))
                self._mount_success += mounted
                self._mount_total += total

                if self._completed % 100 == 0:
                    elapsed = time.time() - self._progress_start
                    rate = self._completed / elapsed if elapsed > 0 else 0
                    like_rate = self._liked / self._feedback_count if self._feedback_count else 0
                    mount_rate = self._mount_success / self._mount_total if self._mount_total else 0
                    log.info(
                        f"Progress: {self._completed} rollouts | "
                        f"like_rate={like_rate:.1%} | mount_rate={mount_rate:.1%} | "
                        f"cost=${self.total_cost:.2f} | {rate:.1f} r/s | "
                        f"model_obs={self.recommender.n_obs}"
                    )

                if self._completed % 500 == 0:
                    ckpt_path = self.output_dir / "model_checkpoint.json"
                    self.recommender.save(str(ckpt_path))
                    log.info(f"Rolling checkpoint saved: {ckpt_path}")

            async with file_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(rollout, default=str) + "\n")

    async def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "train_rollouts.jsonl"

        log.info("Precomputing pool embeddings...")
        self.pool_emb_matrix, self.pool_entries = precompute_pool_embeddings(self.pool)
        log.info("Precomputing query embeddings...")
        self.query_embs = precompute_query_embeddings(self.tasks)

        done_ids = self._load_done_ids(output_path)
        log.info(f"Resuming from {len(done_ids)} completed rollouts")

        # Load model checkpoint if it exists
        ckpt_path = self.output_dir / "model_checkpoint.json"
        if ckpt_path.exists():
            self.recommender.load(str(ckpt_path))
            log.info(f"Loaded model checkpoint: {self.recommender.n_obs} obs")

        for epoch in range(self.epochs):
            schedule = build_schedule(self.tasks, self.agents, seed=self.seed + epoch)
            remaining = [
                (i + epoch * len(schedule), task, agent)
                for i, (task, agent) in enumerate(schedule)
                if f"{task.get('uuid') or task.get('task_id')}|{agent}|{epoch}" not in done_ids
            ]

            log.info(f"Epoch {epoch + 1}/{self.epochs}: {len(remaining)} rollouts "
                     f"({len(schedule) - len(remaining)} already done), epsilon={self.epsilon}")

            if not remaining:
                log.info(f"Epoch {epoch + 1} already complete, skipping")
                continue

            sem = asyncio.Semaphore(self.concurrency)
            file_lock = asyncio.Lock()
            t0 = time.time()
            self._progress_start = t0

            tasks_coroutines = [
                self._run_one(rid, task, agent, epoch, output_path, sem, file_lock)
                for rid, task, agent in remaining
            ]
            await asyncio.gather(*tasks_coroutines, return_exceptions=True)

            elapsed = time.time() - t0
            n_obs = self.recommender.n_obs
            log.info(f"Epoch {epoch + 1} done in {elapsed / 60:.1f}min | "
                     f"model obs={n_obs} | cost so far: ${self.total_cost:.2f}")

            # Save checkpoint after each epoch
            ckpt_path = ROOT / "results" / f"{self.model_type}_epoch{epoch + 1}.json"
            self.recommender.save(str(ckpt_path))
            rolling_ckpt = self.output_dir / "model_checkpoint.json"
            self.recommender.save(str(rolling_ckpt))
            log.info(f"Checkpoint saved: {ckpt_path}")

            if self._budget_exceeded:
                log.warning("Budget exceeded — stopping early")
                break

        # Save final model
        final_path = ROOT / "results" / f"{self.model_type}_trained.json"
        self.recommender.save(str(final_path))
        log.info(f"Training complete. Final model: {final_path}")
        log.info(f"Total cost: ${self.total_cost:.2f} | Total observations: {self.recommender.n_obs}")


def main():
    parser = argparse.ArgumentParser(description="Latent factor training runner")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--retrieve-n", type=int, default=DEFAULT_RETRIEVE_N)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default="results/train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon-greedy exploration rate (0=greedy, 0.1=10% random)")
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent names. Defaults to all 5 agents.")
    parser.add_argument("--model", type=str, default="latent_factor",
                        choices=["latent_factor", "latent_factor_mtl"],
                        help="Recommender model to train.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pool_path = Path(args.pool) if args.pool else ROOT / "data" / "pool" / "combined_pool.json"
    with open(pool_path) as f:
        pool = json.load(f)
    log.info(f"Pool: {len(pool)} servers")

    if args.tasks:
        with open(args.tasks) as f:
            tasks = json.load(f)
    else:
        with open(ROOT / "data" / "tasks_train.json") as f:
            tasks = json.load(f)
    log.info(f"Training tasks: {len(tasks)}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    agents = [a.strip() for a in args.agents.split(",")] if args.agents else None

    runner = TrainingRunner(
        pool=pool, tasks=tasks, k=args.k, retrieve_n=args.retrieve_n,
        concurrency=args.concurrency, budget=args.budget,
        output_dir=output_dir, seed=args.seed,
        epochs=args.epochs, epsilon=args.epsilon, agents=agents,
        model_type=args.model,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()

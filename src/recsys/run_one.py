"""Test script: run 1 rollout end-to-end using the verified pool."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.exp2.config import POOL_DIR
from src.exp2.pipeline import run_rollout
from src.exp2.recommenders.semantic import SemanticRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main():
    # Load verified pool
    pool_path = POOL_DIR / "verified_pool.json"
    with open(pool_path) as f:
        pool = json.load(f)
    print(f"Loaded pool: {len(pool)} servers")

    # Pick a search task from the benchmark
    tasks_path = Path(__file__).resolve().parent.parent.parent / "data" / "search" / "search_0725_single_v2.json"
    with open(tasks_path) as f:
        tasks = json.load(f)

    task = tasks[0]
    task_query = task.get("query") or task.get("question") or task.get("instruction", "")
    print(f"Task: {task_query}")
    print(f"Task ID: {task.get('uuid', 'unknown')}")
    print()

    recommender = SemanticRecommender()

    rollout = await run_rollout(
        rollout_id=0,
        task_query=task_query,
        task_id=task.get("uuid", "test_0"),
        agent_name="haiku-4.5",
        k=5,
        recommender=recommender,
        retrieve_n=46,  # all pool servers as candidates
        use_fallbacks=False,  # pool already has fallbacks built in
        pool=pool,
    )

    print("\n" + "=" * 60)
    print("ROLLOUT RESULT")
    print("=" * 60)
    print(json.dumps(rollout, indent=2, default=str))

    # Save
    out = POOL_DIR / "rollout_test.json"
    with open(out, "w") as f:
        json.dump(rollout, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())

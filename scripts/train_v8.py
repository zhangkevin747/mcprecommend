"""Train LF v8: agent × server × task (W projection).

Loads training rollouts, pre-computes task embeddings one-at-a-time
(avoids token-limit issues with batching long queries), then replays
rollouts for N epochs with task_emb passed to each update() call.

Usage:
    .venv/bin/python scripts/train_v8.py --epochs 50 --output results/latent_factor_v8_trained.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.exp2.config import EMBEDDING_MODEL, OPENAI_API_KEY
from src.exp2.recommenders.latent_factor import LatentFactorRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CACHE_PATH = ROOT / "results" / "train_task_emb_cache.npz"


def load_rollouts(path: Path) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def build_task_map(rollouts: list[dict]) -> dict[str, str]:
    """Map task_id → task_query from rollouts."""
    task_map = {}
    for r in rollouts:
        tid = r.get("task_id", "")
        q = r.get("task_query", "")
        if tid and q:
            task_map[tid] = q
    return task_map


def embed_one(client: openai.OpenAI, text: str) -> np.ndarray:
    """Embed a single text string, truncating to 2000 chars to stay within token limit."""
    text = text[:2000]
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def load_or_build_cache(task_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Load cached embeddings; compute and save any that are missing."""
    cache: dict[str, np.ndarray] = {}

    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        for key in data.files:
            cache[key] = data[key]
        log.info(f"Loaded {len(cache)} cached embeddings from {CACHE_PATH}")

    missing = [tid for tid in task_map if tid not in cache]
    if not missing:
        log.info("All task embeddings cached — skipping API calls")
        return cache

    log.info(f"Computing {len(missing)} missing embeddings (one at a time)...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    for i, tid in enumerate(missing):
        query = task_map[tid]
        emb = embed_one(client, query)
        cache[tid] = emb

        if (i + 1) % 50 == 0 or (i + 1) == len(missing):
            log.info(f"  Embedded {i+1}/{len(missing)} — saving checkpoint...")
            np.savez(CACHE_PATH, **cache)

    log.info(f"Saved {len(cache)} embeddings → {CACHE_PATH}")
    return cache


def train(rollouts: list[dict], emb_cache: dict[str, np.ndarray],
          epochs: int, lr: float, reg: float, reg_W: float, seed: int,
          output_path: Path):
    model = LatentFactorRecommender(latent_dim=8, lr=lr, reg=reg, reg_W=reg_W, seed=seed)

    total = len(rollouts) * epochs
    log.info(f"Training: {len(rollouts)} rollouts × {epochs} epochs = {total} updates")

    covered = sum(1 for r in rollouts if r.get("task_id", "") in emb_cache)
    log.info(f"Task embedding coverage: {covered}/{len(rollouts)} rollouts have embeddings")

    for epoch in range(epochs):
        for r in rollouts:
            task_id = r.get("task_id", "")
            task_emb = emb_cache.get(task_id)
            model.update(r, task_emb=task_emb)

        if (epoch + 1) % 10 == 0:
            total_obs = sum(model.n_obs.values())
            for agent, beta_map in model.beta.items():
                beta_vals = list(beta_map.values())
                n_obs = model.n_obs.get(agent, 0)
                W_mag = float(np.mean(np.abs(model.W[agent]))) if agent in model.W else 0
                log.info(
                    f"Epoch {epoch+1}/{epochs} [{agent}]: "
                    f"{n_obs} obs, {len(beta_map)} servers, "
                    f"beta mean={np.mean(beta_vals):.3f} std={np.std(beta_vals):.3f}, "
                    f"|W| mean={W_mag:.5f}"
                )

    model.save(str(output_path))
    log.info(f"Model saved → {output_path}")

    # Quick diagnostics per agent
    for agent, beta_map in model.beta.items():
        top = sorted(beta_map.items(), key=lambda x: x[1], reverse=True)[:5]
        bot = sorted(beta_map.items(), key=lambda x: x[1])[:3]
        log.info(f"[{agent}] Top-5 beta: " + ", ".join(f"{s}={b:.3f}" for s, b in top))
        log.info(f"[{agent}] Bot-3 beta: " + ", ".join(f"{s}={b:.3f}" for s, b in bot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="results/rollouts_train.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.001)
    parser.add_argument("--reg-W", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/latent_factor_v10_trained.json")
    args = parser.parse_args()

    rollouts_path = ROOT / args.rollouts if not Path(args.rollouts).is_absolute() else Path(args.rollouts)
    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading rollouts from {rollouts_path}")
    rollouts = load_rollouts(rollouts_path)
    log.info(f"Loaded {len(rollouts)} rollouts")

    # Filter out error-only rollouts (no useful signal)
    valid = [r for r in rollouts if not r.get("error") or r.get("inventory_failed")]
    log.info(f"Valid rollouts: {len(valid)} (filtered {len(rollouts) - len(valid)} pure errors)")

    task_map = build_task_map(valid)
    log.info(f"Unique tasks with queries: {len(task_map)}")

    emb_cache = load_or_build_cache(task_map)
    train(valid, emb_cache, args.epochs, args.lr, args.reg, args.reg_W, args.seed, output_path)


if __name__ == "__main__":
    main()

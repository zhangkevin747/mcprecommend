"""Train Tucker recommender.

Model: R̂(a,t,s) = semantic(t,s) + α * (b_s + b_a + Tucker(a,t,s))
Tucker(a,t,s) = (T.T @ task_emb) · (u_a ⊙ w_s)

Usage:
    .venv/bin/python scripts/train_tucker.py --epochs 100 --output results/tucker_trained.json
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
from src.exp2.recommenders.tucker import TuckerRecommender

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
    task_map = {}
    for r in rollouts:
        tid = r.get("task_id", "")
        q = r.get("task_query", "")
        if tid and q:
            task_map[tid] = q
    return task_map


def embed_one(client: openai.OpenAI, text: str) -> np.ndarray:
    text = text[:2000]
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def load_or_build_cache(task_map: dict[str, str]) -> dict[str, np.ndarray]:
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

    log.info(f"Computing {len(missing)} missing embeddings...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    for i, tid in enumerate(missing):
        emb = embed_one(client, task_map[tid])
        cache[tid] = emb
        if (i + 1) % 50 == 0 or (i + 1) == len(missing):
            log.info(f"  Embedded {i+1}/{len(missing)} — saving checkpoint...")
            np.savez(CACHE_PATH, **cache)

    log.info(f"Saved {len(cache)} embeddings → {CACHE_PATH}")
    return cache


def train(rollouts: list[dict], emb_cache: dict[str, np.ndarray],
          epochs: int, lr: float, lr_tucker: float, reg: float, reg_T: float, seed: int,
          output_path: Path):
    model = TuckerRecommender(latent_dim=8, lr=lr, lr_tucker=lr_tucker, reg=reg, reg_T=reg_T, seed=seed)

    log.info(f"Training: {len(rollouts)} rollouts × {epochs} epochs = {len(rollouts)*epochs} updates")
    covered = sum(1 for r in rollouts if r.get("task_id", "") in emb_cache)
    log.info(f"Task embedding coverage: {covered}/{len(rollouts)} rollouts")

    for epoch in range(epochs):
        for r in rollouts:
            task_id = r.get("task_id", "")
            task_emb = emb_cache.get(task_id)
            model.update(r, task_emb=task_emb)

        if (epoch + 1) % 10 == 0:
            total_obs = sum(model.n_obs.values())
            T_mag = float(np.mean(np.abs(model.T)))
            b_s_vals = list(model.b_s.values())
            log.info(
                f"Epoch {epoch+1}/{epochs}: {total_obs} obs, {len(model.b_s)} servers, "
                f"b_s mean={np.mean(b_s_vals):.3f} std={np.std(b_s_vals):.3f}, |T|={T_mag:.5f}"
            )
            for agent in model.b_a:
                n = model.n_obs.get(agent, 0)
                log.info(f"  [{agent}] n={n}, b_a={model.b_a[agent]:.3f}")

    model.save(str(output_path))
    log.info(f"Model saved → {output_path}")

    # Diagnostics
    top_b_s = sorted(model.b_s.items(), key=lambda x: x[1], reverse=True)[:5]
    bot_b_s = sorted(model.b_s.items(), key=lambda x: x[1])[:3]
    log.info("Top-5 server bias: " + ", ".join(f"{s}={b:.3f}" for s, b in top_b_s))
    log.info("Bot-3 server bias: " + ", ".join(f"{s}={b:.3f}" for s, b in bot_b_s))
    for agent, b_a in model.b_a.items():
        log.info(f"[{agent}] b_a={b_a:.3f}, u_a_mag={np.mean(np.abs(model.u_a[agent])):.4f}")
    T_row_norms = np.linalg.norm(model.T, axis=1)
    log.info(f"|T| row norms: mean={T_row_norms.mean():.5f}, max={T_row_norms.max():.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="results/rollouts_train.jsonl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-tucker", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=0.001)
    parser.add_argument("--reg-T", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/tucker_trained.json")
    args = parser.parse_args()

    rollouts_path = ROOT / args.rollouts if not Path(args.rollouts).is_absolute() else Path(args.rollouts)
    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading rollouts from {rollouts_path}")
    rollouts = load_rollouts(rollouts_path)
    log.info(f"Loaded {len(rollouts)} rollouts")

    valid = [r for r in rollouts if not r.get("error") or r.get("inventory_failed")]
    log.info(f"Valid rollouts: {len(valid)} (filtered {len(rollouts) - len(valid)} pure errors)")

    task_map = build_task_map(valid)
    log.info(f"Unique tasks with queries: {len(task_map)}")

    emb_cache = load_or_build_cache(task_map)
    train(valid, emb_cache, args.epochs, args.lr, args.lr_tucker, args.reg, args.reg_T, args.seed, output_path)


if __name__ == "__main__":
    main()

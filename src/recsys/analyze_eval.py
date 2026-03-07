"""Analyze held-out evaluation results: compare recommenders.

Usage:
    python -m src.exp2.analyze_eval [--eval-dir results/eval]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_rollouts(path: Path) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def compute_metrics(rollouts: list[dict]) -> dict:
    """Compute all evaluation metrics from rollouts."""
    n = len(rollouts)
    if n == 0:
        return {}

    # Filter out error rollouts
    valid = [r for r in rollouts if not r.get("error")]
    n_valid = len(valid)

    # 1. Server CTR: fraction of mounted servers that agent selected
    server_ctrs = []
    for r in valid:
        mounted = len(r.get("inventory_mounted", []))
        if mounted == 0:
            continue
        used_servers = set()
        for ts in r.get("tools_selected", []):
            if ":" in ts:
                used_servers.add(ts.split(":")[0])
        server_ctrs.append(len(used_servers) / mounted)

    # 2. Feedback scores
    liked = 0
    neutral = 0
    disliked = 0
    for r in valid:
        for fb in r.get("feedback", {}).values():
            if not isinstance(fb, dict):
                continue
            rating = fb.get("rating", "")
            if rating == "liked":
                liked += 1
            elif rating == "neutral":
                neutral += 1
            elif rating == "disliked":
                disliked += 1
    total_fb = liked + neutral + disliked

    # Feedback score: liked=1, neutral=0.5, disliked=0
    fb_score = (liked * 1.0 + neutral * 0.5) / total_fb if total_fb else 0

    # 3. Broken tool rate
    total_selected = sum(len(r.get("tools_selected", [])) for r in valid)
    total_errored = sum(len(r.get("tools_errored", [])) for r in valid)

    # 4. Retry/abandon rate
    total_abandoned = sum(len(r.get("tools_abandoned", [])) for r in valid)

    # 5. Mount success rate + avg failures per rollout
    total_mounted = sum(len(r.get("inventory_mounted", [])) for r in valid)
    total_failed = sum(len(r.get("inventory_failed", [])) for r in valid)
    avg_mount_failures = total_failed / n_valid if n_valid else 0

    # 6. Tool used rate (rollouts where agent used at least one tool)
    tool_used = sum(1 for r in valid if r.get("tools_selected"))

    # 7. Per-agent breakdown
    agent_metrics = {}
    for agent in ["haiku-4.5", "gpt-4o-mini", "gpt-5-mini"]:
        ar = [r for r in valid if r.get("agent") == agent]
        if not ar:
            continue
        al = sum(1 for r in ar for fb in r.get("feedback", {}).values()
                 if isinstance(fb, dict) and fb.get("rating") == "liked")
        ad = sum(1 for r in ar for fb in r.get("feedback", {}).values()
                 if isinstance(fb, dict) and fb.get("rating") == "disliked")
        an = sum(1 for r in ar for fb in r.get("feedback", {}).values()
                 if isinstance(fb, dict) and fb.get("rating") == "neutral")
        at = al + ad + an
        ctrs = []
        for r in ar:
            m = len(r.get("inventory_mounted", []))
            if m == 0:
                continue
            used = set()
            for ts in r.get("tools_selected", []):
                if ":" in ts:
                    used.add(ts.split(":")[0])
            ctrs.append(len(used) / m)
        agent_metrics[agent] = {
            "n": len(ar),
            "liked_pct": round(al / at * 100, 1) if at else 0,
            "disliked_pct": round(ad / at * 100, 1) if at else 0,
            "fb_score": round((al * 1.0 + an * 0.5) / at, 3) if at else 0,
            "server_ctr": round(sum(ctrs) / len(ctrs), 3) if ctrs else 0,
        }

    return {
        "n_total": n,
        "n_valid": n_valid,
        "n_errors": n - n_valid,
        "server_ctr": round(sum(server_ctrs) / len(server_ctrs), 3) if server_ctrs else 0,
        "liked_pct": round(liked / total_fb * 100, 1) if total_fb else 0,
        "neutral_pct": round(neutral / total_fb * 100, 1) if total_fb else 0,
        "disliked_pct": round(disliked / total_fb * 100, 1) if total_fb else 0,
        "feedback_score": round(fb_score, 3),
        "broken_tool_rate": round(total_errored / total_selected * 100, 1) if total_selected else 0,
        "abandon_rate": round(total_abandoned / total_selected * 100, 1) if total_selected else 0,
        "mount_success_rate": round(total_mounted / (total_mounted + total_failed) * 100, 1) if (total_mounted + total_failed) else 0,
        "avg_mount_failures": round(avg_mount_failures, 2),
        "tool_used_rate": round(tool_used / n_valid * 100, 1) if n_valid else 0,
        "avg_cost_usd": round(sum(r.get("cost_usd", 0) for r in rollouts) / n, 4) if n else 0,
        "per_agent": agent_metrics,
    }


def print_comparison(all_metrics: dict[str, dict]):
    """Print comparison table."""
    methods = list(all_metrics.keys())

    print("\n" + "=" * 90)
    print("HELD-OUT EVALUATION: RECOMMENDER COMPARISON")
    print("=" * 90)

    # Main metrics table
    header = f"{'Metric':<25}" + "".join(f"{m:>16}" for m in methods)
    print(f"\n{header}")
    print("-" * len(header))

    rows = [
        ("Rollouts (valid)", "n_valid"),
        ("Server CTR", "server_ctr"),
        ("Liked %", "liked_pct"),
        ("Disliked %", "disliked_pct"),
        ("Feedback Score", "feedback_score"),
        ("Broken Tool Rate %", "broken_tool_rate"),
        ("Abandon Rate %", "abandon_rate"),
        ("Mount Success %", "mount_success_rate"),
        ("Avg Failures/Rollout", "avg_mount_failures"),
        ("Tool Used Rate %", "tool_used_rate"),
        ("Avg Cost (USD)", "avg_cost_usd"),
    ]

    for label, key in rows:
        vals = []
        for m in methods:
            v = all_metrics[m].get(key, "?")
            if isinstance(v, float):
                vals.append(f"{v:>16.3f}" if v < 1 else f"{v:>16.1f}")
            else:
                vals.append(f"{v:>16}")
        print(f"{label:<25}" + "".join(vals))

    # Per-agent breakdown
    print(f"\n{'--- Per-Agent Feedback Score ---':}")
    for agent in ["haiku-4.5", "gpt-4o-mini", "gpt-5-mini"]:
        row = f"  {agent:<23}"
        for m in methods:
            am = all_metrics[m].get("per_agent", {}).get(agent, {})
            fb = am.get("fb_score", "?")
            row += f"{fb:>16.3f}" if isinstance(fb, (int, float)) else f"{fb:>16}"
        print(row)

    print(f"\n{'--- Per-Agent Liked %     ---':}")
    for agent in ["haiku-4.5", "gpt-4o-mini", "gpt-5-mini"]:
        row = f"  {agent:<23}"
        for m in methods:
            am = all_metrics[m].get("per_agent", {}).get(agent, {})
            lp = am.get("liked_pct", "?")
            row += f"{lp:>16.1f}" if isinstance(lp, (int, float)) else f"{lp:>16}"
        print(row)

    print(f"\n{'--- Per-Agent Server CTR  ---':}")
    for agent in ["haiku-4.5", "gpt-4o-mini", "gpt-5-mini"]:
        row = f"  {agent:<23}"
        for m in methods:
            am = all_metrics[m].get("per_agent", {}).get(agent, {})
            ctr = am.get("server_ctr", "?")
            row += f"{ctr:>16.3f}" if isinstance(ctr, (int, float)) else f"{ctr:>16}"
        print(row)

    # Winner summary
    print(f"\n{'=' * 90}")
    print("WINNERS")
    print("=" * 90)
    key_metrics = [
        ("Feedback Score (higher=better)", "feedback_score", True),
        ("Liked % (higher=better)", "liked_pct", True),
        ("Server CTR (higher=better)", "server_ctr", True),
        ("Broken Tool Rate (lower=better)", "broken_tool_rate", False),
        ("Abandon Rate (lower=better)", "abandon_rate", False),
    ]
    for label, key, higher_is_better in key_metrics:
        best_m = None
        best_v = None
        for m in methods:
            v = all_metrics[m].get(key, None)
            if v is None:
                continue
            if best_v is None or (higher_is_better and v > best_v) or (not higher_is_better and v < best_v):
                best_m = m
                best_v = v
        print(f"  {label}: {best_m} ({best_v})")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, default="results/eval")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir

    all_metrics = {}
    for path in sorted(eval_dir.glob("eval_*.jsonl")):
        rec_name = path.stem.replace("eval_", "")
        rollouts = load_rollouts(path)
        if rollouts:
            metrics = compute_metrics(rollouts)
            all_metrics[rec_name] = metrics
            print(f"Loaded {rec_name}: {len(rollouts)} rollouts")

    if not all_metrics:
        print("No eval results found! Run: python -m src.exp2.run_eval")
        return

    print_comparison(all_metrics)

    # Save raw metrics
    out_path = eval_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Raw metrics saved to {out_path}")


if __name__ == "__main__":
    main()

"""Generate presentation charts for MCP Tool Recommender results.

Charts:
  1. Liked % by method × agent (grouped bar)
  2. Liked / Neutral / Disliked stacked bar
  3. Delta over semantic baseline
  4. Convergence curve from training rollouts
  5. Learned server bias (top/bottom servers from Tucker)

Usage:
    .venv/bin/python scripts/plot_results.py --out results/plots/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── colour palette ────────────────────────────────────────────────────────────
AGENT_COLORS = {
    "gpt-4o-mini": "#4C72B0",
    "gpt-5-mini":  "#DD8452",
}
METHOD_COLORS = {
    "random":    "#9E9E9E",
    "semantic":  "#64B5F6",
    "sem+pop":   "#81C784",
    "lf_v10":    "#FFB74D",
    "tucker":    "#E57373",
}
METHOD_LABELS = {
    "random":    "Random",
    "semantic":  "Semantic",
    "sem+pop":   "Sem+Pop",
    "lf_v10":    "LF v10",
    "tucker":    "Tucker",
}

# ── data loading ──────────────────────────────────────────────────────────────

def load_dedup(paths):
    seen = set()
    rollouts = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("task_id", ""), r.get("agent", ""))
                if key not in seen:
                    seen.add(key)
                    rollouts.append(r)
    return rollouts


def tool_ratings(rollouts):
    """Return (liked, neutral, disliked, total) counts from tool-level feedback."""
    liked = neutral = disliked = 0
    for r in rollouts:
        for fb in r.get("feedback", {}).values():
            if not isinstance(fb, dict):
                continue
            rat = fb.get("rating", "")
            if rat == "liked":
                liked += 1
            elif rat == "neutral":
                neutral += 1
            elif rat == "disliked":
                disliked += 1
    total = liked + neutral + disliked
    return liked, neutral, disliked, total


def per_agent_metrics(rollouts):
    by_agent = defaultdict(list)
    for r in rollouts:
        by_agent[r.get("agent", "?")].append(r)
    out = {}
    for agent, rs in by_agent.items():
        l, n, d, t = tool_ratings(rs)
        out[agent] = {
            "liked":    round(100 * l / t, 1) if t else 0,
            "neutral":  round(100 * n / t, 1) if t else 0,
            "disliked": round(100 * d / t, 1) if t else 0,
            "fb_score": round((l + 0.5 * n) / t, 3) if t else 0,
            "n": len(rs),
        }
    return out


# ── chart 1: grouped bar — liked % by method × agent ─────────────────────────

def plot_liked_grouped(all_metrics, agents, methods, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    n_methods = len(methods)
    n_agents  = len(agents)
    bar_w     = 0.35
    group_gap = 0.15
    group_w   = n_agents * bar_w + group_gap

    for mi, method in enumerate(methods):
        for ai, agent in enumerate(agents):
            x = mi * group_w + ai * bar_w
            val = all_metrics[method].get(agent, {}).get("liked", 0)
            color = AGENT_COLORS[agent]
            edge = "white" if method != "tucker" else "black"
            lw   = 0.5      if method != "tucker" else 1.5
            ax.bar(x, val, width=bar_w - 0.03, color=color,
                   edgecolor=edge, linewidth=lw, zorder=3)
            ax.text(x + (bar_w - 0.03) / 2, val + 0.8, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # x-axis ticks at group centres
    tick_xs = [mi * group_w + (n_agents - 1) * bar_w / 2 for mi in range(n_methods)]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=11)
    ax.set_ylabel("Liked % (tool-rating level)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Recommendation Quality: Liked % by Method and Agent", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [mpatches.Patch(color=AGENT_COLORS[a], label=a) for a in agents]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper left")

    plt.tight_layout()
    path = out_dir / "1_liked_grouped.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── chart 2: stacked bar — liked/neutral/disliked ────────────────────────────

def plot_stacked(all_metrics, agents, methods, out_dir):
    fig, axes = plt.subplots(1, len(agents), figsize=(11, 5), sharey=True)

    for ax, agent in zip(axes, agents):
        liked_vals    = [all_metrics[m].get(agent, {}).get("liked",    0) for m in methods]
        neutral_vals  = [all_metrics[m].get(agent, {}).get("neutral",  0) for m in methods]
        disliked_vals = [all_metrics[m].get(agent, {}).get("disliked", 0) for m in methods]
        xs = np.arange(len(methods))
        bw = 0.55

        b1 = ax.bar(xs, liked_vals,    width=bw, color="#66BB6A", label="Liked",    zorder=3)
        b2 = ax.bar(xs, neutral_vals,  width=bw, bottom=liked_vals, color="#FFA726", label="Neutral",  zorder=3)
        bottoms = [l + n for l, n in zip(liked_vals, neutral_vals)]
        b3 = ax.bar(xs, disliked_vals, width=bw, bottom=bottoms,    color="#EF5350", label="Disliked", zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=10, rotation=15, ha="right")
        ax.set_title(agent, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("% of tool ratings", fontsize=11)

    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Rating Breakdown by Method and Agent", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "2_stacked_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── chart 3: delta over semantic ─────────────────────────────────────────────

def plot_delta(all_metrics, agents, methods, out_dir):
    non_semantic = [m for m in methods if m != "semantic"]
    fig, axes = plt.subplots(1, len(agents), figsize=(10, 5), sharey=True)

    for ax, agent in zip(axes, agents):
        base = all_metrics["semantic"].get(agent, {}).get("liked", 0)
        deltas = [all_metrics[m].get(agent, {}).get("liked", 0) - base for m in non_semantic]
        colors = ["#EF9A9A" if d < 0 else "#81C784" for d in deltas]

        xs = np.arange(len(non_semantic))
        bars = ax.bar(xs, deltas, width=0.55, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(0, color="black", linewidth=1)
        for i, (x, d) in enumerate(zip(xs, deltas)):
            sign = "+" if d >= 0 else ""
            ax.text(x, d + (0.3 if d >= 0 else -0.8), f"{sign}{d:.1f}pp",
                    ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in non_semantic], fontsize=10, rotation=15, ha="right")
        ax.set_title(f"{agent}\n(baseline: semantic = {base:.0f}%)", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Δ Liked % vs Semantic", fontsize=11)

    fig.suptitle("Improvement Over Semantic Baseline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "3_delta_semantic.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── chart 4: convergence from training rollouts ───────────────────────────────

def plot_convergence(train_path, out_dir, window=100):
    rollouts = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    # compute per-rollout feedback score
    scores_by_agent = defaultdict(list)   # agent → [(rollout_idx, score)]
    all_scores = []

    for i, r in enumerate(rollouts):
        agent = r.get("agent", "?")
        fb = r.get("feedback", {})
        ratings = [v.get("rating", "") for v in fb.values() if isinstance(v, dict)]
        if not ratings:
            continue
        score = sum({"liked": 1.0, "neutral": 0.5, "disliked": 0.0}.get(rt, 0.5) for rt in ratings) / len(ratings)
        scores_by_agent[agent].append((i, score))
        all_scores.append((i, score))

    def rolling_avg(pairs, win):
        if len(pairs) < win:
            return [], []
        idxs, vals = zip(*pairs)
        cumsum = np.cumsum(vals)
        rolled = (cumsum[win - 1:] - np.concatenate([[0], cumsum[:-win]])) / win
        return list(idxs[win - 1:]), list(rolled)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Overall rolling avg
    ox, oy = rolling_avg(all_scores, window)
    ax.plot(ox, oy, color="black", linewidth=2.5, label=f"All agents (n={len(all_scores)})", zorder=5)

    # Per-agent
    for agent, pairs in sorted(scores_by_agent.items()):
        color = AGENT_COLORS.get(agent, "#999")
        xs, ys = rolling_avg(pairs, window // 2)
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7, linestyle="--", label=agent)

    ax.set_xlabel("Training rollout index", fontsize=11)
    ax.set_ylabel(f"Feedback score (rolling avg, window={window})", fontsize=11)
    ax.set_title("Training Convergence: Feedback Score Over Rollouts", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = out_dir / "4_convergence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── chart 5: learned server bias ─────────────────────────────────────────────

def plot_server_bias(model_path, out_dir, top_n=12, bot_n=5):
    with open(model_path) as f:
        model = json.load(f)

    b_s = model.get("b_s", {})
    if not b_s:
        print("No b_s in model, skipping chart 5")
        return

    sorted_servers = sorted(b_s.items(), key=lambda x: x[1], reverse=True)
    top = sorted_servers[:top_n]
    bot = sorted_servers[-bot_n:]
    entries = bot[::-1] + [("", None)] + top[::-1]  # bottom up on horizontal bar

    labels, values, colors = [], [], []
    for name, val in entries:
        if val is None:
            labels.append("")
            values.append(0)
            colors.append("white")
        else:
            short = name[:35] + ("…" if len(name) > 35 else "")
            labels.append(short)
            values.append(val)
            colors.append("#81C784" if val >= 0 else "#EF5350")

    fig, ax = plt.subplots(figsize=(9, 8))
    ys = np.arange(len(labels))
    ax.barh(ys, values, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Learned server quality bias (b_s)", fontsize=11)
    ax.set_title(f"Tucker Model: Learned Server Quality Bias\n(top {top_n} and bottom {bot_n} of {len(b_s)} servers)",
                 fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = out_dir / "5_server_bias.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all eval data
    file_map = {
        "random":   [ROOT / "results/eval_canonical/baselines_random_gpt4o.jsonl",
                     ROOT / "results/eval_canonical/baselines_random_gpt5.jsonl"],
        "semantic": [ROOT / "results/eval_canonical/baselines_semantic_gpt4o.jsonl",
                     ROOT / "results/eval_canonical/baselines_semantic_gpt5.jsonl"],
        "sem+pop":  [ROOT / "results/eval_canonical/baselines_semantic_pop_gpt4o.jsonl",
                     ROOT / "results/eval_canonical/baselines_semantic_pop_gpt5.jsonl"],
        "lf_v10":   [ROOT / "results/eval_canonical/lf_v10_both.jsonl"],
        "tucker":   [ROOT / "results/eval/eval_tucker.jsonl"],
    }

    all_metrics = {}
    for method, paths in file_map.items():
        rollouts = load_dedup(paths)
        all_metrics[method] = per_agent_metrics(rollouts)
        print(f"Loaded {method}: {sum(m['n'] for m in all_metrics[method].values())} rollouts")

    agents  = ["gpt-4o-mini", "gpt-5-mini"]
    methods = ["random", "semantic", "sem+pop", "lf_v10", "tucker"]

    plot_liked_grouped(all_metrics, agents, methods, out_dir)
    plot_stacked(all_metrics, agents, methods, out_dir)
    plot_delta(all_metrics, agents, methods, out_dir)
    plot_convergence(ROOT / "results/rollouts_train.jsonl", out_dir)
    plot_server_bias(ROOT / "results/tucker_trained.json", out_dir)

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()

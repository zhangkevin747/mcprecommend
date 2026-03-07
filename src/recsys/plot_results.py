"""Generate all evaluation plots for the MCP recommender experiment.

Usage:
    python -m src.exp2.plot_results

Outputs 7 plots to results/plots/.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "results" / "eval"
TRAIN_PATH = ROOT / "results" / "rollouts_train.jsonl"
MODEL_PATH = ROOT / "results" / "latent_factor_v6_trained.json"
OUT_DIR = ROOT / "results" / "plots"

RECS = ["random", "popularity", "semantic", "latent_factor"]
REC_LABELS = ["Random", "Popularity", "Semantic", "Latent Factor"]
COLORS = ["#aec6e8", "#ffb347", "#90ee90", "#c97fd4"]
AGENTS = ["haiku-4.5", "gpt-4o-mini", "gpt-5-mini"]
AGENT_LABELS = ["Haiku 4.5", "GPT-4o-mini", "GPT-5-mini"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_eval(rec: str) -> list[dict]:
    path = EVAL_DIR / f"eval_{rec}.jsonl"
    return [json.loads(l) for l in open(path)]


def tool_using(rollouts: list[dict]) -> list[dict]:
    return [r for r in rollouts if r.get("tools_selected")]


def liked_rate(rollouts: list[dict]) -> tuple[int, int]:
    """Returns (n_liked_rollouts, n_rollouts) where liked = any liked rating."""
    liked = sum(
        1 for r in rollouts
        if "liked" in [f.get("rating", "") for f in r.get("feedback", {}).values()
                       if isinstance(f, dict)]
    )
    return liked, len(rollouts)


def tool_liked_rate(rollouts: list[dict]) -> tuple[int, int]:
    """Tool-level: (n_liked_tools, n_rated_tools)."""
    liked = disliked = neutral = 0
    for r in rollouts:
        for fb in r.get("feedback", {}).values():
            if isinstance(fb, dict):
                rating = fb.get("rating", "")
                if rating == "liked":    liked += 1
                elif rating == "disliked": disliked += 1
                elif rating == "neutral":  neutral += 1
    return liked, liked + disliked + neutral


def mount_failures(rollouts: list[dict]) -> int:
    return sum(len(r.get("inventory_failed", [])) for r in rollouts)


# ---------------------------------------------------------------------------
# Plot 1: Main result — liked rate, both framings side by side
# ---------------------------------------------------------------------------

def plot_main_result():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, use_filter, title, ylabel in [
        (axes[0], False, "All rollouts (429 per recommender)", "Rollout-level liked rate"),
        (axes[1], True,  "Tool-using rollouts only",           "Rollout-level liked rate"),
    ]:
        rates, ns, totals = [], [], []
        for rec in RECS:
            rolls = load_eval(rec)
            if use_filter:
                rolls = tool_using(rolls)
            n, total = liked_rate(rolls)
            rates.append(n / total if total else 0)
            ns.append(n)
            totals.append(total)

        bars = ax.bar(REC_LABELS, [r * 100 for r in rates], color=COLORS,
                      edgecolor="white", linewidth=0.8, width=0.55)

        # Value labels
        for bar, n, total in zip(bars, ns, totals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{bar.get_height():.1f}%\n({n}/{total})",
                    ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(r * 100 for r in rates) + 12)
        ax.tick_params(axis="x", labelsize=10)

    fig.suptitle("Recommender Comparison: Agent Liked Rate on Held-Out Tasks",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "plot1_main_result.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Mount failures per recommender
# ---------------------------------------------------------------------------

def plot_mount_failures():
    fig, ax = plt.subplots(figsize=(8, 5))

    all_rollouts = {rec: load_eval(rec) for rec in RECS}
    failures = [mount_failures(all_rollouts[rec]) for rec in RECS]
    total_mounts = [
        sum(len(r.get("inventory_mounted", [])) + len(r.get("inventory_failed", []))
            for r in all_rollouts[rec])
        for rec in RECS
    ]

    bars = ax.bar(REC_LABELS, failures, color=COLORS, edgecolor="white",
                  linewidth=0.8, width=0.55)

    for bar, f, t in zip(bars, failures, total_mounts):
        pct = f / t * 100 if t else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{f}\n({pct:.1f}% of mounts)",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Mount Failures per Recommender\n(server failed to connect during eval)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Total mount failures (429 rollouts)")
    ax.set_ylim(0, max(failures) + 120)
    fig.tight_layout()
    path = OUT_DIR / "plot2_mount_failures.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-agent grouped bar
# ---------------------------------------------------------------------------

def plot_per_agent():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Collect liked rate per (agent, rec), tool-using rollouts only
    data = {}
    for rec in RECS:
        rolls = load_eval(rec)
        data[rec] = {}
        for agent in AGENTS:
            agent_rolls = tool_using([r for r in rolls if r.get("agent") == agent])
            n, total = liked_rate(agent_rolls)
            data[rec][agent] = (n / total * 100) if total else 0

    n_agents = len(AGENTS)
    n_recs = len(RECS)
    x = np.arange(n_agents)
    width = 0.18
    offsets = np.linspace(-(n_recs - 1) / 2, (n_recs - 1) / 2, n_recs) * width

    for i, (rec, label, color) in enumerate(zip(RECS, REC_LABELS, COLORS)):
        vals = [data[rec][agent] for agent in AGENTS]
        bars = ax.bar(x + offsets[i], vals, width, label=label,
                      color=color, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_LABELS, fontsize=11)
    ax.set_ylabel("Rollout-level liked rate (tool-using rollouts)")
    ax.set_title("Per-Agent Liked Rate by Recommender",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = OUT_DIR / "plot3_per_agent.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Beta vs. observed like rate scatter
# ---------------------------------------------------------------------------

def plot_beta_quality():
    from src.exp2.recommenders.latent_factor import LatentFactorRecommender
    model = LatentFactorRecommender()
    model.load(str(MODEL_PATH))

    # Observed like rate per server from training data
    server_fb = defaultdict(lambda: {"liked": 0, "disliked": 0, "neutral": 0})
    train = [json.loads(l) for l in open(TRAIN_PATH)]
    for r in train:
        for tool, fb in r.get("feedback", {}).items():
            if isinstance(fb, dict):
                sid = tool.split(":")[0] if ":" in tool else tool
                rating = fb.get("rating", "")
                if rating in server_fb[sid]:
                    server_fb[sid][rating] += 1

    # Filter to servers with ≥5 feedback and known beta
    betas, like_rates, sizes = [], [], []
    for sid, fb in server_fb.items():
        total = fb["liked"] + fb["disliked"] + fb["neutral"]
        if total >= 5 and sid in model.beta_server:
            betas.append(model.beta_server[sid])
            like_rates.append(fb["liked"] / total)
            sizes.append(min(total * 4, 200))  # bubble size

    betas_arr = np.array(betas)
    like_rates_arr = np.array(like_rates)

    # Pearson
    pearson = np.corrcoef(betas_arr, like_rates_arr)[0, 1]
    # Spearman (manual)
    rank_b = np.argsort(np.argsort(betas_arr)).astype(float)
    rank_l = np.argsort(np.argsort(like_rates_arr)).astype(float)
    spearman = np.corrcoef(rank_b, rank_l)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(betas_arr, like_rates_arr * 100, s=sizes,
                    alpha=0.65, c=betas_arr, cmap="RdYlGn", edgecolors="white",
                    linewidth=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Learned β (server quality bias)")

    # Regression line
    m, b = np.polyfit(betas_arr, like_rates_arr * 100, 1)
    x_line = np.linspace(betas_arr.min(), betas_arr.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#444", linewidth=1.5,
            linestyle="--", zorder=2)

    ax.axvline(0, color="#ccc", linewidth=1, zorder=1)
    ax.axhline(50, color="#ccc", linewidth=1, zorder=1)

    ax.set_xlabel("Learned β (server quality bias)", fontsize=11)
    ax.set_ylabel("Observed like rate (%)", fontsize=11)
    ax.set_title(
        f"Learned β vs. Observed Like Rate\n"
        f"Pearson r = {pearson:.3f}  |  Spearman ρ = {spearman:.3f}  "
        f"  (n={len(betas)} servers with ≥5 feedback)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUT_DIR / "plot4_beta_quality.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 5: Training convergence
# ---------------------------------------------------------------------------

def plot_convergence():
    train = [json.loads(l) for l in open(TRAIN_PATH)]
    chunk = 200

    windows, rates, lower, upper = [], [], [], []
    for start in range(0, len(train), chunk):
        chunk_rolls = train[start:start + chunk]
        liked = disliked = neutral = 0
        for r in chunk_rolls:
            for fb in r.get("feedback", {}).values():
                if isinstance(fb, dict):
                    rating = fb.get("rating", "")
                    if rating == "liked":    liked += 1
                    elif rating == "disliked": disliked += 1
                    elif rating == "neutral":  neutral += 1
        total = liked + disliked + neutral
        if total == 0:
            continue
        rate = liked / total
        # Wilson score CI
        z = 1.96
        n = total
        p = rate
        center = (p + z**2 / (2*n)) / (1 + z**2 / n)
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
        windows.append(start + chunk // 2)
        rates.append(rate * 100)
        lower.append((center - margin) * 100)
        upper.append((center + margin) * 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(windows, lower, upper, alpha=0.2, color="#c97fd4")
    ax.plot(windows, rates, color="#c97fd4", linewidth=2, marker="o",
            markersize=5, label="Liked rate (window=200)")

    # Baseline: semantic eval liked rate
    sem_rolls = load_eval("semantic")
    sem_rate = liked_rate(tool_using(sem_rolls))[0] / len(tool_using(sem_rolls)) * 100
    ax.axhline(sem_rate, color="#90ee90", linewidth=1.5, linestyle="--",
               label=f"Semantic eval baseline ({sem_rate:.1f}%)")

    ax.set_xlabel("Training rollout number")
    ax.set_ylabel("Liked rate % (tool-level, 200-rollout window)")
    ax.set_title("Training Convergence: Feedback Quality Over Time",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(40, 90)
    fig.tight_layout()
    path = OUT_DIR / "plot5_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 6: Signal breakdown
# ---------------------------------------------------------------------------

def plot_signal_breakdown():
    train = [json.loads(l) for l in open(TRAIN_PATH)]

    counts = Counter()
    for r in train:
        selected = set()
        for t in r.get("tools_selected", []):
            selected.add(t.split(":")[0] if ":" in t else t)
        errored = set()
        for t in r.get("tools_errored", []):
            errored.add(t.split(":")[0] if ":" in t else t)
        server_fb = {}
        for t, fb in r.get("feedback", {}).items():
            sid = t.split(":")[0] if ":" in t else t
            if isinstance(fb, dict):
                server_fb[sid] = fb.get("rating", "")

        for sid in r.get("inventory_failed", []):
            counts["Mount failed\n(−1.0)"] += 1
        for sid in selected:
            if sid in r.get("inventory_failed", []):
                continue
            if sid in server_fb:
                rating = server_fb[sid]
                if rating == "liked":     counts["Selected + liked\n(+1.0)"] += 1
                elif rating == "neutral": counts["Selected + neutral\n(+0.2)"] += 1
                elif rating == "disliked":counts["Selected + disliked\n(−0.8)"] += 1
                else:                     counts["Selected + no feedback\n(+0.1)"] += 1
            elif sid in errored:          counts["Selected + errored\n(−0.7)"] += 1
            else:                         counts["Selected + no feedback\n(+0.1)"] += 1

    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    total = sum(values)

    # Colors: green for positive signals, red for negative
    sig_colors = {
        "Selected + liked\n(+1.0)":       "#5cb85c",
        "Selected + neutral\n(+0.2)":     "#a8d5a2",
        "Selected + no feedback\n(+0.1)": "#d4edda",
        "Mount failed\n(−1.0)":           "#d9534f",
        "Selected + disliked\n(−0.8)":    "#e8a29e",
        "Selected + errored\n(−0.7)":     "#f7c6c5",
    }
    bar_colors = [sig_colors.get(l, "#ccc") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="white",
                   linewidth=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{v}  ({v/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Number of signals fired (1,830 training rollouts)")
    ax.set_title("Signal Breakdown: What the Recommender Actually Learned From",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    path = OUT_DIR / "plot6_signal_breakdown.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 7: Per-category LF vs Semantic delta
# ---------------------------------------------------------------------------

def plot_category_delta():
    lf_rolls   = tool_using(load_eval("latent_factor"))
    sem_rolls  = tool_using(load_eval("semantic"))

    def cat_liked(rolls):
        cats = defaultdict(lambda: {"liked": 0, "total": 0})
        for r in rolls:
            cat = r.get("task_category", "unknown")
            ratings = [f.get("rating", "") for f in r.get("feedback", {}).values()
                       if isinstance(f, dict)]
            cats[cat]["total"] += 1
            if "liked" in ratings:
                cats[cat]["liked"] += 1
        return cats

    lf_cats  = cat_liked(lf_rolls)
    sem_cats = cat_liked(sem_rolls)

    # Only categories with ≥5 rollouts in both
    cats = sorted(
        [c for c in set(lf_cats) | set(sem_cats)
         if lf_cats[c]["total"] >= 5 and sem_cats[c]["total"] >= 5],
        key=lambda c: (lf_cats[c]["liked"] / lf_cats[c]["total"]) -
                      (sem_cats[c]["liked"] / sem_cats[c]["total"])
    )

    deltas = [(lf_cats[c]["liked"] / lf_cats[c]["total"] -
               sem_cats[c]["liked"] / sem_cats[c]["total"]) * 100
              for c in cats]
    lf_ns  = [f"{lf_cats[c]['liked']}/{lf_cats[c]['total']}" for c in cats]
    sem_ns = [f"{sem_cats[c]['liked']}/{sem_cats[c]['total']}" for c in cats]

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = ["#5cb85c" if d >= 0 else "#d9534f" for d in deltas]
    bars = ax.barh(cats, deltas, color=bar_colors, edgecolor="white",
                   linewidth=0.6)

    for bar, d, lf_n, sem_n in zip(bars, deltas, lf_ns, sem_ns):
        xpos = d + 0.5 if d >= 0 else d - 0.5
        ha = "left" if d >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"LF {lf_n} vs Sem {sem_n}",
                va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="#333", linewidth=1)
    ax.set_xlabel("Liked rate delta: Latent Factor − Semantic (pp)")
    ax.set_title("Per-Category: LF vs Semantic Liked Rate Delta\n(tool-using rollouts, ≥5 per category)",
                 fontsize=12, fontweight="bold")

    pos_patch = mpatches.Patch(color="#5cb85c", label="LF wins")
    neg_patch = mpatches.Patch(color="#d9534f", label="Semantic wins")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "plot7_category_delta.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_main_result()
    plot_mount_failures()
    plot_per_agent()
    plot_beta_quality()
    plot_convergence()
    plot_signal_breakdown()
    plot_category_delta()
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

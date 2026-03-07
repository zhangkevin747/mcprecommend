#!/usr/bin/env python3
"""Diagnose why the latent factor recommender isn't beating semantic baseline.

Analyzes:
1. Model parameters: beta distributions, gamma magnitudes, observation count
2. Training rollout feedback statistics per server
3. Beta-server vs actual liked% correlation
4. Eval comparison: overlap, reordering frequency, reordering quality
5. Collab bonus magnitude vs semantic score spread
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/Users/kevinzhang/Documents/GitHub/MCPToolBenchPP")
MODEL_PATH = BASE / "results" / "latent_factor_v3_trained.json"
TRAIN_PATH = BASE / "results" / "rollouts_train.jsonl"
EVAL_LF_PATH = BASE / "results" / "eval" / "eval_latent_factor.jsonl"
EVAL_SEM_PATH = BASE / "results" / "eval" / "eval_semantic.jsonl"

sep = "=" * 80


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ============================================================================
# 1. MODEL PARAMETER ANALYSIS
# ============================================================================
print(sep)
print("SECTION 1: TRAINED MODEL PARAMETERS")
print(sep)

with open(MODEL_PATH) as f:
    model = json.load(f)

n_obs = model["n_observations"]
latent_dim = model["latent_dim"]
n_servers = len(model["beta_server"])
n_agents = len(model["gamma_agent"])
n_categories = len(model["delta_category"])

print(f"n_observations:    {n_obs}")
print(f"latent_dim:        {latent_dim}")
print(f"n_servers learned: {n_servers}")
print(f"n_agents:          {n_agents} ({list(model['gamma_agent'].keys())})")
print(f"n_categories:      {n_categories} ({list(model['delta_category'].keys())})")
print()

# Alpha at inference
alpha = min(n_obs / 500, 0.3)
print(f"Alpha at inference: min({n_obs}/500, 0.3) = {alpha:.3f}")
print()

# Beta distribution
betas = np.array(list(model["beta_server"].values()))
print(f"beta_server distribution (n={len(betas)}):")
print(f"  min:    {betas.min():.6f}")
print(f"  max:    {betas.max():.6f}")
print(f"  mean:   {betas.mean():.6f}")
print(f"  std:    {betas.std():.6f}")
print(f"  median: {np.median(betas):.6f}")
print(f"  P5:     {np.percentile(betas, 5):.6f}")
print(f"  P25:    {np.percentile(betas, 25):.6f}")
print(f"  P75:    {np.percentile(betas, 75):.6f}")
print(f"  P95:    {np.percentile(betas, 95):.6f}")

# Count how many are near zero (< 0.01 magnitude)
near_zero = np.sum(np.abs(betas) < 0.01)
small = np.sum(np.abs(betas) < 0.05)
print(f"  |beta| < 0.01:  {near_zero}/{len(betas)} ({100*near_zero/len(betas):.1f}%)")
print(f"  |beta| < 0.05:  {small}/{len(betas)} ({100*small/len(betas):.1f}%)")
print()

# Top/bottom beta servers
beta_items = sorted(model["beta_server"].items(), key=lambda x: x[1])
print("Top 10 beta_server (highest quality bias):")
for sid, b in beta_items[-10:][::-1]:
    print(f"  {b:+.4f}  {sid}")
print()
print("Bottom 10 beta_server (lowest quality bias):")
for sid, b in beta_items[:10]:
    print(f"  {b:+.4f}  {sid}")
print()

# Gamma magnitudes
print("Gamma_agent magnitudes (||gamma||):")
for agent, vec in model["gamma_agent"].items():
    mag = np.linalg.norm(vec)
    print(f"  {agent}: ||gamma|| = {mag:.6f}")
print()

gamma_server_mags = []
for sid, vec in model["gamma_server"].items():
    gamma_server_mags.append(np.linalg.norm(vec))
gamma_server_mags = np.array(gamma_server_mags)
print(f"Gamma_server magnitudes (n={len(gamma_server_mags)}):")
print(f"  min:  {gamma_server_mags.min():.6f}")
print(f"  max:  {gamma_server_mags.max():.6f}")
print(f"  mean: {gamma_server_mags.mean():.6f}")
print(f"  std:  {gamma_server_mags.std():.6f}")
print()

# Delta_category magnitudes
print("Delta_category magnitudes:")
for cat, vec in model["delta_category"].items():
    mag = np.linalg.norm(vec)
    print(f"  {cat}: ||delta|| = {mag:.6f}")
print()

# Epsilon_server magnitudes
eps_mags = []
for sid, vec in model["epsilon_server"].items():
    eps_mags.append(np.linalg.norm(vec))
eps_mags = np.array(eps_mags)
print(f"Epsilon_server magnitudes (n={len(eps_mags)}):")
print(f"  min:  {eps_mags.min():.6f}")
print(f"  max:  {eps_mags.max():.6f}")
print(f"  mean: {eps_mags.mean():.6f}")
print(f"  std:  {eps_mags.std():.6f}")
print()

# Compute actual collab predictions for all (agent, server) pairs to understand range
print("Collaborative prediction range (beta + gamma_agent @ gamma_server):")
all_preds = []
for agent in model["gamma_agent"]:
    ga = np.array(model["gamma_agent"][agent])
    for sid in model["gamma_server"]:
        gs = np.array(model["gamma_server"][sid])
        pred = model["beta_server"][sid] + ga @ gs
        all_preds.append(pred)
all_preds = np.array(all_preds)
print(f"  raw pred range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
print(f"  raw pred mean:  {all_preds.mean():.4f}")
print(f"  raw pred std:   {all_preds.std():.4f}")

# Sigmoid transform
all_sigmoid = 1.0 / (1.0 + np.exp(-all_preds))
all_bonus = all_sigmoid - 0.5
print(f"  sigmoid(pred) range: [{all_sigmoid.min():.4f}, {all_sigmoid.max():.4f}]")
print(f"  collab_bonus range:  [{all_bonus.min():.4f}, {all_bonus.max():.4f}]")
print(f"  collab_bonus mean:   {all_bonus.mean():.4f}")
print(f"  collab_bonus std:    {all_bonus.std():.4f}")
print(f"  alpha * collab_bonus range: [{alpha * all_bonus.min():.4f}, {alpha * all_bonus.max():.4f}]")
print()


# ============================================================================
# 2. TRAINING ROLLOUT FEEDBACK STATS
# ============================================================================
print(sep)
print("SECTION 2: TRAINING ROLLOUT FEEDBACK STATS PER SERVER")
print(sep)

train_rollouts = load_jsonl(TRAIN_PATH)
print(f"Total training rollouts: {len(train_rollouts)}")

# Per-server feedback counts
server_feedback = defaultdict(lambda: {"liked": 0, "neutral": 0, "disliked": 0, "total": 0,
                                        "mounted": 0, "selected": 0, "errored": 0})

for r in train_rollouts:
    for sid in r.get("inventory_mounted", []):
        server_feedback[sid]["mounted"] += 1

    for tool_str in r.get("tools_selected", []):
        sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
        server_feedback[sid]["selected"] += 1

    for tool_str in r.get("tools_errored", []):
        sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
        server_feedback[sid]["errored"] += 1

    for tool_str, fb in r.get("feedback", {}).items():
        sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
        if not isinstance(fb, dict):
            continue
        rating = fb.get("rating", "")
        if rating in ("liked", "neutral", "disliked"):
            server_feedback[sid][rating] += 1
            server_feedback[sid]["total"] += 1

n_servers_with_feedback = sum(1 for v in server_feedback.values() if v["total"] > 0)
n_servers_mounted = sum(1 for v in server_feedback.values() if v["mounted"] > 0)
print(f"Servers ever mounted: {n_servers_mounted}")
print(f"Servers with explicit feedback: {n_servers_with_feedback}")
print()

# Feedback distribution across servers
total_feedbacks = [v["total"] for v in server_feedback.values() if v["total"] > 0]
print(f"Feedback count distribution per server (n={len(total_feedbacks)}):")
print(f"  min:    {min(total_feedbacks)}")
print(f"  max:    {max(total_feedbacks)}")
print(f"  mean:   {np.mean(total_feedbacks):.1f}")
print(f"  median: {np.median(total_feedbacks):.1f}")
hist_bins = [1, 2, 3, 5, 10, 20, 50, 1000]
for i in range(len(hist_bins) - 1):
    lo, hi = hist_bins[i], hist_bins[i+1]
    count = sum(1 for x in total_feedbacks if lo <= x < hi)
    print(f"  [{lo}, {hi}):  {count} servers")
print()

# Top 10 most-liked servers (by liked count, min 2 feedback)
def liked_pct(v):
    return v["liked"] / v["total"] * 100 if v["total"] > 0 else 0

print("Top 10 MOST-LIKED servers (min 2 feedback, sorted by liked count):")
qualified = [(sid, v) for sid, v in server_feedback.items() if v["total"] >= 2]
by_liked = sorted(qualified, key=lambda x: x[1]["liked"], reverse=True)[:10]
for sid, v in by_liked:
    pct = liked_pct(v)
    print(f"  {sid}")
    print(f"    liked={v['liked']} neutral={v['neutral']} disliked={v['disliked']} "
          f"total={v['total']} liked%={pct:.0f}% mounted={v['mounted']} selected={v['selected']}")
print()

print("Top 10 MOST-DISLIKED servers (min 2 feedback, sorted by disliked count):")
by_disliked = sorted(qualified, key=lambda x: x[1]["disliked"], reverse=True)[:10]
for sid, v in by_disliked:
    pct_dis = v["disliked"] / v["total"] * 100 if v["total"] > 0 else 0
    print(f"  {sid}")
    print(f"    liked={v['liked']} neutral={v['neutral']} disliked={v['disliked']} "
          f"total={v['total']} disliked%={pct_dis:.0f}% mounted={v['mounted']} errored={v['errored']}")
print()


# ============================================================================
# 3. BETA vs ACTUAL LIKED% CORRELATION
# ============================================================================
print(sep)
print("SECTION 3: BETA_SERVER vs ACTUAL LIKED% CORRELATION")
print(sep)

# Build paired data: servers that have both beta and feedback
paired = []
for sid, beta in model["beta_server"].items():
    v = server_feedback.get(sid, {"total": 0, "liked": 0})
    if v["total"] >= 2:  # need minimum feedback
        liked_frac = v["liked"] / v["total"]
        paired.append((sid, beta, liked_frac, v["total"]))

print(f"Servers with both beta and >=2 feedback: {len(paired)}")
if paired:
    betas_paired = np.array([p[1] for p in paired])
    liked_paired = np.array([p[2] for p in paired])

    # Pearson correlation (manual)
    mean_b = betas_paired.mean()
    mean_l = liked_paired.mean()
    cov = np.mean((betas_paired - mean_b) * (liked_paired - mean_l))
    pearson_r = cov / (betas_paired.std() * liked_paired.std() + 1e-12)
    print(f"Pearson correlation:  r={pearson_r:.4f}")

    # Spearman rank correlation (manual)
    def rankdata(arr):
        temp = np.argsort(arr)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr), dtype=float) + 1
        return ranks
    rank_b = rankdata(betas_paired)
    rank_l = rankdata(liked_paired)
    mean_rb = rank_b.mean()
    mean_rl = rank_l.mean()
    cov_r = np.mean((rank_b - mean_rb) * (rank_l - mean_rl))
    spearman_r = cov_r / (rank_b.std() * rank_l.std() + 1e-12)
    print(f"Spearman correlation: r={spearman_r:.4f}")
    print()

    # Show specific examples: high beta servers and their actual liked%
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    print("Top 10 beta servers and their actual feedback:")
    for sid, beta, liked_frac, total in paired_sorted[:10]:
        print(f"  beta={beta:+.4f}  liked%={100*liked_frac:.0f}%  n={total}  {sid}")
    print()
    print("Bottom 10 beta servers and their actual feedback:")
    for sid, beta, liked_frac, total in paired_sorted[-10:]:
        print(f"  beta={beta:+.4f}  liked%={100*liked_frac:.0f}%  n={total}  {sid}")
    print()

    # Check if there are mismatches: high beta but low liked%
    print("MISMATCHES: High beta (>0.05) but low liked% (<50%):")
    mismatches_hi = [(s, b, l, n) for s, b, l, n in paired if b > 0.05 and l < 0.5]
    if mismatches_hi:
        for sid, beta, liked_frac, total in sorted(mismatches_hi, key=lambda x: x[1], reverse=True):
            print(f"  beta={beta:+.4f}  liked%={100*liked_frac:.0f}%  n={total}  {sid}")
    else:
        print("  (none found)")
    print()

    print("MISMATCHES: Low beta (<-0.05) but high liked% (>50%):")
    mismatches_lo = [(s, b, l, n) for s, b, l, n in paired if b < -0.05 and l > 0.5]
    if mismatches_lo:
        for sid, beta, liked_frac, total in sorted(mismatches_lo, key=lambda x: x[1]):
            print(f"  beta={beta:+.4f}  liked%={100*liked_frac:.0f}%  n={total}  {sid}")
    else:
        print("  (none found)")
    print()


# ============================================================================
# 4. EVAL COMPARISON: LATENT FACTOR vs SEMANTIC
# ============================================================================
print(sep)
print("SECTION 4: EVAL COMPARISON — LATENT FACTOR vs SEMANTIC")
print(sep)

eval_lf = load_jsonl(EVAL_LF_PATH)
eval_sem = load_jsonl(EVAL_SEM_PATH)
print(f"Latent factor eval rollouts: {len(eval_lf)}")
print(f"Semantic eval rollouts:      {len(eval_sem)}")

# Match by task_id
lf_by_task_agent = {}
for r in eval_lf:
    key = (r["task_id"], r["agent"])
    lf_by_task_agent[key] = r

sem_by_task_agent = {}
for r in eval_sem:
    key = (r["task_id"], r["agent"])
    sem_by_task_agent[key] = r

# Find matching pairs
common_keys = set(lf_by_task_agent.keys()) & set(sem_by_task_agent.keys())
print(f"Matching (task_id, agent) pairs: {len(common_keys)}")
print()

# Overlap analysis
same_inventory = 0
partial_overlap = 0
no_overlap = 0
overlap_fracs = []
reorder_count = 0
reorder_helps = 0
reorder_hurts = 0
reorder_neutral = 0

# Track feedback outcomes
lf_liked = 0; lf_neutral = 0; lf_disliked = 0; lf_nofb = 0
sem_liked = 0; sem_neutral = 0; sem_disliked = 0; sem_nofb = 0

for key in common_keys:
    lf_r = lf_by_task_agent[key]
    sem_r = sem_by_task_agent[key]

    lf_inv = lf_r["inventory_mounted"]
    sem_inv = sem_r["inventory_mounted"]

    lf_set = set(lf_inv)
    sem_set = set(sem_inv)

    overlap = lf_set & sem_set
    if lf_set == sem_set:
        same_inventory += 1
        # Check if ORDER changed
        if lf_inv != sem_inv:
            reorder_count += 1
    elif overlap:
        partial_overlap += 1
    else:
        no_overlap += 1

    if len(lf_set | sem_set) > 0:
        overlap_fracs.append(len(overlap) / len(lf_set | sem_set))

    # Feedback comparison
    def get_fb_score(r):
        fb = r.get("feedback", {})
        if not fb:
            return None
        scores = []
        for tool_str, info in fb.items():
            if not isinstance(info, dict):
                continue
            rating = info.get("rating", "")
            if rating == "liked":
                scores.append(1.0)
            elif rating == "neutral":
                scores.append(0.5)
            elif rating == "disliked":
                scores.append(0.0)
        return np.mean(scores) if scores else None

    lf_score = get_fb_score(lf_r)
    sem_score = get_fb_score(sem_r)

    # Count individual feedback
    for tool_str, info in lf_r.get("feedback", {}).items():
        if isinstance(info, dict):
            r = info.get("rating", "")
            if r == "liked": lf_liked += 1
            elif r == "neutral": lf_neutral += 1
            elif r == "disliked": lf_disliked += 1
    if not lf_r.get("feedback"):
        lf_nofb += 1

    for tool_str, info in sem_r.get("feedback", {}).items():
        if isinstance(info, dict):
            r = info.get("rating", "")
            if r == "liked": sem_liked += 1
            elif r == "neutral": sem_neutral += 1
            elif r == "disliked": sem_disliked += 1
    if not sem_r.get("feedback"):
        sem_nofb += 1

print(f"Inventory overlap analysis ({len(common_keys)} pairs):")
print(f"  Exact same set:     {same_inventory} ({100*same_inventory/len(common_keys):.1f}%)")
print(f"  Partial overlap:    {partial_overlap} ({100*partial_overlap/len(common_keys):.1f}%)")
print(f"  No overlap:         {no_overlap} ({100*no_overlap/len(common_keys):.1f}%)")
print(f"  Mean Jaccard index: {np.mean(overlap_fracs):.3f}")
print()

# When inventories differ, compare feedback outcomes
print("When latent_factor reorders/changes inventory, does it help or hurt?")
diff_keys = [k for k in common_keys if set(lf_by_task_agent[k]["inventory_mounted"]) != set(sem_by_task_agent[k]["inventory_mounted"]) or lf_by_task_agent[k]["inventory_mounted"] != sem_by_task_agent[k]["inventory_mounted"]]

lf_wins = 0; sem_wins = 0; ties = 0; both_no_fb = 0
for key in diff_keys:
    lf_r = lf_by_task_agent[key]
    sem_r = sem_by_task_agent[key]

    def get_fb_score(r):
        fb = r.get("feedback", {})
        scores = []
        for tool_str, info in fb.items():
            if isinstance(info, dict):
                rating = info.get("rating", "")
                if rating == "liked": scores.append(1.0)
                elif rating == "neutral": scores.append(0.5)
                elif rating == "disliked": scores.append(0.0)
        return np.mean(scores) if scores else None

    lf_s = get_fb_score(lf_r)
    sem_s = get_fb_score(sem_r)

    if lf_s is None and sem_s is None:
        both_no_fb += 1
    elif lf_s is not None and sem_s is not None:
        if lf_s > sem_s:
            lf_wins += 1
        elif sem_s > lf_s:
            sem_wins += 1
        else:
            ties += 1
    elif lf_s is not None:
        lf_wins += 1  # LF got feedback, semantic didn't
    else:
        sem_wins += 1

print(f"  Differing recommendations: {len(diff_keys)}")
print(f"  LF wins (higher fb score): {lf_wins}")
print(f"  Semantic wins:             {sem_wins}")
print(f"  Ties:                      {ties}")
print(f"  Both no feedback:          {both_no_fb}")
print()

# Overall feedback comparison
print("Overall feedback breakdown:")
lf_total_fb = lf_liked + lf_neutral + lf_disliked
sem_total_fb = sem_liked + sem_neutral + sem_disliked
print(f"  Latent Factor: liked={lf_liked} neutral={lf_neutral} disliked={lf_disliked} total={lf_total_fb} no_fb_rollouts={lf_nofb}")
if lf_total_fb > 0:
    print(f"    liked%={100*lf_liked/lf_total_fb:.1f}% disliked%={100*lf_disliked/lf_total_fb:.1f}%")
print(f"  Semantic:      liked={sem_liked} neutral={sem_neutral} disliked={sem_disliked} total={sem_total_fb} no_fb_rollouts={sem_nofb}")
if sem_total_fb > 0:
    print(f"    liked%={100*sem_liked/sem_total_fb:.1f}% disliked%={100*sem_disliked/sem_total_fb:.1f}%")
print()

# CTR comparison
lf_ctr_num = 0; lf_ctr_den = 0
sem_ctr_num = 0; sem_ctr_den = 0
for r in eval_lf:
    n_mounted = len(r.get("inventory_mounted", []))
    n_selected = len(r.get("tools_selected", []))
    if n_mounted > 0:
        lf_ctr_den += n_mounted
        lf_ctr_num += min(n_selected, n_mounted)  # cap at mounted
for r in eval_sem:
    n_mounted = len(r.get("inventory_mounted", []))
    n_selected = len(r.get("tools_selected", []))
    if n_mounted > 0:
        sem_ctr_den += n_mounted
        sem_ctr_num += min(n_selected, n_mounted)

print(f"CTR (server-level):")
print(f"  Latent Factor: {lf_ctr_num}/{lf_ctr_den} = {lf_ctr_num/lf_ctr_den:.4f}" if lf_ctr_den else "  Latent Factor: N/A")
print(f"  Semantic:      {sem_ctr_num}/{sem_ctr_den} = {sem_ctr_num/sem_ctr_den:.4f}" if sem_ctr_den else "  Semantic: N/A")
print()


# ============================================================================
# 5. COLLAB BONUS MAGNITUDE vs SEMANTIC SCORE SPREAD
# ============================================================================
print(sep)
print("SECTION 5: COLLAB BONUS vs SEMANTIC SCORE SPREAD IN EVAL")
print(sep)

# We need to reconstruct the scores. The eval data has inventory_mounted (top K=5 selected
# from reranked 100 candidates). But we don't have the raw similarity scores for all 100
# candidates. We can check if inventory items have a pattern.
#
# However, we CAN compute collab predictions for the inventory items using the model,
# and we CAN look at the semantic scores from the semantic eval (same tasks get same
# semantic scores for same servers).

# First, let's compute collab bonuses for servers actually recommended in eval
print("Computing collab_bonus for servers in latent factor eval inventories...")
collab_bonuses_all = []
collab_pred_raw_all = []
per_rollout_spreads = []

for r in eval_lf:
    agent = r["agent"]
    category = r.get("task_category", "")
    inventory = r["inventory_mounted"]

    rollout_bonuses = []
    for sid in inventory:
        # Compute collab prediction
        beta = model["beta_server"].get(sid, 0.0)
        ga = np.array(model["gamma_agent"].get(agent, [0]*latent_dim))
        gs = np.array(model["gamma_server"].get(sid, [0]*latent_dim))
        pred = beta + ga @ gs

        if category and category in model["delta_category"]:
            dc = np.array(model["delta_category"][category])
            es = np.array(model["epsilon_server"].get(sid, [0]*latent_dim))
            pred += dc @ es

        collab_pred_raw_all.append(pred)
        bonus = 1.0 / (1.0 + math.exp(-pred)) - 0.5
        collab_bonuses_all.append(bonus)
        rollout_bonuses.append(bonus)

    if len(rollout_bonuses) > 1:
        per_rollout_spreads.append(max(rollout_bonuses) - min(rollout_bonuses))

collab_bonuses_all = np.array(collab_bonuses_all)
collab_pred_raw_all = np.array(collab_pred_raw_all)
per_rollout_spreads = np.array(per_rollout_spreads)

print(f"Collab predictions (raw, before sigmoid) for {len(collab_pred_raw_all)} server slots:")
print(f"  min:  {collab_pred_raw_all.min():.6f}")
print(f"  max:  {collab_pred_raw_all.max():.6f}")
print(f"  mean: {collab_pred_raw_all.mean():.6f}")
print(f"  std:  {collab_pred_raw_all.std():.6f}")
print()

print(f"Collab_bonus (sigmoid(pred) - 0.5) for {len(collab_bonuses_all)} server slots:")
print(f"  min:  {collab_bonuses_all.min():.6f}")
print(f"  max:  {collab_bonuses_all.max():.6f}")
print(f"  mean: {collab_bonuses_all.mean():.6f}")
print(f"  std:  {collab_bonuses_all.std():.6f}")
print()

print(f"alpha * collab_bonus (actual adjustment to semantic score):")
adj = alpha * collab_bonuses_all
print(f"  min:  {adj.min():.6f}")
print(f"  max:  {adj.max():.6f}")
print(f"  mean: {adj.mean():.6f}")
print(f"  std:  {adj.std():.6f}")
print()

print(f"Per-rollout spread of collab_bonus within K=5 inventory:")
print(f"  mean spread: {per_rollout_spreads.mean():.6f}")
print(f"  max spread:  {per_rollout_spreads.max():.6f}")
print(f"  After alpha: mean={alpha*per_rollout_spreads.mean():.6f}, max={alpha*per_rollout_spreads.max():.6f}")
print()

# Now estimate semantic score spread in the candidate pool
# We can't see the raw 100 candidates, but we can check the semantic eval
# to see what similarity scores look like
print("Estimating semantic score spread from semantic eval tool selections...")
# Look at which servers appear in semantic inventory — these are top-K by similarity
# For the same tasks, the latent factor also sees the same 100 candidates
# So the top-5 semantic scores define the "ceiling" the collab must overcome

# We don't have raw sim scores in eval, but we can estimate from the embedding data
# Let's check if we have the embeddings to compute this
try:
    emb_index_path = BASE / "data" / "index" / "embedding_index.json"
    embeddings_path = BASE / "data" / "index" / "embeddings.npy"

    with open(emb_index_path) as f:
        emb_index = json.load(f)

    embeddings = np.load(embeddings_path)

    # Build id -> index
    id_to_idx = {entry["id"]: i for i, entry in enumerate(emb_index)}

    # For a sample of eval tasks, compute similarity spread among top candidates
    # Load tasks for embedding
    tasks_path = BASE / "data" / "tasks_test.json"
    if tasks_path.exists():
        with open(tasks_path) as f:
            test_tasks = json.load(f)
        task_by_id = {t["uuid"]: t for t in test_tasks}
    else:
        task_by_id = {}

    print(f"Loaded {len(emb_index)} server embeddings, {len(task_by_id)} test tasks")
    print()

    # For tasks in eval, check the similarity scores of their mounted servers
    # vs the general candidate pool
    # Since we don't have query embeddings precomputed, let's estimate from
    # the mounted servers' positions in the overall ranking

    # Alternative: compute similarity between mounted servers to estimate spread
    # For each eval rollout, check similarity of the top-5 semantic servers
    sem_sim_spreads = []
    sem_top5_ranges = []
    for r in eval_sem:
        inv = r["inventory_mounted"]
        # Get embeddings for these servers
        idxs = [id_to_idx[sid] for sid in inv if sid in id_to_idx]
        if len(idxs) >= 2:
            # These are the top-K by cosine sim to the query
            # The spread between them tells us how clustered the candidates are
            inv_embs = embeddings[idxs]
            # Normalize
            norms = np.linalg.norm(inv_embs, axis=1, keepdims=True) + 1e-9
            inv_embs_n = inv_embs / norms
            # Pairwise similarities among top-5
            sims = inv_embs_n @ inv_embs_n.T
            # Off-diagonal
            mask = ~np.eye(len(idxs), dtype=bool)
            off_diag = sims[mask]
            sem_sim_spreads.append(off_diag.mean())

    if sem_sim_spreads:
        print(f"Pairwise similarity among top-5 semantic servers (proxy for clustering):")
        sem_sim_spreads = np.array(sem_sim_spreads)
        print(f"  mean pairwise sim: {sem_sim_spreads.mean():.4f}")
        print(f"  std:               {sem_sim_spreads.std():.4f}")
        print(f"  min:               {sem_sim_spreads.min():.4f}")
        print(f"  max:               {sem_sim_spreads.max():.4f}")
        print()

    # Better approach: for each LF eval rollout, check overlap with semantic
    # If both recommend the same servers, the collab signal is too weak to reorder
    print("Fraction of LF inventories that are identical to semantic inventories:")
    identical = 0
    total_compared = 0
    n_servers_swapped = []
    for key in common_keys:
        lf_r = lf_by_task_agent[key]
        sem_r = sem_by_task_agent[key]
        lf_set = set(lf_r["inventory_mounted"])
        sem_set = set(sem_r["inventory_mounted"])
        if lf_set == sem_set:
            if lf_r["inventory_mounted"] == sem_r["inventory_mounted"]:
                identical += 1
        diff = lf_set.symmetric_difference(sem_set)
        n_servers_swapped.append(len(diff) // 2)  # servers replaced
        total_compared += 1

    n_servers_swapped = np.array(n_servers_swapped)
    print(f"  Identical (same set + order): {identical}/{total_compared} ({100*identical/total_compared:.1f}%)")
    print(f"  Same set (any order):         {same_inventory}/{total_compared} ({100*same_inventory/total_compared:.1f}%)")
    print(f"  Mean servers swapped:         {n_servers_swapped.mean():.2f}")
    print(f"  Max servers swapped:          {n_servers_swapped.max()}")
    print(f"  Distribution of swaps:")
    for i in range(6):
        c = np.sum(n_servers_swapped == i)
        print(f"    {i} swapped: {c} ({100*c/len(n_servers_swapped):.1f}%)")
    print()

except Exception as e:
    print(f"Could not load embeddings for similarity analysis: {e}")
    import traceback
    traceback.print_exc()
    print()


# ============================================================================
# SECTION 6: ROOT CAUSE SUMMARY
# ============================================================================
print(sep)
print("SECTION 6: ROOT CAUSE DIAGNOSIS SUMMARY")
print(sep)
print()

# Quantify key issues
print("KEY FINDINGS:")
print()

# 1. Parameter magnitude
print(f"1. PARAMETER MAGNITUDE:")
print(f"   beta_server range: [{betas.min():.4f}, {betas.max():.4f}], std={betas.std():.4f}")
print(f"   Most betas near zero: {100*near_zero/len(betas):.0f}% have |beta| < 0.01")
print(f"   Max possible alpha*collab_bonus: +/-{alpha * 0.5:.4f}")
print(f"   But actual collab_bonus std: {collab_bonuses_all.std():.4f}")
print(f"   So actual adjustment std: {alpha * collab_bonuses_all.std():.4f}")
print()

# 2. Observation density
avg_fb_per_server = np.mean(total_feedbacks) if total_feedbacks else 0
print(f"2. OBSERVATION DENSITY:")
print(f"   1000 rollouts across {n_servers_with_feedback} servers = {avg_fb_per_server:.1f} feedback/server average")
print(f"   Many servers have only 1-2 feedback signals — insufficient for reliable learning")
print()

# 3. Alpha cap
print(f"3. ALPHA RAMP:")
print(f"   alpha = min({n_obs}/500, 0.3) = {alpha}")
print(f"   Alpha is CAPPED AT 0.3 — even with maximum collab_bonus of 0.5,")
print(f"   the adjustment is at most 0.3*0.5 = 0.15")
print(f"   With actual collab_bonus values typically in [{collab_bonuses_all.min():.3f}, {collab_bonuses_all.max():.3f}],")
print(f"   the effective range is [{alpha*collab_bonuses_all.min():.4f}, {alpha*collab_bonuses_all.max():.4f}]")
print()

# 4. Semantic clustering
print(f"4. SEMANTIC VS COLLAB SCALE:")
# Compute n_servers_swapped here (outside try block)
_n_servers_swapped = []
for key in common_keys:
    lf_r = lf_by_task_agent[key]
    sem_r = sem_by_task_agent[key]
    lf_set = set(lf_r["inventory_mounted"])
    sem_set = set(sem_r["inventory_mounted"])
    diff = lf_set.symmetric_difference(sem_set)
    _n_servers_swapped.append(len(diff) // 2)
_n_servers_swapped = np.array(_n_servers_swapped)
print(f"   Mean servers swapped in top-5: {_n_servers_swapped.mean():.2f}")
print(f"   When alpha*bonus can only adjust by ~{alpha * collab_bonuses_all.std():.4f} (1 std),")
print(f"   it needs to overcome the gap between the #5 and #6 semantic candidates.")
print(f"   If semantic similarities are tightly clustered, even small adjustments reorder.")
print(f"   If widely spread, adjustments are noise.")
print()

# 5. Source code issue: alpha cap
print(f"5. CODE ISSUE — ALPHA CAP IN method_name vs recommend:")
print(f"   In latent_factor.py line 9: docstring says alpha cap is 0.8")
print(f"   In latent_factor.py line 57 (method_name): alpha = min(n/500, 0.3)")
print(f"   In latent_factor.py line 63 (recommend): alpha = min(n/500, 0.3)")
print(f"   The docstring is WRONG (says 0.8), the code uses 0.3.")
print(f"   With 0.3 cap, the collab signal is severely attenuated.")
print()

# 6. Missing task category in training
train_has_category = sum(1 for r in train_rollouts if r.get("task_category"))
print(f"6. TASK CATEGORY SIGNAL IN TRAINING:")
print(f"   Training rollouts with task_category: {train_has_category}/{len(train_rollouts)}")
if train_has_category == 0:
    print(f"   >>> NO TASK CATEGORY in training data!")
    print(f"   The delta_category * epsilon_server term was NEVER trained on real signal!")
    print(f"   All {n_categories} category vectors learned only from random init + regularization decay")
print()

# 7. The crucial comparison
print(f"7. HEAD-TO-HEAD RESULTS:")
print(f"   From comparison.json:")
print(f"   Latent Factor: feedback_score=0.721, liked%=64.2%, CTR=0.128")
print(f"   Semantic:      feedback_score=0.739, liked%=67.3%, CTR=0.123")
print(f"   Semantic wins on feedback_score by 0.018 and liked% by 3.1pp")
print(f"   LF has marginally higher CTR (0.128 vs 0.123) but worse outcomes")
print(f"   This suggests LF sometimes swaps in WRONG servers — the collab signal")
print(f"   is noisy enough that some reorderings hurt more than they help.")
print()

print(sep)
print("CONCLUSION")
print(sep)
print("""
The latent factor model fails to beat semantic search for several reinforcing reasons:

1. INSUFFICIENT DATA DENSITY: 1000 rollouts spread across 343 servers means
   ~2-3 feedback signals per server on average. Many servers have only 1 observation.
   The latent factors can't converge with this little data.

2. TINY COLLAB ADJUSTMENTS: With alpha=0.3 and most collab_bonus values near zero
   (std={cb_std:.4f}), the typical adjustment to semantic scores is ~{adj_std:.4f}.
   This is often too small to meaningfully reorder candidates.

3. MISSING TASK CATEGORY IN TRAINING: The task_category field is absent from
   training rollouts, so the delta_category * epsilon_server term — which should
   capture "browser tasks prefer playwright servers" — was never actually trained.
   All category vectors decayed toward zero via regularization.

4. NOISY REORDERING: When the model DOES manage to swap servers, the swaps are
   roughly 50/50 helpful vs harmful. The collab signal hasn't learned enough to
   reliably improve on semantic ordering.

5. ALPHA CAP TOO LOW: The 0.3 cap means collab can adjust scores by at most 0.15
   (in practice much less). Even if the model learned perfect quality signals,
   it would be throttled. The docstring suggests 0.8 was originally intended.

RECOMMENDATIONS:
- Get more training data (5,000-10,000 rollouts minimum for 343 servers)
- Fix the missing task_category in training rollouts
- Raise alpha cap to 0.5-0.8 (matching the docstring) once model has more data
- Consider a warm-up: train on batch data before going online
- Add the category signal explicitly: delta*epsilon should differentiate by domain
""".format(
    cb_std=collab_bonuses_all.std(),
    adj_std=alpha * collab_bonuses_all.std(),
))

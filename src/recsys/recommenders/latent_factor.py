"""Latent factor recommender with projected task embeddings.

Model:
    score(agent_a, server_s, task_t) = β_s + γ_a · γ_s + (P · task_emb_t) · ε_s

    β_s:   server bias (scalar per server) — captures "this server is broken/good"
    γ_a:   agent latent factors (latent_dim per agent) — agent preferences
    γ_s:   server latent factors (latent_dim per server) — server characteristics
    P:     projection matrix (latent_dim × emb_dim) — shared, compresses task embedding
    ε_s:   server task-affinity (latent_dim per server) — "good for this type of task"

SGD updates for each observation (agent_a, server_s, task_t, score_y):
    task_proj = P · task_emb_t                    # (latent_dim,)
    predicted = β_s + γ_a · γ_s + task_proj · ε_s
    error = y - predicted

    β_s   += η · (error - λ · β_s)
    γ_a   += η · (error · γ_s - λ · γ_a)
    γ_s   += η · (error · γ_a - λ · γ_s)
    ε_s   += η · (error · task_proj - λ · ε_s)
    P     += η · (error · outer(ε_s, task_emb_t) - λ_P · P)

Task embedding is frozen (from text-embedding-3-small). Only β, γ, ε, P are learned.

Recommendation:
    final_score = semantic_sim + α · clamp(predicted, -1, 1)
    α = min(n_obs / 200, 0.6)  — warm-up from pure semantic to blended
"""

import json
import logging

import numpy as np

from .base import BaseRecommender

log = logging.getLogger(__name__)

_EMB_DIM = 1536  # text-embedding-3-small


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, latent_dim: int = 16, lr: float = 0.01, reg: float = 0.001,
                 reg_P: float = 0.0001, seed: int = 42):
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.reg_P = reg_P
        self.rng = np.random.RandomState(seed)

        # Shared projection: (latent_dim, emb_dim)
        self.P = self.rng.randn(latent_dim, _EMB_DIM) * 0.001

        # Per-server parameters
        self.beta: dict[str, float] = {}                    # β_s
        self.gamma_s: dict[str, np.ndarray] = {}            # γ_s (latent_dim,)
        self.epsilon: dict[str, np.ndarray] = {}            # ε_s (latent_dim,)

        # Per-agent parameters
        self.gamma_a: dict[str, np.ndarray] = {}            # γ_a (latent_dim,)

        self.n_obs = 0

    def _ensure_server(self, server_id: str):
        if server_id not in self.beta:
            self.beta[server_id] = 0.0
            self.gamma_s[server_id] = self.rng.randn(self.latent_dim) * 0.01
            self.epsilon[server_id] = self.rng.randn(self.latent_dim) * 0.01

    def _ensure_agent(self, agent: str):
        if agent not in self.gamma_a:
            self.gamma_a[agent] = self.rng.randn(self.latent_dim) * 0.01

    def _predict(self, agent: str, server_id: str, task_emb=None) -> float:
        self._ensure_agent(agent)
        self._ensure_server(server_id)

        score = self.beta[server_id]
        score += self.gamma_a[agent] @ self.gamma_s[server_id]

        if task_emb is not None:
            task_proj = self.P @ task_emb  # (latent_dim,)
            score += task_proj @ self.epsilon[server_id]

        return score

    @property
    def method_name(self) -> str:
        return f"latent_factor_n{self.n_obs}"

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int,
                  task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        alpha = min(self.n_obs / 200, 0.6)

        scored = []
        for c in candidates:
            server_id = c["id"]
            semantic = c.get("similarity", 0.0)

            if server_id in self.beta:
                collab_raw = self._predict(agent, server_id, task_emb=task_emb)
                collab = max(-1.0, min(1.0, collab_raw))
                final = semantic + alpha * collab
            else:
                final = semantic

            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:k]]

        # Epsilon-greedy exploration: replace some top-K picks with random candidates
        if epsilon > 0.0 and len(scored) > k:
            n_explore = max(1, round(k * epsilon))
            if self.rng.random() < epsilon:
                rest = [c for _, c in scored[k:]]
                explore_idx = self.rng.choice(len(rest), size=min(n_explore, len(rest)), replace=False)
                explored = [rest[i] for i in explore_idx]
                result = result[:k - len(explored)] + explored

        return result

    def update(self, rollout: dict, task_emb=None) -> None:
        agent = rollout.get("agent", "")
        if not agent:
            return

        signals = self._extract_signals(rollout)
        if not signals:
            return

        self._ensure_agent(agent)

        for server_id, target in signals.items():
            self._ensure_server(server_id)

            pred = self._predict(agent, server_id, task_emb=task_emb)
            error = target - pred

            # β_s update
            self.beta[server_id] += self.lr * (error - self.reg * self.beta[server_id])

            # γ_a, γ_s updates
            ga = self.gamma_a[agent].copy()
            gs = self.gamma_s[server_id].copy()
            self.gamma_a[agent] += self.lr * (error * gs - self.reg * ga)
            self.gamma_s[server_id] += self.lr * (error * ga - self.reg * gs)

            # ε_s, P updates (only with task embedding)
            if task_emb is not None:
                task_proj = self.P @ task_emb  # (latent_dim,)
                es = self.epsilon[server_id].copy()
                self.epsilon[server_id] += self.lr * (error * task_proj - self.reg * es)
                self.P += self.lr * (error * np.outer(es, task_emb) - self.reg_P * self.P)

        self.n_obs += 1

    def _extract_signals(self, rollout: dict) -> dict[str, float]:
        """Extract per-server training signals from a rollout."""

        def _extract_sids(key):
            sids = set()
            for tool_str in rollout.get(key, []):
                sids.add(tool_str.split(":")[0] if ":" in tool_str else tool_str)
            return sids

        tools_selected = _extract_sids("tools_selected")
        tools_errored = _extract_sids("tools_errored")
        tools_abandoned = _extract_sids("tools_abandoned")

        # Extract per-tool feedback ratings
        feedback = rollout.get("feedback", {})

        # Handle relevance check format (tools_relevant: bool)
        if isinstance(feedback, dict) and "tools_relevant" in feedback:
            if not feedback["tools_relevant"]:
                # Agent said tools were irrelevant — weak negative for all mounted
                signals = {}
                for sid in rollout.get("inventory_mounted", []):
                    signals[sid] = -0.3
                for sid in rollout.get("inventory_failed", []):
                    signals[sid] = -3.0
                return signals
            else:
                # Agent said tools were relevant but didn't use them — no signal
                signals = {}
                for sid in rollout.get("inventory_failed", []):
                    signals[sid] = -3.0
                return signals

        # Standard per-tool ratings — accumulate all tool signals per server, then average
        RATING_VALUES = {"liked": 1.0, "neutral": 0.2, "disliked": -1.0}
        server_feedback: dict[str, list[float]] = {}
        for tool_str, fb in feedback.items():
            sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
            if isinstance(fb, dict):
                rating = fb.get("rating", "")
                if rating in RATING_VALUES:
                    server_feedback.setdefault(sid, []).append(RATING_VALUES[rating])

        signals: dict[str, float] = {}

        # Mount failures — very strong negative (server is completely useless)
        for sid in rollout.get("inventory_failed", []):
            signals[sid] = -3.0

        # Tools that were selected and used
        for sid in tools_selected:
            if sid in signals:
                continue
            if sid in server_feedback:
                signals[sid] = sum(server_feedback[sid]) / len(server_feedback[sid])
            elif sid in tools_errored:
                signals[sid] = -0.5
            elif sid in tools_abandoned:
                signals[sid] = -0.4
            else:
                signals[sid] = 0.1  # selected but no explicit feedback

        # Mounted but agent didn't select any tools — weak negative CTR
        if not tools_selected:
            for sid in rollout.get("inventory_mounted", []):
                if sid not in signals:
                    signals[sid] = -0.1

        return signals

    def save(self, path: str):
        data = {
            "latent_dim": self.latent_dim,
            "n_obs": self.n_obs,
            "P": self.P.tolist(),
            "beta": self.beta,
            "gamma_s": {sid: v.tolist() for sid, v in self.gamma_s.items()},
            "gamma_a": {agent: v.tolist() for agent, v in self.gamma_a.items()},
            "epsilon": {sid: v.tolist() for sid, v in self.epsilon.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        log.info(f"Model saved: {self.n_obs} obs, {len(self.gamma_a)} agents, "
                 f"{len(self.beta)} servers → {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)

        self.latent_dim = data.get("latent_dim", 16)
        self.n_obs = data.get("n_obs", 0)
        self.P = np.array(data["P"])
        self.beta = data["beta"]
        self.gamma_s = {sid: np.array(v) for sid, v in data["gamma_s"].items()}
        self.gamma_a = {agent: np.array(v) for agent, v in data["gamma_a"].items()}
        self.epsilon = {sid: np.array(v) for sid, v in data["epsilon"].items()}

        log.info(f"Model loaded: {self.n_obs} obs, {len(self.gamma_a)} agents, "
                 f"{len(self.beta)} servers from {path}")

"""Latent factor recommender — fully per-agent factorization.

Each agent gets its own parameter set: no shared parameters across agents.

Model per agent:
    score(server, task_emb) =
        β_{agent,server}                         # per-agent server quality bias
        + (W_agent^T · task_emb) · ε_{agent,server}  # per-agent task-server relevance

    W_agent: (emb_dim × latent_dim) — per-agent task projection
    ε_{agent,server}: (latent_dim,) — server's task-affinity for this agent

    final = semantic_sim + α * clamp(score, -1, 1)
    α = min(n_obs_agent / 500, 0.5)  — per-agent warm-up
"""

import json
import logging

import numpy as np

from .base import BaseRecommender

log = logging.getLogger(__name__)

_EMB_DIM = 1536  # text-embedding-3-small


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, latent_dim: int = 8, lr: float = 0.01, reg: float = 0.001,
                 reg_W: float = 0.001, seed: int = 42):
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.reg_W = reg_W
        self.rng = np.random.RandomState(seed)

        # Per-agent parameters: agent → {server_id → value}
        self.beta: dict[str, dict[str, float]] = {}          # β_{agent, server}
        self.epsilon: dict[str, dict[str, np.ndarray]] = {}  # ε_{agent, server}
        self.W: dict[str, np.ndarray] = {}                   # W_agent (emb_dim, latent_dim)

        # Per-agent observation counts for alpha warm-up
        self.n_obs: dict[str, int] = {}

    def _ensure_params(self, agent: str, server_id: str):
        if agent not in self.beta:
            self.beta[agent] = {}
            self.epsilon[agent] = {}
            self.W[agent] = np.zeros((_EMB_DIM, self.latent_dim))
            self.n_obs[agent] = 0
        if server_id not in self.beta[agent]:
            self.beta[agent][server_id] = 0.0
            self.epsilon[agent][server_id] = self.rng.randn(self.latent_dim) * 0.01

    def _predict(self, agent: str, server_id: str, task_emb=None) -> float:
        self._ensure_params(agent, server_id)
        score = self.beta[agent][server_id]
        if task_emb is not None:
            task_proj = self.W[agent].T @ task_emb  # (latent_dim,)
            score += task_proj @ self.epsilon[agent][server_id]
        return score

    @property
    def method_name(self) -> str:
        total_obs = sum(self.n_obs.values())
        return f"latent_factor_per_agent_n{total_obs}"

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int,
                  task_category: str = "", task_emb=None) -> list[dict]:
        n_obs = self.n_obs.get(agent, 0)
        alpha = min(n_obs / 500, 0.5)

        scored = []
        for c in candidates:
            server_id = c["id"]
            semantic = c.get("similarity", 0.0)
            if agent in self.beta and server_id in self.beta[agent]:
                collab_raw = self._predict(agent, server_id, task_emb=task_emb)
                collab_bonus = max(-1.0, min(1.0, collab_raw))
                final = semantic + alpha * collab_bonus
            else:
                final = semantic
            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def update(self, rollout: dict, task_emb=None) -> None:
        agent = rollout.get("agent", "")
        if not agent:
            return

        def _extract_sids(key):
            sids = set()
            for tool_str in rollout.get(key, []):
                sids.add(tool_str.split(":")[0] if ":" in tool_str else tool_str)
            return sids

        tools_selected = _extract_sids("tools_selected")
        tools_errored = _extract_sids("tools_errored")
        tools_abandoned = _extract_sids("tools_abandoned")

        server_feedback: dict[str, str] = {}
        for tool_str, fb in rollout.get("feedback", {}).items():
            sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
            if isinstance(fb, dict):
                server_feedback[sid] = fb.get("rating", "")

        signals: dict[str, float] = {}

        for sid in rollout.get("inventory_failed", []):
            signals[sid] = -1.0

        for sid in tools_selected:
            if sid in signals:
                continue
            if sid in server_feedback:
                rating = server_feedback[sid]
                if rating == "liked":
                    signals[sid] = 1.0
                elif rating == "neutral":
                    signals[sid] = 0.2
                elif rating == "disliked":
                    signals[sid] = -0.8
            elif sid in tools_errored:
                signals[sid] = -0.7
            elif sid in tools_abandoned:
                signals[sid] = -0.4
            else:
                signals[sid] = 0.1

        # No-tool penalty: agent ignored the entire inventory — weak negative CTR signal
        if not tools_selected:
            for mount in rollout.get("inventory_mounted", []):
                sid = mount.split(":")[0] if ":" in mount else mount
                if sid not in signals:  # don't override inventory_failed
                    signals[sid] = -0.1

        # SGD updates — per-agent params only
        for server_id, target in signals.items():
            self._ensure_params(agent, server_id)
            pred = self._predict(agent, server_id, task_emb=task_emb)
            error = target - pred

            self.beta[agent][server_id] += self.lr * (error - self.reg * self.beta[agent][server_id])

            if task_emb is not None:
                es = self.epsilon[agent][server_id].copy()
                task_proj = self.W[agent].T @ task_emb
                self.W[agent] += self.lr * (error * np.outer(task_emb, es) - self.reg_W * self.W[agent])
                self.epsilon[agent][server_id] += self.lr * (error * task_proj - self.reg * es)

        self.n_obs[agent] = self.n_obs.get(agent, 0) + 1

    def save(self, path: str):
        data = {
            "latent_dim": self.latent_dim,
            "n_obs": self.n_obs,
            "beta": self.beta,
            "epsilon": {
                agent: {sid: v.tolist() for sid, v in smap.items()}
                for agent, smap in self.epsilon.items()
            },
            "W": {agent: m.tolist() for agent, m in self.W.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        total = sum(self.n_obs.values())
        log.info(f"Model saved: {total} total obs, {len(self.beta)} agents → {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)

        # Support both old (global) and new (per-agent) formats
        if "beta" in data and isinstance(data["beta"], dict) and \
                any(isinstance(v, dict) for v in data["beta"].values()):
            # New per-agent format
            self.n_obs = data.get("n_obs", {})
            self.beta = data["beta"]
            self.epsilon = {
                agent: {sid: np.array(v) for sid, v in smap.items()}
                for agent, smap in data.get("epsilon", {}).items()
            }
            self.W = {agent: np.array(m) for agent, m in data.get("W", {}).items()}
        else:
            # Legacy format: promote global params under a "__global__" key
            log.warning("Loading legacy model format — global params promoted to '__global__' agent")
            self.n_obs = {"__global__": data.get("n_observations", 0)}
            self.beta = {"__global__": data.get("beta_server", {})}
            self.epsilon = {
                "__global__": {sid: np.array(v) for sid, v in data.get("epsilon_server", {}).items()}
            }
            W_raw = data.get("W")
            self.W = {"__global__": np.array(W_raw) if W_raw else np.zeros((_EMB_DIM, self.latent_dim))}

        total = sum(self.n_obs.values())
        log.info(f"Model loaded: {total} total obs, {len(self.beta)} agents from {path}")

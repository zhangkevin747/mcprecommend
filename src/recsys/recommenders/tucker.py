"""Tucker recommender: bilinear agent-task-server factorization.

Model:
    R̂(a, t, s) = content(t, s) + α * (b_s + b_a + Tucker(a, t, s))

    content(t, s) = cosine_sim(task_emb, server_emb)  [pre-computed, cold-start]
    Tucker(a, t, s) = (T.T @ task_emb) · (u_a ⊙ w_s)

    Shared parameters (trained on ALL agents' data):
        T:   (emb_dim × r)  — task projector
        b_s: dict[server_id → float]  — server quality bias
        w_s: dict[server_id → ndarray(r)]  — server latent factors

    Per-agent parameters:
        b_a: dict[agent → float]  — agent rating bias
        u_a: dict[agent → ndarray(r)]  — agent latent factors

    α = min(n_obs_agent / 500, 0.5)  per-agent warm-up

Key improvement over v10:
    T and w_s are shared, so every SGD step propagates across agents and tasks.
    b_s is global, so quality signal from all 3 agents pools together.
"""

import json
import logging

import numpy as np

from .base import BaseRecommender

log = logging.getLogger(__name__)

_EMB_DIM = 1536  # text-embedding-3-small


class TuckerRecommender(BaseRecommender):
    def __init__(self, latent_dim: int = 8, lr: float = 0.01, lr_tucker: float = None,
                 reg: float = 0.001, reg_T: float = 0.0001, seed: int = 42):
        self.r = latent_dim
        self.lr = lr
        self.lr_tucker = lr_tucker if lr_tucker is not None else lr  # separate lr for Tucker factors
        self.reg = reg
        self.reg_T = reg_T
        self.rng = np.random.RandomState(seed)

        # Shared parameters
        # Initialize T with small random noise (not zeros) so u_a/w_s gradients are non-zero from step 1
        self.T = self.rng.randn(_EMB_DIM, self.r) * 0.001  # task projector
        self.b_s: dict[str, float] = {}                 # server quality bias
        self.w_s: dict[str, np.ndarray] = {}            # server latent factor

        # Per-agent parameters
        self.b_a: dict[str, float] = {}                 # agent rating bias
        self.u_a: dict[str, np.ndarray] = {}            # agent latent factor

        # Observation counts
        self.n_obs: dict[str, int] = {}

    def _ensure_params(self, agent: str, server_id: str):
        if agent not in self.b_a:
            self.b_a[agent] = 0.0
            self.u_a[agent] = self.rng.randn(self.r) * 0.1
            self.n_obs[agent] = 0
        if server_id not in self.b_s:
            self.b_s[server_id] = 0.0
            self.w_s[server_id] = self.rng.randn(self.r) * 0.1

    def _predict(self, agent: str, server_id: str, task_emb=None) -> float:
        self._ensure_params(agent, server_id)
        score = self.b_s[server_id] + self.b_a[agent]
        if task_emb is not None:
            task_proj = self.T.T @ task_emb          # (r,)
            score += task_proj @ (self.u_a[agent] * self.w_s[server_id])
        return score

    @property
    def method_name(self) -> str:
        total_obs = sum(self.n_obs.values())
        return f"tucker_n{total_obs}"

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int,
                  task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        n_obs = self.n_obs.get(agent, 0)
        alpha = min(n_obs / 500, 0.5)

        scored = []
        for c in candidates:
            server_id = c["id"]
            semantic = c.get("similarity", 0.0)
            if agent in self.b_a and server_id in self.b_s:
                collab = self._predict(agent, server_id, task_emb=task_emb)
                final = semantic + alpha * max(-1.0, min(1.0, collab))
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

        for server_id, target in signals.items():
            self._ensure_params(agent, server_id)
            pred = self._predict(agent, server_id, task_emb=task_emb)
            error = target - pred

            # Bias updates (global server bias, per-agent bias)
            self.b_s[server_id] += self.lr * (error - self.reg * self.b_s[server_id])
            self.b_a[agent] += self.lr * (error - self.reg * self.b_a[agent])

            # Tucker interaction updates
            if task_emb is not None:
                task_proj = self.T.T @ task_emb                        # (r,)
                u = self.u_a[agent]
                w = self.w_s[server_id]

                # Shared T: gradient = error * outer(task_emb, u ⊙ w)
                self.T += self.lr_tucker * (error * np.outer(task_emb, u * w) - self.reg_T * self.T)

                # Per-agent u_a and shared w_s
                u_grad = error * task_proj * w - self.reg * u
                w_grad = error * task_proj * u - self.reg * w
                self.u_a[agent] += self.lr_tucker * u_grad
                self.w_s[server_id] += self.lr_tucker * w_grad

        self.n_obs[agent] = self.n_obs.get(agent, 0) + 1

    def save(self, path: str):
        data = {
            "latent_dim": self.r,
            "n_obs": self.n_obs,
            "T": self.T.tolist(),
            "b_s": self.b_s,
            "b_a": self.b_a,
            "w_s": {sid: v.tolist() for sid, v in self.w_s.items()},
            "u_a": {agent: v.tolist() for agent, v in self.u_a.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        total = sum(self.n_obs.values())
        log.info(f"Tucker model saved: {total} total obs, {len(self.b_s)} servers → {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.r = data.get("latent_dim", self.r)
        self.n_obs = data.get("n_obs", {})
        T_raw = data.get("T")
        self.T = np.array(T_raw) if T_raw else self.rng.randn(_EMB_DIM, self.r) * 0.001
        self.b_s = data.get("b_s", {})
        self.b_a = data.get("b_a", {})
        self.w_s = {sid: np.array(v) for sid, v in data.get("w_s", {}).items()}
        self.u_a = {agent: np.array(v) for agent, v in data.get("u_a", {}).items()}
        total = sum(self.n_obs.values())
        log.info(f"Tucker model loaded: {total} total obs, {len(self.b_s)} servers from {path}")

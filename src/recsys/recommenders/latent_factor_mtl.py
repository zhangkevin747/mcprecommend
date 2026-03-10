"""Multi-task latent factor recommender with 3 heads.

Shared representation (same as latent_factor.py):
    h(a, s, τ) = [β_s, γ_a · γ_s, (P · φ_τ) · ε_s]   — 3-dim feature vector

Three task-specific heads:
    Head 1 — mountability:   P(mount)    = σ(W₁ · h + b₁)
    Head 2 — task relevance: P(relevant) = σ(W₂ · h + b₂)
    Head 3 — engagement:     P(engaged)  = σ(W₃ · h + b₃)

Joint training loss:
    L = λ₁ L_mount + λ₂ L_relevant + λ₃ L_engaged

Final ranking score:
    f̃(a, s, τ) = w₁ P(mount) + w₂ P(relevant) + w₃ P(engaged)

Cold-start blending wraps around this combined score exactly as before.
"""

import json
import logging

import numpy as np

from .base import BaseRecommender

log = logging.getLogger(__name__)

_EMB_DIM = 1536  # text-embedding-3-small
_H_DIM = 3       # [β_s, γ_a · γ_s, task_proj · ε_s]


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + np.exp(-x))


def _bce_grad(pred: float, target: float) -> float:
    """Gradient of BCE loss w.r.t. logit: pred - target (after sigmoid)."""
    return pred - target


class LatentFactorMTLRecommender(BaseRecommender):
    def __init__(self, latent_dim: int = 16, lr: float = 0.01, reg: float = 0.001,
                 reg_P: float = 0.0001, seed: int = 42,
                 lambda_mount: float = 1.0, lambda_relevant: float = 1.0,
                 lambda_engaged: float = 1.0,
                 w_mount: float = 0.4, w_relevant: float = 0.3, w_engaged: float = 0.3):
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.reg_P = reg_P
        self.rng = np.random.RandomState(seed)

        # Loss weights
        self.lambda_mount = lambda_mount
        self.lambda_relevant = lambda_relevant
        self.lambda_engaged = lambda_engaged

        # Ranking combination weights
        self.w_mount = w_mount
        self.w_relevant = w_relevant
        self.w_engaged = w_engaged

        # Shared projection: (latent_dim, emb_dim)
        self.P = self.rng.randn(latent_dim, _EMB_DIM) * 0.001

        # Per-server parameters
        self.beta: dict[str, float] = {}                    # β_s
        self.gamma_s: dict[str, np.ndarray] = {}            # γ_s (latent_dim,)
        self.epsilon: dict[str, np.ndarray] = {}            # ε_s (latent_dim,)

        # Per-agent parameters
        self.gamma_a: dict[str, np.ndarray] = {}            # γ_a (latent_dim,)

        # Head parameters: W (3,) and b (scalar) per head
        self.W1 = self.rng.randn(_H_DIM) * 0.01  # mountability
        self.b1 = 0.0
        self.W2 = self.rng.randn(_H_DIM) * 0.01  # relevance
        self.b2 = 0.0
        self.W3 = self.rng.randn(_H_DIM) * 0.01  # engagement
        self.b3 = 0.0

        self.n_obs = 0

    def _ensure_server(self, server_id: str):
        if server_id not in self.beta:
            self.beta[server_id] = 0.0
            self.gamma_s[server_id] = self.rng.randn(self.latent_dim) * 0.01
            self.epsilon[server_id] = self.rng.randn(self.latent_dim) * 0.01

    def _ensure_agent(self, agent: str):
        if agent not in self.gamma_a:
            self.gamma_a[agent] = self.rng.randn(self.latent_dim) * 0.01

    def _compute_h(self, agent: str, server_id: str, task_emb=None) -> np.ndarray:
        """Compute shared representation h = [β_s, γ_a · γ_s, task_proj · ε_s]."""
        self._ensure_agent(agent)
        self._ensure_server(server_id)

        h = np.zeros(_H_DIM)
        h[0] = self.beta[server_id]
        h[1] = self.gamma_a[agent] @ self.gamma_s[server_id]
        if task_emb is not None:
            task_proj = self.P @ task_emb
            h[2] = task_proj @ self.epsilon[server_id]
        return h

    def _head_logits(self, h: np.ndarray) -> tuple[float, float, float]:
        """Compute raw logits for each head."""
        z1 = self.W1 @ h + self.b1
        z2 = self.W2 @ h + self.b2
        z3 = self.W3 @ h + self.b3
        return z1, z2, z3

    def _head_probs(self, h: np.ndarray) -> tuple[float, float, float]:
        """Compute P(mount), P(relevant), P(engaged)."""
        z1, z2, z3 = self._head_logits(h)
        return _sigmoid(z1), _sigmoid(z2), _sigmoid(z3)

    def _combined_score(self, agent: str, server_id: str, task_emb=None) -> float:
        """Combined ranking score from all 3 heads."""
        h = self._compute_h(agent, server_id, task_emb)
        p_mount, p_relevant, p_engaged = self._head_probs(h)
        return self.w_mount * p_mount + self.w_relevant * p_relevant + self.w_engaged * p_engaged

    @property
    def method_name(self) -> str:
        return f"latent_factor_mtl_n{self.n_obs}"

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int,
                  task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        alpha = min(self.n_obs / 200, 0.6)

        scored = []
        for c in candidates:
            server_id = c["id"]
            semantic = c.get("similarity", 0.0)

            if server_id in self.beta:
                collab_raw = self._combined_score(agent, server_id, task_emb)
                # Map from [0,1] probability range to [-1,1] for blending
                collab = max(-1.0, min(1.0, 2.0 * collab_raw - 1.0))
                final = semantic + alpha * collab
            else:
                final = semantic

            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:k]]

        # Epsilon-greedy exploration
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

        labels = self._extract_labels(rollout)
        if not labels:
            return

        self._ensure_agent(agent)

        for server_id, head_labels in labels.items():
            self._ensure_server(server_id)

            h = self._compute_h(agent, server_id, task_emb)
            z1, z2, z3 = self._head_logits(h)
            p1, p2, p3 = _sigmoid(z1), _sigmoid(z2), _sigmoid(z3)

            # Accumulate gradient on h from all applicable heads
            dL_dh = np.zeros(_H_DIM)

            if "mount" in head_labels:
                y = head_labels["mount"]
                g = _bce_grad(p1, y) * self.lambda_mount
                dL_dh += g * self.W1
                # Update head params
                self.W1 -= self.lr * (g * h + self.reg * self.W1)
                self.b1 -= self.lr * g

            if "relevant" in head_labels:
                y = head_labels["relevant"]
                g = _bce_grad(p2, y) * self.lambda_relevant
                dL_dh += g * self.W2
                self.W2 -= self.lr * (g * h + self.reg * self.W2)
                self.b2 -= self.lr * g

            if "engaged" in head_labels:
                y = head_labels["engaged"]
                g = _bce_grad(p3, y) * self.lambda_engaged
                dL_dh += g * self.W3
                self.W3 -= self.lr * (g * h + self.reg * self.W3)
                self.b3 -= self.lr * g

            # Backprop through h to shared params
            # h = [β_s, γ_a · γ_s, task_proj · ε_s]
            # dL/dβ_s = dL/dh[0]
            dL_db = dL_dh[0]
            self.beta[server_id] -= self.lr * (dL_db + self.reg * self.beta[server_id])

            # dL/d(γ_a · γ_s) = dL/dh[1]
            dL_dga_gs = dL_dh[1]
            ga = self.gamma_a[agent].copy()
            gs = self.gamma_s[server_id].copy()
            self.gamma_a[agent] -= self.lr * (dL_dga_gs * gs + self.reg * ga)
            self.gamma_s[server_id] -= self.lr * (dL_dga_gs * ga + self.reg * gs)

            # dL/d(task_proj · ε_s) = dL/dh[2]
            if task_emb is not None:
                dL_dtp_es = dL_dh[2]
                task_proj = self.P @ task_emb
                es = self.epsilon[server_id].copy()
                self.epsilon[server_id] -= self.lr * (dL_dtp_es * task_proj + self.reg * es)
                self.P -= self.lr * (dL_dtp_es * np.outer(es, task_emb) + self.reg_P * self.P)

        self.n_obs += 1

    def _extract_labels(self, rollout: dict) -> dict[str, dict[str, float]]:
        """Extract per-server, per-head binary labels from a rollout.

        Returns: {server_id: {"mount": 0/1, "relevant": 0/1, "engaged": 0/1}}
        Only includes heads that are applicable for each server.
        """

        def _extract_sids(key):
            sids = set()
            for tool_str in rollout.get(key, []):
                sids.add(tool_str.split(":")[0] if ":" in tool_str else tool_str)
            return sids

        tools_selected = _extract_sids("tools_selected")
        tools_errored = _extract_sids("tools_errored")
        tools_abandoned = _extract_sids("tools_abandoned")

        feedback = rollout.get("feedback", {})
        mounted = set(rollout.get("inventory_mounted", []))
        failed = set(rollout.get("inventory_failed", []))

        # Parse per-tool ratings
        RATING_MAP = {"liked": 1, "neutral": 0, "disliked": -1}
        server_ratings: dict[str, list[int]] = {}
        for tool_str, fb in feedback.items():
            if tool_str == "tools_relevant":
                continue
            sid = tool_str.split(":")[0] if ":" in tool_str else tool_str
            if isinstance(fb, dict):
                rating = fb.get("rating", "")
                if rating in RATING_MAP:
                    server_ratings.setdefault(sid, []).append(RATING_MAP[rating])

        tools_relevant = None
        if isinstance(feedback, dict) and "tools_relevant" in feedback:
            tools_relevant = feedback["tools_relevant"]

        labels: dict[str, dict[str, float]] = {}

        # --- Mount failed servers ---
        for sid in failed:
            labels[sid] = {"mount": 0.0}
            # No relevance or engagement labels (can't assess what didn't mount)

        # --- Mounted servers ---
        for sid in mounted:
            head = {"mount": 1.0}

            # Head 2: relevance
            if tools_relevant is not None:
                # Relevance check format
                if not tools_relevant:
                    head["relevant"] = 0.0  # agent said tools irrelevant
                # If tools_relevant=True but not used, no relevance label
            elif sid in tools_selected:
                # Was selected — relevant
                head["relevant"] = 1.0
            elif not tools_selected:
                # Nothing was selected — weak irrelevant
                head["relevant"] = 0.0

            # Head 3: engagement (only if mounted AND selected)
            if sid in tools_selected:
                if sid in server_ratings:
                    ratings = server_ratings[sid]
                    # Any liked → engaged=1, any disliked → engaged=0
                    if any(r == 1 for r in ratings):
                        head["engaged"] = 1.0
                    elif any(r == -1 for r in ratings):
                        head["engaged"] = 0.0
                    else:
                        # All neutral — mild positive
                        head["engaged"] = 1.0
                elif sid in tools_errored:
                    head["engaged"] = 0.0
                elif sid in tools_abandoned:
                    head["engaged"] = 0.0
                else:
                    # Selected but no rating — mild positive
                    head["engaged"] = 1.0

            labels[sid] = head

        return labels

    def save(self, path: str):
        data = {
            "model_type": "latent_factor_mtl",
            "latent_dim": self.latent_dim,
            "n_obs": self.n_obs,
            "P": self.P.tolist(),
            "beta": self.beta,
            "gamma_s": {sid: v.tolist() for sid, v in self.gamma_s.items()},
            "gamma_a": {agent: v.tolist() for agent, v in self.gamma_a.items()},
            "epsilon": {sid: v.tolist() for sid, v in self.epsilon.items()},
            "W1": self.W1.tolist(), "b1": self.b1,
            "W2": self.W2.tolist(), "b2": self.b2,
            "W3": self.W3.tolist(), "b3": self.b3,
            "w_mount": self.w_mount, "w_relevant": self.w_relevant, "w_engaged": self.w_engaged,
            "lambda_mount": self.lambda_mount, "lambda_relevant": self.lambda_relevant,
            "lambda_engaged": self.lambda_engaged,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        log.info(f"MTL model saved: {self.n_obs} obs, {len(self.gamma_a)} agents, "
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

        self.W1 = np.array(data["W1"])
        self.b1 = data["b1"]
        self.W2 = np.array(data["W2"])
        self.b2 = data["b2"]
        self.W3 = np.array(data["W3"])
        self.b3 = data["b3"]

        self.w_mount = data.get("w_mount", 0.4)
        self.w_relevant = data.get("w_relevant", 0.3)
        self.w_engaged = data.get("w_engaged", 0.3)
        self.lambda_mount = data.get("lambda_mount", 1.0)
        self.lambda_relevant = data.get("lambda_relevant", 1.0)
        self.lambda_engaged = data.get("lambda_engaged", 1.0)

        log.info(f"MTL model loaded: {self.n_obs} obs, {len(self.gamma_a)} agents, "
                 f"{len(self.beta)} servers from {path}")

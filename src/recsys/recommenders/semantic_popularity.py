"""Semantic + Popularity hybrid recommender.

Combines cosine similarity (task relevance) with Smithery use_count / GitHub stars
(popularity signal) using a weighted score. This is the "obvious practitioner baseline":
filter to relevant servers via semantic search, then prefer well-known ones.

Score = α * semantic_sim + (1-α) * normalized_popularity
α = 0.7  (semantic dominates but popularity has real weight)

Popularity normalization: log1p(use_count) / log1p(MAX_USE_COUNT), capped at 1.
"""

import math

from .base import BaseRecommender

# Approximate max use_count seen in the pool (exa ≈ 1.6M)
_MAX_USE_COUNT = 2_000_000
_MAX_STARS = 50_000
_ALPHA = 0.7  # weight on semantic similarity


def _pop_score(c: dict) -> float:
    use = c.get("use_count", 0) or 0
    stars = c.get("stars", 0) or 0
    # Log-normalize each signal to [0, 1]
    norm_use = math.log1p(use) / math.log1p(_MAX_USE_COUNT)
    norm_stars = math.log1p(stars) / math.log1p(_MAX_STARS)
    # Average the two signals (so neither alone dominates)
    return (norm_use + norm_stars) / 2


class SemanticPopularityRecommender(BaseRecommender):
    @property
    def method_name(self) -> str:
        return "semantic_popularity"

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int, task_category: str = "", task_emb=None) -> list[dict]:
        scored = []
        for c in candidates:
            sem = c.get("similarity", 0.0)
            pop = _pop_score(c)
            final = _ALPHA * sem + (1 - _ALPHA) * pop
            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

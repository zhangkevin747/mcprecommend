"""Random baseline recommender.

Selects K random servers from semantic retrieval candidates. Seeded for reproducibility.
"""

import random

from .base import BaseRecommender


class RandomRecommender(BaseRecommender):
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int, task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        pool = list(candidates)
        self._rng.shuffle(pool)
        return pool[:k]

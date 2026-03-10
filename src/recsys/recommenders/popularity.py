"""Popularity baseline recommender.

Ranks candidates by a popularity score: use_count + stars * 100 + tool_count.
"""

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int, task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        def score(c):
            use_count = c.get("use_count", 0) or 0
            stars = c.get("stars", 0) or 0
            tool_count = len(c.get("tools", []))
            return use_count + stars * 100 + tool_count

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[:k]

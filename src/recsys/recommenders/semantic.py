"""Semantic search recommender (cold start baseline).

Ranks candidates by cosine similarity, breaking ties with popularity (use_count/stars).
"""

from .base import BaseRecommender


class SemanticRecommender(BaseRecommender):
    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int, task_category: str = "", task_emb=None) -> list[dict]:
        # Already sorted by similarity from retriever, but re-sort with popularity tiebreak
        def score(c):
            sim = c.get("similarity", 0)
            pop = c.get("use_count", 0) or 0
            stars = c.get("stars") or 0
            # Normalize popularity to [0, 0.01] range so it only breaks ties
            pop_bonus = min(pop / 1_000_000, 0.01) + min(stars / 100_000, 0.01)
            return sim + pop_bonus

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[:k]

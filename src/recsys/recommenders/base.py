"""Base recommender interface."""

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, agent: str, task_query: str, candidates: list[dict], k: int, task_category: str = "", task_emb=None, epsilon: float = 0.0) -> list[dict]:
        """Select top K candidates from the candidate list.

        Args:
            agent: Agent identifier (e.g., "haiku-4.5")
            task_query: The task query string
            candidates: List of candidate server dicts from retriever
            k: Number of servers to recommend
            task_category: Task category (e.g., "search", "finance")
            task_emb: Optional pre-computed task embedding (np.ndarray, 1536-dim)

        Returns:
            Top K candidates, ranked.
        """
        ...

    def update(self, rollout: dict, task_emb=None) -> None:
        """Online learning hook. Called after each rollout. Default no-op."""
        pass

    @property
    def method_name(self) -> str:
        """Name for the reranking method, used in rollout logs."""
        return self.__class__.__name__

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel


class BaseRanker(ABC):
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model

    def get_top_n_keywords(
        self,
        words: np.ndarray[str],
        top_n: int,
    ) -> List[str]:
        embeddings = self.embedding_model(words)

        if len(embeddings) == 0:
            return []

        top_n_indices = self.get_ranked_embedding(embeddings)
        top_n_keywords: List[str] = words[top_n_indices][:top_n].tolist()

        return top_n_keywords

    @abstractmethod
    def get_ranked_embedding(self, embeddings: np.ndarray[np.ndarray[float]]) -> np.ndarray[int]:
        """Ranking the embeddings."""

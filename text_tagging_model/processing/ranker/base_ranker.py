from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel


class BaseRanker(ABC):
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model

    def get_top_n_keywords(
        self,
        text: Union[str, List[str]],
        words: np.ndarray[str],
        top_n: int,
    ) -> List[str]:
        embeddings = self.embedding_model(words)
        if len(embeddings) == 0:
            return []

        text_embedding = self.get_text_embedding(text)
        top_n_indices = self.get_ranked_embedding(text_embedding, embeddings)
        top_n_keywords: List[str] = words[top_n_indices][:top_n].tolist()

        return top_n_keywords

    def get_text_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.embedding_model.get_sentence_emb(" ".join(text))

    @abstractmethod
    def get_ranked_embedding(
        self,
        text_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray[int]:
        """Ranking the embeddings. The method to be implemented by derived classes."""

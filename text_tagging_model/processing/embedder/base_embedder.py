from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingModel(ABC):
    def __call__(self, words: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        return self.get_embeddings(words)

    @abstractmethod
    def get_embeddings(self, words: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        """Returns embedding for each word in words."""

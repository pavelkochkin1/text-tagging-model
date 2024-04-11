from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseEmbeddingModel(ABC):
    def __call__(self, words: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        return self.get_embeddings(words)

    @abstractmethod
    def get_embeddings(self, words: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        """Returns embedding for each word in words."""

    @abstractmethod
    def get_sentence_emb(self, text: Union[List[str], str]) -> np.ndarray[float]:
        """Returns embedding for full text"""

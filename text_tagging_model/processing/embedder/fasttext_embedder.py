import os

import fasttext
import numpy as np

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel


class FastTextEmbedder(BaseEmbeddingModel):
    def __init__(self, model_path: str = "./cc.ru.300.bin") -> None:
        abs_path = os.path.abspath(model_path)
        self.ft_model = fasttext.load_model(abs_path)

    def get_embeddings(self, words: np.ndarray[str]):
        embeddings = np.array([np.array(self.ft_model[word]) for word in words])
        return embeddings

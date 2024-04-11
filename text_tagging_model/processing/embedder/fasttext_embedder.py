import os
from typing import List, Union

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

    def get_sentence_emb(self, text: Union[List[str], str]):
        emb = self.ft_model.get_sentence_vector(text)
        return emb

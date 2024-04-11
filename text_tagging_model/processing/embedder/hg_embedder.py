from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel


class HGEmbedder(BaseEmbeddingModel):
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.sent_model = SentenceTransformer(model_name)

    def get_embeddings(self, words: np.ndarray[str]):
        tokenized = self.tokenizer(
            words.tolist(),
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        embeddings = (
            self.model(**tokenized, output_hidden_states=True)
            .last_hidden_state[:, 0, :]
            .detach()
            .numpy()
        )

        return embeddings

    def get_sentence_emb(self, text: Union[List[str], str]) -> np.ndarray:
        return self.sent_model.encode(" ".join(text))

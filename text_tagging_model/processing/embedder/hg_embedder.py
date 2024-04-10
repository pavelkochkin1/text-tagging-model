import numpy as np
from transformers import AutoModel, AutoTokenizer

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel


class HGEmbedder(BaseEmbeddingModel):
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, words: np.ndarray[str]):
        tokenized = self.tokenizer(words, return_tensors="pt", truncation=True, padding=True)
        embeddings = self.model(**tokenized, output_hidden_states=True).last_hidden_state[:, 0, :]

        return embeddings

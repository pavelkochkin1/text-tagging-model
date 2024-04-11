import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from text_tagging_model.processing.embedder.base_embedder import BaseEmbeddingModel
from text_tagging_model.processing.ranker.base_ranker import BaseRanker


class MaxDistanceRanker(BaseRanker):
    def __init__(self, embedding_model: BaseEmbeddingModel, distance_metric: str = "cosine"):
        super().__init__(embedding_model)
        self.distance_metric = distance_metric

    def get_ranked_embedding(
        self,
        text_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray[int]:
        distances = pairwise_distances(embeddings, metric=self.distance_metric).mean(axis=1)
        top_n_indices = (-distances).argsort()
        return top_n_indices

    # def get_text_embedding(self, text: List[str]) -> np.ndarray:
    #     # Переопределение не требуется, если ранжирование не зависит от текста.
    #     return super().get_text_embedding(text)

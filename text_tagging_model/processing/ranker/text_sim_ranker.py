import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text_tagging_model.processing.ranker.base_ranker import BaseRanker


class TextSimRanker(BaseRanker):
    def get_ranked_embedding(
        self,
        text_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray[int]:
        similarity_scores = np.array(
            [cosine_similarity([text_embedding], [vector])[0][0] for vector in embeddings]
        )
        top_n_indices = (-similarity_scores).argsort()
        return top_n_indices

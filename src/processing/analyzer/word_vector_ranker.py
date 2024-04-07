import os.path

import fasttext
import fasttext.util
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from logger_config import logger
from src.processing.utils import languages


class WordVectorRanker:
    """
    The class is used to get the most distant words from others.

    Attributes
    ----------
    language : str
        language that will be used. available: 'russian', 'english'
    model_name : str
        Name of model from fasttext (will be download if not exists)
        or path to already downloaded .bin file with embeddings.

    Methods
    -------
    get_top_n_keywords(words, top_n, distance_metric)
        Returns n indexes of embeddings with max distance from others.
    """

    def __init__(self, language: str, model_name: str = "cc.ru.300.bin"):
        if not os.path.exists(model_name):
            logger.info(
                f"There is no model: {model_name}; starting downloading model from fasttext repo."
            )
            fasttext.util.download_model(language, if_exists="ignore")

        if language not in languages:
            raise ValueError(f"Wrong language! Available languages: {languages.keys()}")

        self.ft_model = fasttext.load_model(model_name)

    def get_top_n_keywords(
        self,
        words: np.ndarray,
        top_n: int,
        distance_metric: str = "cosine",
    ) -> np.ndarray:
        """Returns n indexes of embeddings with max distance from others.

        Args:
            words (np.array): array with embeddings of words
            top_n (int): number of words in output
            distance_metric (str, optional): distance metric,
            available ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
            Defaults to "cosine".

        Returns:
            np.ndarray: array with indexes
        """
        vectors = np.array([self.ft_model[word] for word in words])

        distances = pairwise_distances(vectors, metric=distance_metric).mean(axis=1)
        top_n_indices = (-distances).argsort()
        top_n_keywords = words[top_n_indices][:top_n]

        return top_n_keywords

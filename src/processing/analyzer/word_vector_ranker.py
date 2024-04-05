import os.path

import fasttext
import fasttext.util
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from constants import DISTANCE_METRIC, logger


class WordVectorRanker:

    def __init__(self, language="ru", model_name: str = "cc.ru.300.bin"):
        if not os.path.exists(model_name):
            fasttext.util.download_model(language, if_exists="ignore")
        self.ft_model = fasttext.load_model(model_name)

    def get_top_n_keywords(self, words: np.array, top_n: int):
        vectors = np.array([self.ft_model[word] for word in words])

        distances = pairwise_distances(vectors, metric=DISTANCE_METRIC).mean(axis=1)
        top_n_indices = (-distances).argsort()
        top_n_keywords = words[top_n_indices][:top_n]

        return top_n_keywords

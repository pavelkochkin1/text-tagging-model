from collections import Counter
from itertools import chain

import numpy as np

from text_tagging_model.models.base_extractor import BaseExtractor
from text_tagging_model.models.rake_based_model.keyphrases_extractor import RakeKeyphrasesExtractor
from text_tagging_model.processing.embedder.fasttext_embedder import FastTextEmbedder
from text_tagging_model.processing.normalizers import NounsKeeper, PunctDeleter, StopwordsDeleter
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.ranker.max_distance_ranker import MaxDistanceRanker


class TagsExtractor(BaseExtractor):
    def __init__(
        self,
        language: str = "russian",
        fasttext_model_path: str = "cc.ru.300.bin",
        min_cnt_keyword: int = 2,
    ) -> None:
        self.extractor = RakeKeyphrasesExtractor(language=language)
        self.normalizer = NormalizersPipe(
            [
                PunctDeleter(),
                StopwordsDeleter(language),
                NounsKeeper(language),
            ],
            final_split=True,
        )

        embedder = FastTextEmbedder(fasttext_model_path)
        self.ranker = MaxDistanceRanker(embedder)
        self.min_cnt_keyword = min_cnt_keyword

    def extract(
        self,
        text: str,
        top_n: int,
    ) -> np.ndarray:
        """Returns extracted keywords from the text

        Args:
            text (str): text to extract
            top_n (int): number of words to extract
            min_keyword_cnt (int): min number of words in the extracted phrases
            distance_metric (str, optional): distance metric,
            available ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
            Defaults to "cosine".

        Returns:
            np.ndarray: array with extracted keywords
        """

        keyphrases_with_scores = self.extractor.extract(text.lower())
        keyphrases = [text for _, text in keyphrases_with_scores]

        normalized_keyphrases = list(map(self.normalizer.normalize, keyphrases))
        normalized_words = list(chain(*normalized_keyphrases))

        most_co_occurring_words = np.array(
            [
                word
                for word, cnt in Counter(normalized_words).most_common(top_n)
                if cnt >= self.min_cnt_keyword
            ]
        )

        # keywords = most_co_occurring_words.tolist()
        keywords = self.ranker.get_top_n_keywords(normalized_words, most_co_occurring_words, top_n)

        return keywords

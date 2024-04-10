from collections import Counter
from itertools import chain
from typing import List

import numpy as np
from tqdm import tqdm

from text_tagging_model.models.rake_based_model.keyphrases_extractor import RakeKeyphrasesExtractor
from text_tagging_model.processing.embedder.fasttext_embedder import FastTextEmbedder
from text_tagging_model.processing.normalizers import NounsKeeper, PunctDeleter, StopwordsDeleter
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.ranker.max_distance_ranker import MaxDistanceRanker


class TagsExtractor:
    """
    The class is used to extract keywords from the text.

    Attributes
    ----------
    language : str
        language that will be used. available: 'russian', 'english'
    model_name : str
        Name of model from fasttext (will be download if not exists)
        or path to already downloaded .bin file with embeddings.

    Methods
    -------
    extract(text, top_n, min_keyword_cnt, distance_metric)
        Returns list with extracted keywords
    """

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

    def extract_for_corpus(
        self,
        texts: List[str],
        top_n: int,
    ) -> np.ndarray:
        """Returns extracted keywords for corpus of texts

        Args:
            texts (List[str]): list with texts
            top_n (int): number of words to extract
            min_keyword_cnt (int): min number of words in the extracted phrases
            distance_metric (str, optional): distance metric,
            available ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
            Defaults to "cosine".

        Returns:
            np.ndarray: array with arrays extracted keywords
        """
        extracted_keywords = list()

        for text in tqdm(texts):
            keywords = self.extract(text, top_n)
            extracted_keywords.append(keywords)

        return extracted_keywords

    def extract(
        self,
        text: str,
        top_n: int,
        # min_keyword_cnt: int,
        # distance_metric: str = "cosine",
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

        keywords = self.ranker.get_top_n_keywords(most_co_occurring_words, top_n)

        return keywords

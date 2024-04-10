from collections import Counter
from typing import List

import numpy as np
from tqdm import tqdm

from text_tagging_model.processing.embedder.hg_embedder import HGEmbedder
from text_tagging_model.processing.normalizers import NounsKeeper, PunctDeleter, StopwordsDeleter
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.ranker.max_distance_ranker import MaxDistanceRanker
from text_tagging_model.processing.summarizator.bart_summarization import MBartSummarizator


class TagSumExtractor:
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
        summarizator_model: str = "IlyaGusev/mbart_ru_sum_gazeta",
        embedder_model: str = "cointegrated/rubert-tiny2",
        language: str = "russian",
        min_cnt_keyword: int = 2,
        device: str = "cpu",
    ) -> None:
        self.summarizator = MBartSummarizator(summarizator_model, device=device)
        self.normalizer = NormalizersPipe(
            [
                PunctDeleter(),
                StopwordsDeleter(language),
                NounsKeeper(language),
            ],
            final_split=True,
        )

        embedder = HGEmbedder(embedder_model)
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

        keyphrase = self.summarizator.get_summary(text.lower())
        normalized_keyphrase = list(map(self.normalizer.normalize, keyphrase))
        most_co_occurring_words = np.array(
            [
                word
                for word, cnt in Counter(normalized_keyphrase).most_common(top_n)
                if cnt >= self.min_cnt_keyword
            ]
        )

        keywords = self.ranker.get_top_n_keywords(most_co_occurring_words, top_n)

        return keywords

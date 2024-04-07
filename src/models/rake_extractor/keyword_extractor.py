import numpy as np

from logger_config import logger
from models.rake_extractor.keyphrases_extractor import RakeKeyphrasesExtractor
from processing.analyzer.word_vector_ranker import WordVectorRanker
from processing.normalizers.text_normalizer import TextNormalizer


class KeywordExtractor:
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

    def __init__(self, language: str = "russian", model_name: str = "cc.ru.300.bin"):
        self.language = language
        self.extractor = RakeKeyphrasesExtractor(language=language)
        self.normalizer = TextNormalizer(language=language)
        self.ranker = WordVectorRanker(language=language, model_name=model_name)

    def extract(
        self,
        text: str,
        top_n: int,
        min_keyword_cnt: int,
        distance_metric: str = "cosine",
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

        logger.info("Keywords extraction...")
        keyphrases_with_scores = self.extractor.extract(text)
        keyphrases = [text for _, text in keyphrases_with_scores]

        logger.info("Keywords normalization and filtering by count...")
        normalized_words = self.normalizer.normalize(keyphrases)

        most_co_occurring_words = np.array(
            [word for word, cnt in normalized_words.most_common(top_n) if cnt >= min_keyword_cnt]
        )

        logger.info("Keywords analysing to get tags...")
        keywords = self.ranker.get_top_n_keywords(most_co_occurring_words, top_n, distance_metric)

        return keywords

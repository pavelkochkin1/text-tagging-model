import numpy as np

from constants import logger
from models.rake_extractor.keyphrases_extractor import RakeKeyphrasesExtractor
from processing.analyzer.word_vector_ranker import WordVectorRanker
from processing.normalizers.text_normalizer import TextNormalizer


class KeywordExtractor:
    def __init__(self, language: str = "russian", model_name: str = "cc.ru.300.bin"):
        self.language = language
        self.extractor = RakeKeyphrasesExtractor(language=language)
        self.normalizer = TextNormalizer(language=language)
        self.ranker = WordVectorRanker(language=language, model_name=model_name)

    def extract(self, text: str, top_n: int, min_keyword_cnt: int) -> np.array:
        logger.info("Keywords extraction...")
        keyphrases = self.extractor.extract(text)

        logger.info("Keywords normalization and filtering by count...")
        normalized_words = self.normalizer.normalize(keyphrases)

        most_co_occurring_words = np.array(
            [word for word, cnt in normalized_words.most_common(top_n) if cnt >= min_keyword_cnt]
        )

        logger.info("Keywords analysing to get tags...")
        keywords = self.ranker.get_top_n_keywords(most_co_occurring_words, top_n)

        return keywords

import numpy as np

from constants import logger
from models.rake_extractor.keyphrases_extractor import RakeKeyphrasesExtractor
from processing.analyzer.word_vector_ranker import WordVectorRanker
from processing.normalizers.text_normalizer import TextNormalizer


class KeywordExtractor:
    def __init__(self, language):
        self.language = language
        self.extractor = RakeKeyphrasesExtractor(language=language)
        self.normalizer = TextNormalizer(language="ru")
        self.ranker = WordVectorRanker(
            language="ru",
            model_name="/Users/sfnurkaev/Documents/text-tagging-model/resources/models/cc.ru.300.bin",
            # TODO: В докере будем прокидывать в системные переменные
        )

    def extract(self, text: str, top_n: int, min_keyword_cnt: int):
        logger.info("Экстрактим кейворды...")
        keyphrases = self.extractor.extract(text)

        logger.info("Нормализуем и фильтруем кейворды...")
        normalized_words = self.normalizer.normalize(keyphrases)

        most_co_occurring_words = np.array([
            word
            for word, cnt in normalized_words.most_common(top_n)
            if cnt >= min_keyword_cnt
        ])

        logger.info("Анализируем кейворды для хэштэга...")
        keywords = self.ranker.get_top_n_keywords(most_co_occurring_words, top_n)

        return keywords

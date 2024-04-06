from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake


class RakeKeyphrasesExtractor:
    def __init__(self, language: str = "russian", max_length: int = 100000):
        nltk.download("stopwords")
        nltk.download("punkt")

        self.language = language
        self.max_length = max_length
        self.stopwords = stopwords.words(self.language)
        self.rake = Rake(
            stopwords=self.stopwords,
            language=self.language,
            max_length=self.max_length,
        )

    def extract(self, text: str) -> List[Tuple[float, str]]:
        self.rake.extract_keywords_from_text(text)
        keyphrases = self.rake.get_ranked_phrases_with_scores()
        return keyphrases

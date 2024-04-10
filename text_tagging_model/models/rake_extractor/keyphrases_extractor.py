from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake

from text_tagging_model.processing.utils import languages


class RakeKeyphrasesExtractor:
    """
    The class is used to extract keyphrases from the text.

    Attributes
    ----------
    language : str
        language that will be used. available: 'russian', 'english'
    max_length : str
        max length of extracted phrases

    Methods
    -------
    extract(text)
        Returns list with extracted phrases
    """

    def __init__(self, language: str = "russian", max_length: int = 100000):
        if language not in languages:
            raise ValueError(f"Wrong language!\nAvailable languages: {languages.keys()}")

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
        """Returns extracted phrases from the text

        Args:
            text (str): text to extract

        Returns:
            List[Tuple[float, str]]: list with tuples of score and phrase
        """
        self.rake.extract_keywords_from_text(text)
        keyphrases: List[Tuple[float, str]] = self.rake.get_ranked_phrases_with_scores()
        return keyphrases

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake

from constants import KEYPHRASES_MAX_LENGTH, LANGUAGE

nltk.download("stopwords")
nltk.download("punkt")


class RakeKeyphrasesExtractor:
    def __init__(self, language=LANGUAGE, max_length=KEYPHRASES_MAX_LENGTH):
        self.language = language
        self.max_length = max_length
        self.stopwords = stopwords.words(self.language)
        self.rake = Rake(
            stopwords=self.stopwords, language=self.language, max_length=self.max_length
        )

    def extract(self, text):
        self.rake.extract_keywords_from_text(text)
        keyphrases = self.rake.get_ranked_phrases_with_scores()
        return keyphrases

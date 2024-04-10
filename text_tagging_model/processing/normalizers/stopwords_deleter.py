import nltk
from nltk.corpus import stopwords

from text_tagging_model.processing.normalizers.base_normalizer import BaseNormalizer
from text_tagging_model.processing.utils import languages


class StopwordsDeleter(BaseNormalizer):
    def __init__(self, language: str) -> None:
        if language not in languages:
            raise ValueError(f"Wrong language!\nAvailable languages: {languages.keys()}")

        nltk.download("stopwords")
        self.stop_words = set(stopwords.words(language))

    def normalize(self, text: str) -> str:
        clear_text = []
        for word in text.split():
            if word.lower() not in self.stop_words and word != "":
                clear_text.append(word)

        return " ".join(clear_text)

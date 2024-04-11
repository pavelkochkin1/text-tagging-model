import string

from text_tagging_model.processing.normalizers.base_normalizer import BaseNormalizer


class PunctDeleter(BaseNormalizer):
    def __init__(self) -> None:
        self.punct = string.punctuation
        self.punct = self.punct.replace("-", "")

    def normalize(self, text: str) -> str:
        clear_text = text
        for punctuation in self.punct:
            clear_text = clear_text.replace(punctuation, "")

        return clear_text

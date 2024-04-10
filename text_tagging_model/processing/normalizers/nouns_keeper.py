import pymorphy2

from text_tagging_model.processing.normalizers.base_normalizer import BaseNormalizer
from text_tagging_model.processing.utils import languages


class NounsKeeper(BaseNormalizer):
    def __init__(self, language: str, keep_latn: bool = False) -> None:
        if language not in languages:
            raise ValueError(f"Wrong language!\nAvailable languages: {languages.keys()}")

        self.morph = pymorphy2.MorphAnalyzer(lang=languages[language])
        self.keep_latn = keep_latn

    def normalize(self, text: str) -> str:
        nouns = []
        for word in text.split():
            # TODO: подумать над тем, как приводить к существительному ещё и глаголы
            p = self.morph.parse(str(word))[0]
            if p.tag.POS == "NOUN":
                nouns.append(p.normal_form)

            if self.keep_latn and "LATN" in p.tag:
                nouns.append(p.normal_form)

        return " ".join(nouns)

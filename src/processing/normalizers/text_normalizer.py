from collections import Counter
from typing import List, Tuple, Type

import pymorphy2

from processing.utils import languages


class TextNormalizer:
    def __init__(self, language: str):
        if language not in languages:
            raise ValueError(f"Wrong language! Available languages: {languages.keys()}")

        self.morph = pymorphy2.MorphAnalyzer(lang=languages[language])

    def normalize(self, keyphrases: List[Tuple[float, str]]) -> Type[Counter]:
        filtered_tokens = []
        for _, text in keyphrases:
            for word in text.split():
                # TODO: подумать над тем, как приводить к существительному ещё и глаголы
                p = self.morph.parse(str(word))[0]
                if p.tag.POS == "NOUN":
                    filtered_tokens.append(p.normal_form)
        return Counter(filtered_tokens)

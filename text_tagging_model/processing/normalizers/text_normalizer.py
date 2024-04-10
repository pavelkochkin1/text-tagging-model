from collections import Counter
from typing import Any, List

import pymorphy2

from text_tagging_model.processing.utils import languages


class TextNormalizer:
    """
    The class used to get Counter with normalized nouns from texts.

    Attributes
    ----------
    language : str
        language that will be used. available: 'russian', 'english'

    Methods
    -------
    normalize(keyphrases)
        Returns Counter with normalized nouns from texts.
    """

    def __init__(self, language: str):
        if language not in languages:
            raise ValueError(f"Wrong language!\nAvailable languages: {languages.keys()}")

        self.morph = pymorphy2.MorphAnalyzer(lang=languages[language])

    def normalize(self, keyphrases: List[str]) -> Counter[Any]:
        """Returns Counter with normalized nouns from texts.

        Args:
            keyphrases (List[str]): list with texts

        Returns:
            Counter[Any]: Counter with normalized nouns from texts
        """
        filtered_tokens = []
        for text in keyphrases:
            for word in text.split():
                # TODO: подумать над тем, как приводить к существительному ещё и глаголы
                p = self.morph.parse(str(word))[0]
                if p.tag.POS == "NOUN":
                    filtered_tokens.append(p.normal_form)

        cntr = Counter(filtered_tokens)

        return cntr

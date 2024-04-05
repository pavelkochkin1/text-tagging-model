from collections import Counter

import pymorphy2


class TextNormalizer:
    def __init__(self, language="ru"):
        self.morph = pymorphy2.MorphAnalyzer(lang=language)

    def normalize(self, keyphrases):
        filtered_tokens = []
        for _, text in keyphrases:
            for word in text.split():
                # TODO: подумать над тем, как приводить к существительному ещё и глаголы
                p = self.morph.parse(str(word))[0]
                if p.tag.POS == "NOUN":
                    filtered_tokens.append(p.normal_form)
        return Counter(filtered_tokens)

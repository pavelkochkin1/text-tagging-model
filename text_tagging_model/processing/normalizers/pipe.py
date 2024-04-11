from typing import List, Union

from text_tagging_model.processing.normalizers.base_normalizer import BaseNormalizer


class NormalizersPipe:
    def __init__(self, normalizers: List[BaseNormalizer], final_split: bool = False) -> None:
        self.normalizers = normalizers
        self.final_split = final_split

    def normalize(self, text: str) -> Union[str, List[str]]:
        normalized_text = text
        for normalizer in self.normalizers:
            normalized_text = normalizer.normalize(normalized_text)

        if self.final_split:
            return normalized_text.split()

        return normalized_text

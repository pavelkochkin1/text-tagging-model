from abc import ABC, abstractmethod
from typing import List

import numpy as np
from tqdm import tqdm


class BaseExtractor(ABC):
    def extract_for_corpus(
        self,
        texts: List[str],
        top_n: int,
    ) -> np.ndarray:
        extracted_keywords = list()

        for text in tqdm(texts):
            keywords = self.extract(text, top_n)
            extracted_keywords.append(keywords)

        return extracted_keywords

    @abstractmethod
    def extract(self, text: str, top_n: int) -> np.ndarray:
        """Main method to extract tags from text."""

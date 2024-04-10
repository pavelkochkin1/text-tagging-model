from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Some text preprocessing"""

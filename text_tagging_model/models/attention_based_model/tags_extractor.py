from typing import List

from text_tagging_model.models.attention_based_model.attention_extractor import (
    MBartTokenAttentionLevelExtractor,
)
from text_tagging_model.models.base_extractor import BaseExtractor
from text_tagging_model.processing.normalizers.nouns_keeper import (
    BilingualTextNormalizer,
)
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.normalizers.punctuation_deleter import PunctDeleter
from text_tagging_model.processing.normalizers.stopwords_deleter import StopwordsDeleter


class AttentionBasedTagger(BaseExtractor):
    def __init__(self):
        self.normalizer_pipe = NormalizersPipe(
            [
                StopwordsDeleter("russian", drop_english=True, remove_non_alpha=True),
                PunctDeleter(),
            ]
        )
        self.attention_extractor = MBartTokenAttentionLevelExtractor()
        self.phrase_normalizer = BilingualTextNormalizer()

    def extract(self, text: str, top_n: int) -> List[str]:
        # Шаг 1: Нормализация текста
        clear_text = self.normalizer_pipe.normalize(text)

        # Шаг 2: Извлечение внимания токенов
        top_bigrams = self.attention_extractor.get_top_bigrams_by_token_attention(
            clear_text, top_k=30
        )

        # Шаг 3: Подготовка ключевых фраз
        key_phrases = [word for word, attention in top_bigrams]

        # Шаг 4: Нормализация ключевых фраз
        normalized_phrases = self.phrase_normalizer.normalize(key_phrases)
        top_keywords = normalized_phrases[:top_n]

        return top_keywords

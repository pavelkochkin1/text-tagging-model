import os

import pytest

from src.models.rake_extractor.keyword_extractor import KeywordExtractor


@pytest.fixture
def keyword_extractor_model() -> KeywordExtractor:
    model = KeywordExtractor(
        language="russian",
        model_name=os.path.abspath("resources/models/cc.ru.300.bin"),
    )

    return model


def test_rake_extractor_case(keyword_extractor_model: pytest.fixture) -> None:
    source_text = """
        Методы ускорения инференса.
        Дистилляция
        Обычно в этом методе есть какая-то большая модель, которую мы называем учитель (teacher),
        и модель поменьше — студент (student). Хорошим примером будет YandexGPT 3 — большая LLM,
        способная решать задачу с наилучшим качеством, но она совершенно не укладывается
        в наш вычислительный бюджет. Есть модель поменьше, вроде Т5, которая потребляет
        сильно меньше ресурсов, но не решает задачу так же качественно, как YandexGPT 3.
        Задача Knowledge Distillation состоит в том,
        чтобы минимизировать потери (loss) между фичами — предсказаниями учителя и студента.
    """
    tags = keyword_extractor_model.extract(source_text, 5, 2)
    assert tags.tolist() == ["модель", "задача", "метод", "студент", "учитель"]


def test_rake_extractor_errors() -> None:

    with pytest.raises(ValueError):
        KeywordExtractor(
            language="err",
            model_name="text-tagging-model/resources/models/cc.ru.300.bin",
        )

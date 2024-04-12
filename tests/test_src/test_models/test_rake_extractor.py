import pytest

from text_tagging_model.models.rake_based_model import TagsExtractor


@pytest.fixture
def tags_extractor() -> TagsExtractor:
    model = TagsExtractor(
        language="russian",
        fasttext_model_path="./resources/models/cc.ru.300.bin",
    )

    return model


def test_rake_extractor_case(tags_extractor: TagsExtractor) -> None:
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
    tags = tags_extractor.extract(source_text, 5)
    assert tags == ["модель", "задача", "метод", "студент", "учитель"]


def test_rake_extractor_errors() -> None:
    # lang error
    with pytest.raises(ValueError):
        TagsExtractor(
            language="err",
            fasttext_model_path="./resources/models/cc.ru.300.bin",
        )
    # model path error
    with pytest.raises(ValueError):
        TagsExtractor(
            language="err",
            fasttext_model_path="./resources/models/cc.ru.300.bin",
        )

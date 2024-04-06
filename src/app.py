from constants import LANGUAGE, MAX_TOP_KEYWORDS, MIN_KEYWORD_COUNT
from models.rake_extractor.keyword_extractor import KeywordExtractor


class TextTaggingApplication:
    def __init__(self, language="russian"):
        self.language = language
        self.extractor = KeywordExtractor(self.language)

    def get_keywords(self, text, top_n):
        return self.extractor.extract(text, top_n, MIN_KEYWORD_COUNT)


def run():
    app = TextTaggingApplication(language=LANGUAGE)

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

    hashtags = app.get_keywords(source_text, top_n=MAX_TOP_KEYWORDS)
    print(hashtags)


if __name__ == "__main__":
    run()

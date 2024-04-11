import numpy as np

from text_tagging_model.models.base_extractor import BaseExtractor
from text_tagging_model.processing.embedder.hg_embedder import HGEmbedder
from text_tagging_model.processing.normalizers import NounsKeeper, PunctDeleter, StopwordsDeleter
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.ranker.text_sim_ranker import TextSimRanker
from text_tagging_model.processing.summarizator.bart_summarization import MBartSummarizator


class TagSumExtractor(BaseExtractor):
    """
    The class is used to extract keywords from the text.

    Attributes
    ----------
    language : str
        language that will be used. available: 'russian', 'english'
    model_name : str
        Name of model from fasttext (will be download if not exists)
        or path to already downloaded .bin file with embeddings.

    Methods
    -------
    extract(text, top_n, min_keyword_cnt, distance_metric)
        Returns list with extracted keywords
    """

    def __init__(
        self,
        summarizator_model: str = "IlyaGusev/mbart_ru_sum_gazeta",
        embedder_model: str = "cointegrated/rubert-tiny2",
        language: str = "russian",
        device: str = "cpu",
    ) -> None:
        self.summarizator = MBartSummarizator(summarizator_model, device=device)
        self.normalizer = NormalizersPipe(
            [
                PunctDeleter(),
                StopwordsDeleter(language),
                NounsKeeper(language, keep_latn=True),
            ],
            final_split=True,
        )

        embedder = HGEmbedder(embedder_model)
        self.ranker = TextSimRanker(embedder)

    def extract(
        self,
        text: str,
        top_n: int,
    ) -> np.ndarray:
        """Returns extracted keywords from the text

        Args:
            text (str): text to extract
            top_n (int): number of words to extract
            min_keyword_cnt (int): min number of words in the extracted phrases
            distance_metric (str, optional): distance metric,
            available ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
            Defaults to "cosine".

        Returns:
            np.ndarray: array with extracted keywords
        """

        keyphrase = self.summarizator.get_summary(text.lower())
        normalized_keyphrase = self.normalizer.normalize(keyphrase)
        keywords = self.ranker.get_top_n_keywords(
            text=keyphrase,
            words=np.unique(normalized_keyphrase),
            top_n=top_n,
        )

        return keywords

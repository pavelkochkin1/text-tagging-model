import numpy as np

from text_tagging_model.models.base_extractor import BaseExtractor
from text_tagging_model.processing.embedder.hg_embedder import HGEmbedder
from text_tagging_model.processing.normalizers import NounsKeeper, PunctDeleter, StopwordsDeleter
from text_tagging_model.processing.normalizers.pipe import NormalizersPipe
from text_tagging_model.processing.ranker.text_sim_ranker import TextSimRanker
from text_tagging_model.processing.summarizator.bart_summarization import MBartSummarizator


class TagSumExtractor(BaseExtractor):
    """
    A class for extracting tags based on a summarized version of the input text,
    using an embedding-based similarity ranking approach.

    Args:
        summarizator_model (str): Model identifier for the summarization component.
        embedder_model (str): Model identifier for the embedding component used by the ranker.
        language (str): Language used for text processing. Defaults to 'russian'.
        device (str): The device (e.g., 'cpu' or 'cuda') to use for computation. Defaults to 'cpu'.

    Attributes:
        summarizator (MBartSummarizator):
            A model used for summarizing input text to focus on main concepts.
        normalizer (NormalizersPipe):
            A pipeline of text normalizers
            including punctuation and stopwords deletion, as well as noun retention.
        ranker (TextSimRanker):
            Ranks words based on their semantic similarity to the summarized text using embeddings.

    Methods:
        extract(text: str, top_n: int) -> np.ndarray:
            Extracts and returns the top_n most relevant tags from the provided text.
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
        """
        Extracts the top_n most relevant tags from the summarized version of the provided text.

        Args:
            text (str): The text from which to extract tags.
            top_n (int): The number of tags to extract.

        Returns:
            np.ndarray: An array of the top_n extracted tags
                based on their relevance to the summarized content.
        """
        keyphrase = self.summarizator.get_summary(text.lower())
        normalized_keyphrase = self.normalizer.normalize(keyphrase)
        keywords = self.ranker.get_top_n_keywords(
            text=keyphrase,
            words=np.unique(normalized_keyphrase),
            top_n=top_n,
        )

        return keywords

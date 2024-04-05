from typing import Dict, List

import numpy as np
from rouge import Rouge


def rogue_score_corpus(true: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    """Calculate ROGUE-1 (precision, recall and f1-score)

    `Recall = num_words_matches / num_words_in_reference`

    `Precision = num_words_matches / num_words_in_summary`

    `F1 = classic f1 score`

    Args:
        true (List[List[str]]): True list with lists of tags for texts
        pred (List[List[str]]): Predicted tags for texts

    Returns:
        Dict[str, float]: {"recall": num, "precision": num, "f1": num}
    """

    rouge = Rouge()
    rec, prec, f1 = list(), list(), list()

    for true_tags, pred_tags in zip(true, pred):
        true_tags_str = " ".join(true_tags)
        pred_tags_str = " ".join(pred_tags)

        scores = rouge.get_scores(true_tags_str, pred_tags_str)
        rogue_1 = scores[0]["rouge-1"]

        rec.append(rogue_1["r"])
        prec.append(rogue_1["p"])
        f1.append(rogue_1["f"])

    return {
        "recall": np.mean(rec),
        "precision": np.mean(prec),
        "f1": np.mean(f1),
    }

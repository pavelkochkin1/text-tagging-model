from src.models.metrics import rogue_score_corpus


def test_rogue_score_corpus() -> None:
    true = [
        ["a", "b", "c"],
        ["n", "m"],
    ]
    pred = [
        ["a", "b", "c"],
        ["n", "k"],
    ]

    scores = rogue_score_corpus(true, pred)

    assert round(scores["recall"], 2) == 0.75
    assert round(scores["precision"], 2) == 0.75
    assert round(scores["f1"], 2) == 0.75

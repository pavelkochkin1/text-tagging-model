from rouge import Rouge


def score_corpus(true, pred):
    rouge = Rouge()
    scores = rouge.get_scores(true, pred)

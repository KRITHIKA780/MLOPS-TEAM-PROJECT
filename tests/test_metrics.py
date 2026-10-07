import numpy as np

from src.evaluate import ranking_metrics


def test_perfect_ranking_gives_ndcg_one():
    scores = np.array([[0.9, 0.8, 0.1, 0.2, 0.3, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0]])
    rel = np.zeros_like(scores, dtype=bool)
    rel[0, [0, 1]] = True
    m = ranking_metrics(scores, np.zeros_like(rel), rel, k=10)
    assert abs(m["ndcg_at_10"] - 1.0) < 1e-9
    assert abs(m["precision_at_10"] - 0.2) < 1e-9
    assert abs(m["recall_at_10"] - 1.0) < 1e-9


def test_seen_items_are_never_recommended():
    scores = np.array([[5.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    rel = np.zeros_like(scores, dtype=bool)
    rel[0, 0] = True
    seen = np.zeros_like(rel)
    seen[0, 0] = True
    assert ranking_metrics(scores, seen, rel, k=3)["precision_at_10"] == 0.0


def test_hand_computed_ndcg():
    # relevant item at rank 2 -> DCG = 1/log2(3); IDCG = 1
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]])
    rel = np.zeros_like(scores, dtype=bool)
    rel[0, 1] = True
    m = ranking_metrics(scores, np.zeros_like(rel), rel, k=10)
    assert abs(m["ndcg_at_10"] - 1 / np.log2(3)) < 1e-9

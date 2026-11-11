import math

import pytest

from rag_eval_lite.metrics import hit_at_k, mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_found_in_top_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_precision_divides_by_k_even_when_fewer_results():
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)
    assert precision_at_k(["a", "b", "x", "y"], {"a", "b"}, 4) == 0.5


def test_duplicates_do_not_inflate_scores():
    assert precision_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_ndcg_binary_matches_hand_calculation():
    # relevant at ranks 1 and 3; ideal puts them at ranks 1 and 2
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_graded_prefers_high_gain_first():
    gains = {"a": 3, "b": 1}
    assert ndcg_at_k(["a", "b"], gains, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], gains, 2) < 1.0


def test_ndcg_ideal_is_capped_at_k():
    # three relevant, k=1: finding one at rank 1 is a perfect top-1
    assert ndcg_at_k(["a"], {"a", "b", "c"}, 1) == pytest.approx(1.0)


def test_empty_relevant_set_scores_zero():
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert ndcg_at_k(["a"], {}, 3) == 0.0


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5

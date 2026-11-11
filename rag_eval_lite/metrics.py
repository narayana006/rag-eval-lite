"""Rank-based retrieval metrics.

Every function takes the ranked list of retrieved ids (best first) and the
relevant ids for one question, and returns a float between 0 and 1. Duplicate
ids in the ranking are ignored after their first appearance, so a retriever
cannot score higher by returning the same passage twice.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _dedupe(retrieved: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top k."""
    _check_k(k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k slots filled with a relevant id.

    The denominator is always k, so returning fewer than k results is not
    rewarded.
    """
    _check_k(k)
    relevant = set(relevant)
    top = _dedupe(retrieved)[:k]
    return sum(1 for item in top if item in relevant) / k


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant id is in the top k, else 0.0."""
    _check_k(k)
    relevant = set(relevant)
    return 1.0 if any(item in relevant for item in _dedupe(retrieved)[:k]) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None
) -> float:
    """1 / rank of the first relevant id (0.0 if none is found in the top k)."""
    if k is not None:
        _check_k(k)
    relevant = set(relevant)
    ranked = _dedupe(retrieved)
    if k is not None:
        ranked = ranked[:k]
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str] | Mapping[str, float],
    k: int,
) -> float:
    """Normalised discounted cumulative gain over the top k.

    ``relevant`` is either a collection of ids (binary gain of 1) or a mapping
    of id to a graded gain such as ``{"a": 3, "b": 1}``.
    """
    _check_k(k)
    if isinstance(relevant, Mapping):
        gains = {key: float(value) for key, value in relevant.items() if value > 0}
    else:
        gains = {key: 1.0 for key in relevant}
    if not gains:
        return 0.0
    top = _dedupe(retrieved)[:k]
    dcg = sum(gains.get(item, 0.0) / math.log2(i + 2) for i, item in enumerate(top))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

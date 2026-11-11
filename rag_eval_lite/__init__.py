"""Offline retrieval metrics and deterministic answer checks for RAG pipelines."""

from .checks import contains, exact_match, groundedness, token_f1
from .metrics import hit_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .runner import evaluate

__version__ = "0.1.0"

__all__ = [
    "contains",
    "evaluate",
    "exact_match",
    "groundedness",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "token_f1",
]

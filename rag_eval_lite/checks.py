"""Deterministic answer checks. No model calls, same input gives same score."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({ch: " " for ch in string.punctuation})

# Words that carry no evidence either way when checking an answer against sources.
STOPWORDS = frozenset(
    """a an the and or but if then so of in on at to for from by with as is are was
    were be been being it its this that these those there here you your we our they
    their he she his her i me my not no yes do does did can could will would should
    may might must have has had what which who whom when where why how than too very
    also just about into over under up down out per each any all some more most""".split()
)


def normalize(text: str) -> str:
    """Lower-case, drop punctuation and articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(answer: str, expected: str) -> float:
    """1.0 if the answers are identical after normalisation."""
    return 1.0 if normalize(answer) == normalize(expected) else 0.0


def contains(answer: str, expected: str) -> float:
    """1.0 if the normalised expected answer appears as whole words in the answer."""
    needle = tokens(expected)
    haystack = tokens(answer)
    if not needle:
        return 0.0
    n = len(needle)
    return 1.0 if any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1)) else 0.0


def token_f1(answer: str, expected: str) -> float:
    """Token-overlap F1 between answer and expected answer (SQuAD style)."""
    got, want = tokens(answer), tokens(expected)
    if not got or not want:
        return 1.0 if got == want else 0.0
    overlap = sum((Counter(got) & Counter(want)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(got)
    recall = overlap / len(want)
    return 2 * precision * recall / (precision + recall)


def groundedness(answer: str, passages: Sequence[str]) -> float | None:
    """Share of the answer's content words that occur in the retrieved passages.

    A cheap lexical stand-in for faithfulness: a low score means the answer
    uses words the sources never mention. It cannot detect a wrong statement
    made from the right words. Returns None when there is nothing to check
    (no content words in the answer).
    """
    content = [tok for tok in tokens(answer) if tok not in STOPWORDS]
    if not content:
        return None
    source_vocab: set[str] = set()
    for passage in passages:
        source_vocab.update(tokens(passage))
    return sum(1 for tok in content if tok in source_vocab) / len(content)


def phrase_check(answer: str, must_contain: Iterable[str] = (), must_not_contain: Iterable[str] = ()) -> float:
    """1.0 only if every required phrase is present and no forbidden phrase is."""
    if not all(contains(answer, phrase) for phrase in must_contain):
        return 0.0
    if any(contains(answer, phrase) for phrase in must_not_contain):
        return 0.0
    return 1.0

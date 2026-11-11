"""Rule-based question generation: fill-in-the-blank questions with known sources.

No model is involved. Each question is a sentence from the corpus with one
distinctive term blanked out, so the answer and the source passage are known
exactly. That is enough to measure retrieval; the questions are easier than
real user questions because they share wording with the source.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from .checks import STOPWORDS, tokens
from .corpus import Passage

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\b\d[\d,.]*\d\b|\b\d\b")
_WORD = re.compile(r"\b[A-Za-z][A-Za-z-]{4,}\b")


def _sentences(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(("#", "|", "```"))]
    flat = " ".join(" ".join(lines).split())
    return [s.strip() for s in _SENTENCE.split(flat) if s.strip()]


def _pick_answer(sentence: str, doc_freq: Counter) -> str | None:
    """Prefer a number; otherwise the word that appears in the fewest passages."""
    number = _NUMBER.search(sentence)
    if number:
        return number.group(0)
    candidates = [w for w in _WORD.findall(sentence) if w.lower() not in STOPWORDS]
    if not candidates:
        return None
    return min(candidates, key=lambda w: (doc_freq[w.lower()], -len(w), w))


def generate_questions(passages: list[Passage], n: int, seed: int = 0) -> list[dict]:
    """Return up to ``n`` cloze questions, spread across passages, in a stable order."""
    doc_freq: Counter = Counter()
    for passage in passages:
        doc_freq.update(set(tokens(passage.text)))
    candidates: list[dict] = []
    for passage in passages:
        for sentence in _sentences(passage.text):
            if not 6 <= len(sentence.split()) <= 40:
                continue
            answer = _pick_answer(sentence, doc_freq)
            if answer is None:
                continue
            blanked = re.sub(rf"(?<!\w){re.escape(answer)}(?!\w)", "____", sentence, count=1)
            if blanked == sentence:
                continue
            candidates.append(
                {
                    "question": f"Fill in the blank: {blanked}",
                    "expected_answer": answer,
                    "relevant_ids": [passage.id],
                }
            )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Round-robin over passages so one long document does not dominate the set.
    by_passage: dict[str, list[dict]] = {}
    for item in candidates:
        by_passage.setdefault(item["relevant_ids"][0], []).append(item)
    picked: list[dict] = []
    queues = list(by_passage.values())
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return [{"id": f"q{number:03d}", **item} for number, item in enumerate(picked, start=1)]

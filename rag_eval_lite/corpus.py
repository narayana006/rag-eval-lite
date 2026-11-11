"""Corpus loading and a small BM25 baseline retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .checks import STOPWORDS, tokens

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str


def load_corpus(root: str | Path) -> list[Passage]:
    """Split every text file under ``root`` into paragraph passages.

    Passage ids look like ``billing.md#2``: the path relative to ``root`` and
    the 1-based paragraph number. Markdown headings stay attached to the
    paragraph that follows them.
    """
    root = Path(root)
    if root.is_file():
        files, base = [root], root.parent
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        base = root
    else:
        raise FileNotFoundError(f"corpus path not found: {root}")
    passages: list[Passage] = []
    for path in files:
        rel = path.relative_to(base).as_posix()
        blocks = [b.strip() for b in re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))]
        merged: list[str] = []
        pending = ""
        for block in blocks:
            if not block:
                continue
            if block.startswith("#") and "\n" not in block:
                pending = f"{pending}\n{block}".strip()
                continue
            merged.append(f"{pending}\n{block}".strip())
            pending = ""
        if pending:
            merged.append(pending)
        for number, text in enumerate(merged, start=1):
            passages.append(Passage(id=f"{rel}#{number}", text=text, source=rel))
    return passages


class BM25Retriever:
    """Okapi BM25 over passages. A baseline to compare your own retriever against."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self._tf = [Counter(self._terms(p.text)) for p in passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(passages)
        self._idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [tok for tok in tokens(text) if tok not in STOPWORDS]

    def __call__(self, question: str, k: int) -> list[dict]:
        query = self._terms(question)
        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self.k1 * (1 - self.b + self.b * self._len[index] / self._avg_len)
                score += self._idf[term] * freq * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            {"id": self.passages[i].id, "text": self.passages[i].text, "score": round(score, 4)}
            for score, i in scored[:k]
        ]

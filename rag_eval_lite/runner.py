"""Run a question set through a retriever (and optionally an answer step) and score it."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import checks, metrics

Retriever = Callable[[str, int], Sequence[Any]]
Answerer = Callable[[str, Sequence[str]], str]


class Judge(Protocol):
    """Optional extra scorer, for example one backed by a language model.

    Nothing in this package implements it; bring your own. It must return a
    number between 0 and 1.
    """

    def score(self, question: str, answer: str, passages: Sequence[str], expected: str | None) -> float: ...


@dataclass
class Example:
    id: str
    question: str
    relevant_ids: list[str] | dict[str, float]
    expected_answer: str | None = None
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: each line must be a JSON object")
            rows.append(row)
    return rows


def load_dataset(path: str | Path) -> list[Example]:
    examples = []
    for number, row in enumerate(read_jsonl(path), start=1):
        if "question" not in row or "relevant_ids" not in row:
            raise ValueError(f"{path}: record {number} needs 'question' and 'relevant_ids'")
        examples.append(
            Example(
                id=str(row.get("id", f"q{number:03d}")),
                question=row["question"],
                relevant_ids=row["relevant_ids"],
                expected_answer=row.get("expected_answer"),
                must_contain=list(row.get("must_contain", [])),
                must_not_contain=list(row.get("must_not_contain", [])),
            )
        )
    return examples


def load_callable(spec: str) -> Callable:
    """Import ``module:attribute`` (the current directory is importable)."""
    if ":" not in spec:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    target = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def _normalise_hits(hits: Sequence[Any]) -> tuple[list[str], list[str | None]]:
    """Accept ids, (id, text) pairs, dicts or objects with ``id``/``text``."""
    ids: list[str] = []
    texts: list[str | None] = []
    for hit in hits:
        if isinstance(hit, str):
            ident, text = hit, None
        elif isinstance(hit, Mapping):
            ident, text = hit["id"], hit.get("text")
        elif isinstance(hit, (tuple, list)):
            ident, text = hit[0], (hit[1] if len(hit) > 1 else None)
        else:
            ident, text = getattr(hit, "id"), getattr(hit, "text", None)
        ids.append(str(ident))
        texts.append(text)
    return ids, texts


def evaluate(
    examples: Sequence[Example],
    retriever: Retriever | None = None,
    answerer: Answerer | None = None,
    k: int = 5,
    judge: Judge | None = None,
    passage_texts: Mapping[str, str] | None = None,
    run: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Score every example and return ``{"k", "summary", "questions"}``.

    Results come either from calling ``retriever`` (and ``answerer``) or from
    ``run``, a mapping of example id to ``{"retrieved_ids": [...], "answer": "..."}``
    recorded from any pipeline.
    """
    if (retriever is None) == (run is None):
        raise ValueError("pass exactly one of retriever or run")
    passage_texts = passage_texts or {}
    rows = []
    for example in examples:
        if run is not None:
            record = run.get(example.id)
            if record is None:
                raise ValueError(f"run has no record for question id {example.id!r}")
            ids = [str(i) for i in record.get("retrieved_ids", [])]
            texts: list[str | None] = list(record.get("passages", [])) or [None] * len(ids)
            answer = record.get("answer")
        else:
            ids, texts = _normalise_hits(retriever(example.question, k))
            answer = None
        ids, texts = ids[:k], (texts + [None] * len(ids))[:k]
        passages = [t if t is not None else passage_texts.get(i) for i, t in zip(ids, texts)]
        passages = [p for p in passages if p]
        if run is None and answerer is not None:
            answer = answerer(example.question, passages)

        relevant = example.relevant_ids
        scores: dict[str, float | None] = {
            f"recall@{k}": metrics.recall_at_k(ids, relevant, k),
            f"precision@{k}": metrics.precision_at_k(ids, relevant, k),
            f"hit@{k}": metrics.hit_at_k(ids, relevant, k),
            "mrr": metrics.reciprocal_rank(ids, relevant, k),
            f"ndcg@{k}": metrics.ndcg_at_k(ids, relevant, k),
        }
        if answer is not None:
            if example.expected_answer is not None:
                scores["exact_match"] = checks.exact_match(answer, example.expected_answer)
                scores["contains"] = checks.contains(answer, example.expected_answer)
                scores["token_f1"] = checks.token_f1(answer, example.expected_answer)
            if example.must_contain or example.must_not_contain:
                scores["phrase_check"] = checks.phrase_check(answer, example.must_contain, example.must_not_contain)
            if passages:
                scores["groundedness"] = checks.groundedness(answer, passages)
            if judge is not None:
                scores["judge"] = min(1.0, max(0.0, float(judge.score(example.question, answer, passages, example.expected_answer))))
        rows.append(
            {
                "id": example.id,
                "question": example.question,
                "relevant_ids": sorted(relevant),
                "retrieved_ids": ids,
                "expected_answer": example.expected_answer,
                "answer": answer,
                "scores": {name: (None if value is None else round(value, 4)) for name, value in scores.items()},
            }
        )

    summary: dict[str, dict[str, float | int]] = {}
    for row in rows:
        for name, value in row["scores"].items():
            if value is not None:
                summary.setdefault(name, {"sum": 0.0, "n": 0})
                summary[name]["sum"] += value
                summary[name]["n"] += 1
    return {
        "k": k,
        "n_questions": len(rows),
        "summary": {name: {"mean": round(agg["sum"] / agg["n"], 4), "n": agg["n"]} for name, agg in summary.items()},
        "questions": rows,
    }

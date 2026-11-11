"""Command line interface: generate, score, compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .corpus import BM25Retriever, load_corpus
from .generate import generate_questions
from .report import format_comparison, format_summary, render_html
from .runner import evaluate, load_callable, load_dataset, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval-lite", description=__doc__)
    parser.add_argument("--version", action="version", version=f"rag-eval-lite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write fill-in-the-blank questions with known source passages")
    gen.add_argument("corpus", help="directory (or file) of .txt/.md documents")
    gen.add_argument("--n", type=int, default=50, help="number of questions (default 50)")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", default="-", help="output JSONL path (default stdout)")

    score = sub.add_parser("score", help="score a retriever, or a recorded run, against a question set")
    score.add_argument("--dataset", required=True, help="JSONL question set")
    score.add_argument("--corpus", help="document directory; enables the built-in BM25 baseline")
    score.add_argument("--retriever", help="module:function taking (question, k); default is BM25 over --corpus")
    score.add_argument("--answerer", help="module:function taking (question, passages) and returning an answer")
    score.add_argument("--run", help="JSONL of recorded results: id, retrieved_ids, answer")
    score.add_argument("--judge", help="module:factory returning an object with score(question, answer, passages, expected)")
    score.add_argument("-k", "--k", type=int, default=5, help="rank cutoff (default 5)")
    score.add_argument("--json", dest="json_out", help="write full results as JSON")
    score.add_argument("--report", help="write a single-file HTML report")
    score.add_argument("--fail-under", metavar="METRIC=VALUE", action="append", default=[],
                       help="exit 1 if a mean falls below a threshold, e.g. recall@5=0.8")

    cmp_ = sub.add_parser("compare", help="compare two results saved with score --json")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    return parser


def _cmd_generate(args) -> int:
    questions = generate_questions(load_corpus(args.corpus), args.n, seed=args.seed)
    text = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {len(questions)} questions to {args.out}")
    return 0


def _cmd_score(args) -> int:
    examples = load_dataset(args.dataset)
    passages = load_corpus(args.corpus) if args.corpus else []
    texts = {p.id: p.text for p in passages}
    judge = load_callable(args.judge)() if args.judge else None
    if args.run:
        if args.retriever or args.answerer:
            raise ValueError("--run cannot be combined with --retriever or --answerer")
        run = {str(row["id"]): row for row in read_jsonl(args.run)}
        result = evaluate(examples, run=run, k=args.k, judge=judge, passage_texts=texts)
    else:
        if args.retriever:
            retriever = load_callable(args.retriever)
        elif passages:
            retriever = BM25Retriever(passages)
        else:
            raise ValueError("give --retriever, --run, or --corpus (for the BM25 baseline)")
        answerer = load_callable(args.answerer) if args.answerer else None
        result = evaluate(examples, retriever=retriever, answerer=answerer, k=args.k, judge=judge, passage_texts=texts)

    print(format_summary(result))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        Path(args.report).write_text(render_html(result), encoding="utf-8")
        print(f"\nreport: {args.report}")

    failed = False
    for rule in args.fail_under:
        name, _, raw = rule.partition("=")
        if name not in result["summary"]:
            raise ValueError(f"--fail-under: unknown metric {name!r}; have {', '.join(result['summary'])}")
        mean = result["summary"][name]["mean"]
        if mean < float(raw):
            print(f"FAIL {name} = {mean:.3f} is below {float(raw):.3f}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def _cmd_compare(args) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    print(format_comparison(before, after))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"generate": _cmd_generate, "score": _cmd_score, "compare": _cmd_compare}[args.command]
    try:
        return handler(args)
    except (ValueError, FileNotFoundError, ImportError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

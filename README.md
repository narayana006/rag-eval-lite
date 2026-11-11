# rag-eval-lite

Put numbers on a RAG pipeline without calling a model: retrieval metrics (recall@k, precision@k, MRR, nDCG) and deterministic answer checks, from the command line or from Python.

**Status: first working version (0.1.0).** Retrieval metrics, deterministic answer checks, a pluggable retriever interface, a BM25 baseline, rule-based question generation, run comparison and a single-file HTML report all work today, offline, with no dependencies. Model-based faithfulness and relevance scoring is not built in; there is an optional `Judge` interface you can implement yourself. Question generation is fill-in-the-blank only, not model-written questions.

## Install

```bash
pip install git+https://github.com/narayana006/rag-eval-lite
```

Python 3.10 or newer. The package is not on PyPI (the name there belongs to a different project), so install from git.

## Quick start

The repository ships a small fictional help centre (`examples/docs`) and ten hand-written questions (`examples/qa.jsonl`). Score the built-in BM25 baseline on it:

```bash
git clone https://github.com/narayana006/rag-eval-lite
cd rag-eval-lite/examples
rag-eval-lite score --dataset qa.jsonl --corpus docs -k 3
```

```text
questions: 10   k: 3

metric              mean     n
recall@3           0.800    10
precision@3        0.300    10
hit@3              0.800    10
mrr                0.800    10
ndcg@3             0.800    10

no relevant passage in top 3:
  q02  How long do I have to ask for a refund?
  q06  What is the largest file I can attach?
```

Now score your own retriever and answer step. `examples/my_retriever.py` is a deliberately naive one (word overlap, then "pick the most similar sentence"):

```bash
rag-eval-lite score --dataset qa.jsonl --retriever my_retriever:search --answerer my_retriever:answer -k 3
```

```text
questions: 10   k: 3

metric              mean     n
recall@3           0.850    10
precision@3        0.300    10
hit@3              0.900    10
mrr                0.800    10
ndcg@3             0.787    10
exact_match        0.000     9
contains           0.444     9
token_f1           0.264     9
groundedness       1.000    10
phrase_check       0.500     2

no relevant passage in top 3:
  q06  What is the largest file I can attach?
```

The answer step copies sentences from the sources, so groundedness is perfect, but it only contains the expected answer 44% of the time. That gap is the point of measuring both.

## The question set

One JSON object per line:

```json
{"id": "q03", "question": "What happens to my workspace when the card keeps getting declined?", "relevant_ids": ["billing.md#4"], "expected_answer": "the workspace becomes read-only", "must_contain": ["read-only"]}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `question` | yes | The query sent to the retriever |
| `relevant_ids` | yes | Ids of the passages that answer it. A list, or a mapping of id to graded gain (`{"a": 3, "b": 1}`) for nDCG |
| `id` | no | Defaults to `q001`, `q002`, ... |
| `expected_answer` | no | Enables `exact_match`, `contains`, `token_f1` |
| `must_contain`, `must_not_contain` | no | Phrases for `phrase_check` |

Ids are whatever your retriever returns. With `--corpus`, files are split into paragraphs and ids are `<relative path>#<paragraph number>`, with a Markdown heading kept attached to the paragraph after it.

## Plugging in your pipeline

**A function.** `--retriever module:function` imports from the current directory or anything installed. The function takes `(question, k)` and returns a ranked list of ids, `(id, text)` pairs, or dicts with `id` and optional `text`. `--answerer module:function` takes `(question, passages)` and returns a string.

**A recorded run.** If your pipeline is not Python, or is slow, record its output and score the file:

```bash
rag-eval-lite score --dataset qa.jsonl --run run.jsonl
```

Each line of `run.jsonl` is `{"id": "q01", "retrieved_ids": ["..."], "answer": "...", "passages": ["..."]}`; `answer` and `passages` are optional.

**From Python.**

```python
from rag_eval_lite import evaluate, recall_at_k
from rag_eval_lite.runner import load_dataset

result = evaluate(load_dataset("qa.jsonl"), retriever=my_search, answerer=my_answer, k=5)
print(result["summary"]["recall@5"]["mean"])

recall_at_k(["a", "x", "b"], {"a", "b"}, k=2)   # 0.5
```

## What is measured

| Metric | Definition |
| --- | --- |
| `recall@k` | Share of the relevant passages found in the top k |
| `precision@k` | Relevant passages in the top k, divided by k |
| `hit@k` | 1 if any relevant passage is in the top k |
| `mrr` | 1 / rank of the first relevant passage within the top k, else 0 |
| `ndcg@k` | Discounted gain over the top k, normalised by the best possible ordering; binary or graded |
| `exact_match` | Answer equals the expected answer after lower-casing and removing punctuation and articles |
| `contains` | Expected answer appears in the answer as whole words |
| `token_f1` | Token-overlap F1 between answer and expected answer |
| `phrase_check` | All `must_contain` phrases present and no `must_not_contain` phrase present |
| `groundedness` | Share of the answer's non-stopword tokens that occur in the retrieved passages |

Duplicate ids in a ranking count once. The `n` column shows how many questions each mean covers, since answer checks only apply where the question set provides something to check against.

## Other commands

Save results, write the HTML report, and compare two runs:

```bash
rag-eval-lite score --dataset qa.jsonl --corpus docs -k 3 --json bm25.json
rag-eval-lite score --dataset qa.jsonl --retriever my_retriever:search --answerer my_retriever:answer -k 3 --json mine.json --report report.html
rag-eval-lite compare bm25.json mine.json
```

```text
metric            before   after   change
recall@3           0.800   0.850   +0.050
precision@3        0.300   0.300   +0.000
hit@3              0.800   0.900   +0.100
mrr                0.800   0.800   +0.000
ndcg@3             0.800   0.787   -0.013
exact_match            -   0.000        -
contains               -   0.444        -
token_f1               -   0.264        -
groundedness           -   1.000        -
phrase_check           -   0.500        -
```

Gate a CI job on a threshold (exit code 1 when a mean is below it):

```bash
rag-eval-lite score --dataset qa.jsonl --corpus docs -k 3 --fail-under recall@3=0.9
```

```text
FAIL recall@3 = 0.800 is below 0.900
```

Generate a starter question set from a corpus, with no model involved:

```bash
rag-eval-lite generate docs --n 4
```

```text
{"id": "q001", "question": "Fill in the blank: A workspace owner can restore a note from the trash or delete it ____.", "expected_answer": "permanently", "relevant_ids": ["sharing.md#4"]}
{"id": "q002", "question": "Fill in the blank: The Team plan costs ____ dollars per member per month, and the annual option gives two months free.", "expected_answer": "8", "relevant_ids": ["billing.md#1"]}
{"id": "q003", "question": "Fill in the blank: Notes sync between devices every ____ seconds while the app is open.", "expected_answer": "30", "relevant_ids": ["sync.md#1"]}
{"id": "q004", "question": "Fill in the blank: If a card ____ fails, Orbit Notes retries it three times over seven days.", "expected_answer": "payment", "relevant_ids": ["billing.md#4"]}
```

## Optional judge

For a score the deterministic checks cannot give, pass `--judge module:factory`. The factory is called with no arguments and must return an object with:

```python
def score(self, question: str, answer: str, passages: list[str], expected: str | None) -> float: ...
```

The result is clamped to 0..1 and reported as `judge`. Nothing in this package calls a model or the network; what the judge does (a call to any OpenAI-compatible HTTP endpoint, a local model, a rule) is up to you.

## Features

- [x] Retrieval metrics: recall@k, precision@k, hit@k, MRR, nDCG (binary and graded)
- [x] Deterministic answer checks: exact match, contains, token F1, phrase checks, lexical groundedness
- [x] Pluggable retriever interface (`module:function` or a recorded run file)
- [x] BM25 baseline retriever over a folder of text or Markdown files
- [x] Single-file HTML report, JSON results and `compare` between two runs
- [x] Rule-based (fill-in-the-blank) question generation from a corpus
- [ ] Model-written questions and answers from a corpus
- [ ] Built-in model-based faithfulness and relevance scoring (today: bring your own `Judge`)
- [ ] Side-by-side run comparison inside the HTML report

## Limitations

- `groundedness` is lexical. It catches answers that use words the sources never mention; it does not catch a false statement assembled from words that do appear, and it treats paraphrases as unsupported.
- Generated questions reuse the source sentence, so keyword retrievers score close to perfect on them. Use them to smoke-test a pipeline, and hand-written questions to judge quality.
- The corpus loader reads `.txt`, `.md`, `.markdown` and `.rst` and splits on blank lines. If your retriever chunks differently, write `relevant_ids` in your own id scheme and skip `--corpus`.
- Tokenisation is whitespace and punctuation based, with an English stopword list.
- Means are plain averages; no confidence intervals are reported, so small question sets give noisy numbers.

## Contributing

Issues and pull requests are welcome. To work on the code:

```bash
pip install -e ".[test]"
pytest
```

Please add a test with any change to a metric, including a hand-calculated expected value.

Built and maintained by [Narayana Vemburaj](https://github.com/narayana006).

## License

[MIT](LICENSE)

import json

import pytest

from rag_eval_lite.corpus import BM25Retriever, load_corpus
from rag_eval_lite.generate import generate_questions
from rag_eval_lite.runner import Example, evaluate, load_dataset


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "pets.md").write_text(
        "# Cats\n\nCats sleep for around 15 hours every day and hunt mostly at dusk.\n\n"
        "## Dogs\n\nDogs were domesticated from wolves and need a daily walk outside.\n",
        encoding="utf-8",
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "fish.txt").write_text("Goldfish can live for over a decade in a large pond.\n", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")
    return tmp_path


def test_load_corpus_ids_and_heading_attachment(corpus):
    passages = load_corpus(corpus)
    assert [p.id for p in passages] == ["pets.md#1", "pets.md#2", "sub/fish.txt#1"]
    assert passages[0].text.startswith("# Cats\nCats sleep")
    assert passages[1].text.startswith("## Dogs")


def test_load_corpus_missing_path():
    with pytest.raises(FileNotFoundError):
        load_corpus("/no/such/dir")


def test_bm25_ranks_matching_passage_first(corpus):
    retriever = BM25Retriever(load_corpus(corpus))
    hits = retriever("how long do goldfish live", 2)
    assert hits[0]["id"] == "sub/fish.txt#1"
    assert retriever("zzz qqq", 3) == []


def test_evaluate_with_retriever_and_answerer(corpus):
    passages = load_corpus(corpus)
    examples = [
        Example("q1", "How many hours do cats sleep?", ["pets.md#1"], expected_answer="15 hours"),
        Example("q2", "Where do goldfish live?", ["sub/fish.txt#1"], must_contain=["pond"]),
    ]
    result = evaluate(examples, retriever=BM25Retriever(passages), answerer=lambda q, ps: ps[0], k=2)
    assert result["n_questions"] == 2
    assert result["summary"]["recall@2"] == {"mean": 1.0, "n": 2}
    assert result["summary"]["mrr"]["mean"] == 1.0
    assert result["summary"]["contains"] == {"mean": 1.0, "n": 1}
    assert result["summary"]["phrase_check"] == {"mean": 1.0, "n": 1}
    assert result["summary"]["groundedness"]["mean"] == 1.0
    assert "judge" not in result["summary"]


def test_retriever_may_return_ids_pairs_or_objects():
    class Hit:
        id, text = "c", "third"

    def retriever(question, k):
        return ["a", ("b", "second"), Hit()]

    result = evaluate([Example("q", "?", ["c"])], retriever=retriever, k=3, passage_texts={"a": "first"})
    row = result["questions"][0]
    assert row["retrieved_ids"] == ["a", "b", "c"]
    assert row["scores"]["mrr"] == pytest.approx(0.3333)


def test_results_are_truncated_to_k():
    result = evaluate([Example("q", "?", ["c"])], retriever=lambda q, k: ["a", "b", "c"], k=2)
    assert result["questions"][0]["retrieved_ids"] == ["a", "b"]
    assert result["summary"]["recall@2"]["mean"] == 0.0


def test_evaluate_recorded_run():
    examples = [Example("q1", "?", ["p1"], expected_answer="blue")]
    run = {"q1": {"retrieved_ids": ["p9", "p1"], "answer": "It is blue.", "passages": ["sky", "The sky is blue."]}}
    result = evaluate(examples, run=run, k=5)
    scores = result["questions"][0]["scores"]
    assert scores["mrr"] == 0.5
    assert scores["contains"] == 1.0
    assert scores["exact_match"] == 0.0
    assert scores["groundedness"] == 1.0


def test_run_missing_question_is_an_error():
    with pytest.raises(ValueError, match="q1"):
        evaluate([Example("q1", "?", ["p1"])], run={}, k=3)


def test_needs_exactly_one_source_of_results():
    with pytest.raises(ValueError):
        evaluate([], k=3)


def test_optional_judge_is_called_and_clamped():
    class FakeJudge:
        def __init__(self):
            self.calls = []

        def score(self, question, answer, passages, expected):
            self.calls.append((question, answer, tuple(passages), expected))
            return 1.7

    judge = FakeJudge()
    result = evaluate(
        [Example("q1", "colour?", ["p1"], expected_answer="blue")],
        retriever=lambda q, k: [("p1", "The sky is blue.")],
        answerer=lambda q, ps: "blue",
        judge=judge,
        k=1,
    )
    assert judge.calls == [("colour?", "blue", ("The sky is blue.",), "blue")]
    assert result["summary"]["judge"] == {"mean": 1.0, "n": 1}


def test_load_dataset_validates(tmp_path):
    good = tmp_path / "qa.jsonl"
    good.write_text(json.dumps({"question": "q?", "relevant_ids": ["a"]}) + "\n\n", encoding="utf-8")
    assert load_dataset(good)[0].id == "q001"
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q?"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_ids"):
        load_dataset(bad)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.jsonl:1"):
        load_dataset(broken)


def test_generate_is_deterministic_and_answerable(corpus):
    passages = load_corpus(corpus)
    first = generate_questions(passages, n=3, seed=1)
    assert first == generate_questions(passages, n=3, seed=1)
    assert len(first) == 3
    by_id = {p.id: p.text for p in passages}
    for item in first:
        assert "____" in item["question"]
        assert item["expected_answer"] in by_id[item["relevant_ids"][0]]
    # spread across passages before repeating one
    assert len({item["relevant_ids"][0] for item in first}) == 3
    assert any(item["expected_answer"] == "15" for item in generate_questions(passages, n=10))

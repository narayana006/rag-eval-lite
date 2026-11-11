import json
from pathlib import Path

import pytest

from rag_eval_lite.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_score_with_bm25_baseline_writes_json_and_html(tmp_path, capsys):
    out_json, out_html = tmp_path / "r.json", tmp_path / "r.html"
    code = main(["score", "--dataset", str(EXAMPLES / "qa.jsonl"), "--corpus", str(EXAMPLES / "docs"),
                 "-k", "3", "--json", str(out_json), "--report", str(out_html)])
    assert code == 0
    assert "recall@3" in capsys.readouterr().out
    result = json.loads(out_json.read_text())
    assert result["n_questions"] == 10
    html = out_html.read_text()
    assert html.startswith("<!doctype html>") and "billing.md#1" in html


def test_score_with_custom_retriever_and_answerer(monkeypatch, capsys):
    monkeypatch.chdir(EXAMPLES)
    code = main(["score", "--dataset", "qa.jsonl", "--retriever", "my_retriever:search",
                 "--answerer", "my_retriever:answer", "-k", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "groundedness" in out and "token_f1" in out


def test_score_recorded_run_and_fail_under(tmp_path, capsys):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(json.dumps({"id": "a", "question": "q?", "relevant_ids": ["p1"]}) + "\n")
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({"id": "a", "retrieved_ids": ["p2", "p3"]}) + "\n")
    assert main(["score", "--dataset", str(dataset), "--run", str(run), "--fail-under", "recall@5=0.5"]) == 1
    assert "FAIL recall@5" in capsys.readouterr().err
    assert main(["score", "--dataset", str(dataset), "--run", str(run), "--fail-under", "recall@5=0"]) == 0


def test_score_without_any_source_is_a_usage_error(tmp_path, capsys):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(json.dumps({"question": "q?", "relevant_ids": ["p1"]}) + "\n")
    assert main(["score", "--dataset", str(dataset)]) == 2
    assert "error:" in capsys.readouterr().err


def test_html_report_escapes_content(tmp_path):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(json.dumps({"id": "a", "question": "<script>x</script>", "relevant_ids": ["p1"]}) + "\n")
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({"id": "a", "retrieved_ids": ["p1"], "answer": "<b>hi</b>"}) + "\n")
    report = tmp_path / "r.html"
    assert main(["score", "--dataset", str(dataset), "--run", str(run), "--report", str(report)]) == 0
    html = report.read_text()
    assert "<script>x" not in html and "&lt;b&gt;hi" in html


def test_generate_then_score_round_trip(tmp_path, capsys):
    qa = tmp_path / "gen.jsonl"
    assert main(["generate", str(EXAMPLES / "docs"), "--n", "6", "--out", str(qa)]) == 0
    assert len(qa.read_text().splitlines()) == 6
    capsys.readouterr()
    assert main(["score", "--dataset", str(qa), "--corpus", str(EXAMPLES / "docs")]) == 0


def test_compare_shows_change(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"summary": {"mrr": {"mean": 0.5, "n": 2}}}))
    b.write_text(json.dumps({"summary": {"mrr": {"mean": 0.75, "n": 2}, "judge": {"mean": 1.0, "n": 2}}}))
    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "+0.250" in out and "judge" in out

import pytest

from rag_eval_lite.checks import contains, exact_match, groundedness, normalize, phrase_check, token_f1


def test_normalize_strips_case_punctuation_and_articles():
    assert normalize("The  Read-Only, mode!") == "read only mode"


def test_exact_match_ignores_formatting():
    assert exact_match("14 days.", "14 Days") == 1.0
    assert exact_match("about 14 days", "14 days") == 0.0


def test_contains_matches_whole_words_only():
    assert contains("Refunds are available within 14 days of a charge", "14 days") == 1.0
    assert contains("It takes 114 days", "14 days") == 0.0
    assert contains("anything", "") == 0.0


def test_token_f1():
    assert token_f1("14 days", "14 days") == 1.0
    assert token_f1("within 14 days", "14 days") == pytest.approx(0.8)
    assert token_f1("a month", "14 days") == 0.0
    assert token_f1("", "14 days") == 0.0


def test_groundedness_is_share_of_supported_content_words():
    passages = ["Refunds are available within 14 days of a charge."]
    assert groundedness("Refunds are available within 14 days.", passages) == 1.0
    # content words: refunds, take, 90, days -> 2 of 4 supported
    assert groundedness("Refunds take 90 days.", passages) == pytest.approx(0.5)


def test_groundedness_none_when_answer_has_no_content_words():
    assert groundedness("It is.", ["anything"]) is None


def test_phrase_check_requires_all_and_forbids_any():
    answer = "The workspace becomes read-only."
    assert phrase_check(answer, must_contain=["read-only"]) == 1.0
    assert phrase_check(answer, must_contain=["read-only", "deleted"]) == 0.0
    assert phrase_check(answer, must_contain=["read-only"], must_not_contain=["workspace"]) == 0.0

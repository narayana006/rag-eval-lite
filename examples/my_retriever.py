"""A deliberately simple retriever and answer step to evaluate.

Replace these two functions with calls into your own pipeline. The retriever
takes (question, k) and returns ranked passage ids, optionally with text. The
answerer takes (question, passages) and returns a string.
"""

from pathlib import Path

from rag_eval_lite.corpus import load_corpus

PASSAGES = load_corpus(Path(__file__).parent / "docs")


def _words(text):
    return {word.strip(".,?!#").lower() for word in text.split()}


def search(question, k):
    """Rank passages by how many question words they share. No weighting at all."""
    query = _words(question)
    ranked = sorted(PASSAGES, key=lambda p: len(query & _words(p.text)), reverse=True)
    return [{"id": p.id, "text": p.text} for p in ranked[:k]]


def answer(question, passages):
    """Return the sentence from the top passages that shares the most words with the question."""
    query = _words(question)
    sentences = [s.strip() for p in passages[:2] for s in p.replace("\n", ". ").split(". ") if s.strip()]
    if not sentences:
        return "I do not know."
    return max(sentences, key=lambda s: len(query & _words(s)))

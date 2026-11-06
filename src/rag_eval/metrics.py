"""Retrieval metrics and SQuAD v1.1 style answer metrics."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence, Set

from rag_eval.errors import RagEvalError

MAX_ANSWER_CHARS = 1_048_576

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


def validate_k(k: int) -> None:
    """Raise ``RagEvalError('invalid_argument')`` unless ``k`` is an integer >= 1."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RagEvalError("invalid_argument", "k must be an integer >= 1")


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def retrieval_metrics(expected: Set[str], retrieved: Sequence[str], k: int) -> dict[str, float]:
    """Compute recall, precision, MRR and nDCG for one ranked list.

    The retrieved list is deduplicated keeping the first occurrence. ``recall@k`` is
    ``|rel & top-k| / |rel|``; ``precision@k`` divides by ``k`` (TREC convention); ``mrr`` uses
    the rank of the first relevant document in the whole list; ``ndcg@k`` uses binary gains and
    a ``log2(rank + 1)`` discount, with the ideal ranking holding ``min(|rel|, k)`` hits.

    Raises:
        RagEvalError: ``invalid_argument`` for ``k < 1``, an empty ``expected`` set or a
            retrieved list that is not a sequence of strings.
    """
    validate_k(k)
    if isinstance(expected, (str, bytes)) or not expected:
        raise RagEvalError("invalid_argument", "expected must be a non-empty set of ids")
    if isinstance(retrieved, (str, bytes)):
        raise RagEvalError("invalid_argument", "retrieved must be a sequence of ids")
    seen: set[str] = set()
    ranked: list[str] = []
    for doc in retrieved:
        if not isinstance(doc, str):
            raise RagEvalError("invalid_argument", "retrieved ids must be strings")
        if doc not in seen:
            seen.add(doc)
            ranked.append(doc)
    top = ranked[:k]
    hits = sum(1 for doc in top if doc in expected)
    first = next((rank for rank, doc in enumerate(ranked, start=1) if doc in expected), None)
    mrr = 0.0 if first is None else 1.0 / first
    dcg = sum(_discount(rank) for rank, doc in enumerate(top, start=1) if doc in expected)
    idcg = sum(_discount(rank) for rank in range(1, min(len(expected), k) + 1))
    return {
        f"recall@{k}": hits / len(expected),
        f"precision@{k}": hits / k,
        "mrr": mrr,
        f"ndcg@{k}": dcg / idcg,
    }


def normalize_answer(text: str) -> str:
    """Normalize like the official SQuAD v1.1 script.

    Lowercase, remove ASCII punctuation, remove the articles a, an and the, collapse whitespace.
    """
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def answer_metrics(pred: str, ref: str) -> dict[str, float]:
    """Return ``exact_match`` and bag-of-tokens ``token_f1`` of two normalized answers.

    Two empty normalized answers score 1.0; exactly one empty scores 0.0.

    Raises:
        RagEvalError: ``invalid_argument`` when an argument is not a string or is longer than
            ``MAX_ANSWER_CHARS`` characters.
    """
    for value in (pred, ref):
        if not isinstance(value, str):
            raise RagEvalError("invalid_argument", "answers must be strings")
        if len(value) > MAX_ANSWER_CHARS:
            raise RagEvalError(
                "invalid_argument", f"answers must have at most {MAX_ANSWER_CHARS} characters"
            )
    norm_pred = normalize_answer(pred)
    norm_ref = normalize_answer(ref)
    exact = 1.0 if norm_pred == norm_ref else 0.0
    pred_toks = norm_pred.split()
    ref_toks = norm_ref.split()
    f1: float
    if not pred_toks or not ref_toks:
        f1 = 1.0 if pred_toks == ref_toks else 0.0
    else:
        common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if common == 0:
            f1 = 0.0
        else:
            precision = common / len(pred_toks)
            recall = common / len(ref_toks)
            f1 = 2 * precision * recall / (precision + recall)
    return {"exact_match": exact, "token_f1": f1}

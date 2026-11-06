"""Offline evaluation of a RAG system and the resulting report."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rag_eval.errors import RagEvalError, clip
from rag_eval.loader import Kind, validate_record
from rag_eval.metrics import answer_metrics, retrieval_metrics, validate_k

logger = logging.getLogger(__name__)

REPORT_VERSION = 1
_DECIMALS = 6
_ANSWER_METRICS = ("exact_match", "token_f1")


def _rounded(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, _DECIMALS) if isinstance(value, float) else value
        for key, value in item.items()
    }


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


@dataclass(frozen=True)
class EvalReport:
    """Result of :func:`evaluate`.

    Attributes:
        per_item: one dict per dataset item (sorted by id) holding ``id`` and its metrics.
        means: mean of every metric over the items it applies to; metrics without any
            contributing item are absent.
        missing_results: ids of dataset items without a results line.
        skipped: ids of items excluded from retrieval means (empty ``expected_doc_ids``).
        k: the cut-off used for the ``@k`` metrics.
    """

    per_item: tuple[dict[str, Any], ...]
    means: Mapping[str, float]
    missing_results: tuple[str, ...]
    skipped: tuple[str, ...]
    k: int = 5

    def known_metrics(self) -> frozenset[str]:
        """Return the metric names this report can contain."""
        names = [f"recall@{self.k}", f"precision@{self.k}", "mrr", f"ndcg@{self.k}"]
        return frozenset([*names, *_ANSWER_METRICS])

    def to_json(self) -> str:
        """Serialize deterministically: sorted keys, indent 2, floats rounded to 6 decimals."""
        items = sorted(self.per_item, key=lambda item: item["id"])
        payload = {
            "k": self.k,
            "means": {name: round(value, _DECIMALS) for name, value in self.means.items()},
            "missing_results": sorted(self.missing_results),
            "per_item": [_rounded(item) for item in items],
            "skipped": sorted(self.skipped),
            "version": REPORT_VERSION,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)

    def check(self, thresholds: Mapping[str, float]) -> list[str]:
        """Return the sorted names of metrics whose mean is below its threshold or absent.

        Raises:
            RagEvalError: ``unknown_metric`` for a name this report cannot contain,
                ``invalid_argument`` for a threshold that is not a finite number.
        """
        known = self.known_metrics()
        failures: list[str] = []
        for name in sorted(thresholds):
            if name not in known:
                raise RagEvalError("unknown_metric", f"unknown metric '{clip(str(name), 80)}'")
            limit = thresholds[name]
            if not _finite_number(limit):
                raise RagEvalError("invalid_argument", f"threshold for '{name}' must be finite")
            mean = self.means.get(name)
            if mean is None or mean < limit:
                failures.append(name)
        return failures

    def worst(self, metric: str, n: int) -> list[str]:
        """Return the ids of the ``n`` lowest scoring items for ``metric`` (ties by id).

        Raises:
            RagEvalError: ``unknown_metric`` or ``invalid_argument`` (``n`` must be >= 0).
        """
        if metric not in self.known_metrics():
            raise RagEvalError("unknown_metric", f"unknown metric '{clip(str(metric), 80)}'")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise RagEvalError("invalid_argument", "n must be an integer >= 0")
        scored = sorted((item[metric], item["id"]) for item in self.per_item if metric in item)
        return [rid for _, rid in scored[:n]]


def _index(
    records: Sequence[Mapping[str, Any]],
    kind: Kind,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, Mapping):
            raise RagEvalError(kind, "every record must be a mapping")
        clean = validate_record(kind, rec)
        rid = clean["id"]
        if rid in indexed:
            raise RagEvalError(kind, f"duplicate id '{clip(rid, 80)}'")
        indexed[rid] = clean
    return indexed


def evaluate(
    dataset: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
) -> EvalReport:
    """Score ``results`` against the golden ``dataset``.

    Results for ids absent from the dataset are ignored. A dataset item without results scores
    0 on every metric that applies to it and is listed in ``missing_results``. Items with empty
    ``expected_doc_ids`` are excluded from retrieval means and listed in ``skipped``. Answer
    metrics need both a ``reference_answer`` and an ``answer``.

    Raises:
        RagEvalError: ``invalid_argument`` for ``k < 1``; ``dataset`` or ``results`` for
            malformed or duplicate records.
    """
    validate_k(k)
    items = _index(dataset, "dataset")
    found = _index(results, "results")
    per_item: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[str] = []
    for rid in sorted(items):
        item = items[rid]
        result = found.get(rid)
        entry: dict[str, Any] = {"id": rid}
        if result is None:
            missing.append(rid)
        expected = set(item["expected_doc_ids"])
        if expected:
            retrieved = [] if result is None else result["retrieved_doc_ids"]
            entry.update(retrieval_metrics(expected, retrieved, k))
        else:
            skipped.append(rid)
        reference = item.get("reference_answer")
        if reference is not None:
            if result is None:
                entry.update({name: 0.0 for name in _ANSWER_METRICS})
            elif "answer" in result:
                entry.update(answer_metrics(result["answer"], reference))
        per_item.append(entry)
    columns: dict[str, list[float]] = {}
    for entry in per_item:
        for name, value in entry.items():
            if name != "id":
                columns.setdefault(name, []).append(value)
    means = {name: math.fsum(values) / len(values) for name, values in sorted(columns.items())}
    logger.debug(
        "evaluated %d items (%d missing, %d skipped)", len(per_item), len(missing), len(skipped)
    )
    return EvalReport(
        per_item=tuple(per_item),
        means=MappingProxyType(means),
        missing_results=tuple(missing),
        skipped=tuple(skipped),
        k=k,
    )

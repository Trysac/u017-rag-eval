"""Offline RAG evaluation: retrieval and SQuAD-style answer metrics without any LLM call."""

from __future__ import annotations

import logging

from rag_eval.errors import RagEvalError
from rag_eval.loader import load_jsonl
from rag_eval.metrics import answer_metrics, retrieval_metrics
from rag_eval.report import EvalReport, evaluate

__all__ = [
    "EvalReport",
    "RagEvalError",
    "__version__",
    "answer_metrics",
    "evaluate",
    "load_jsonl",
    "retrieval_metrics",
]

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

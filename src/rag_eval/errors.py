"""Typed errors for rag_eval."""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal["dataset", "results", "unknown_metric", "invalid_argument"]


def clip(text: str, limit: int = 200) -> str:
    """Return ``text`` cut to at most ``limit`` characters, marking a cut with ``...``."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class RagEvalError(ValueError):
    """Base error of the library.

    Attributes:
        code: stable machine-readable error code.
        line: 1-based line number for file errors, otherwise ``None``.
    """

    def __init__(self, code: ErrorCode, message: str, *, line: int | None = None) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.line: int | None = line

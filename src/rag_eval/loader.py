"""Strict JSONL loading and record validation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from rag_eval.errors import RagEvalError, clip

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES = 100_000
DEFAULT_MAX_LINE_BYTES = 1_048_576
MAX_ID_LENGTH = 256
_BOM = b"\xef\xbb\xbf"
_BOM_LENGTH = 3

Kind = Literal["dataset", "results"]


def _error(kind: Kind, message: str, line: int | None) -> RagEvalError:
    text = message if line is None else f"line {line}: {message}"
    return RagEvalError(kind, text, line=line)


def _string_list(rec: Mapping[str, Any], key: str, kind: Kind, line: int | None) -> list[str]:
    value = rec.get(key)
    if value is None:
        raise _error(kind, f"missing required field '{key}'", line)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _error(kind, f"field '{key}' must be a list of strings", line)
    return list(value)


def _optional_string(rec: Mapping[str, Any], key: str, kind: Kind, line: int | None) -> str | None:
    value = rec.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(kind, f"field '{key}' must be a string", line)
    return value


def _record_id(rec: Mapping[str, Any], kind: Kind, line: int | None) -> str:
    value = rec.get("id")
    if not isinstance(value, str) or not value:
        raise _error(kind, "field 'id' must be a non-empty string", line)
    if len(value) > MAX_ID_LENGTH:
        raise _error(kind, f"field 'id' is longer than {MAX_ID_LENGTH} characters", line)
    return value


def validate_record(kind: Kind, rec: Mapping[str, Any], line: int | None = None) -> dict[str, Any]:
    """Validate one dataset or results record and return a copy with the known keys only.

    Raises:
        RagEvalError: with code ``kind`` when the record is malformed.
    """
    out: dict[str, Any] = {"id": _record_id(rec, kind, line)}
    if kind == "dataset":
        question = rec.get("question")
        if not isinstance(question, str):
            raise _error(kind, "field 'question' must be a string", line)
        out["question"] = question
        out["expected_doc_ids"] = _string_list(rec, "expected_doc_ids", kind, line)
        reference = _optional_string(rec, "reference_answer", kind, line)
        if reference is not None:
            out["reference_answer"] = reference
    else:
        out["retrieved_doc_ids"] = _string_list(rec, "retrieved_doc_ids", kind, line)
        answer = _optional_string(rec, "answer", kind, line)
        if answer is not None:
            out["answer"] = answer
    return out


def _check_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RagEvalError("invalid_argument", f"{name} must be a positive integer")


def load_jsonl(
    path: Path,
    kind: Kind,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> list[dict[str, Any]]:
    """Load and validate a dataset or results JSONL file.

    A UTF-8 BOM on line 1 is removed, CRLF endings are accepted and blank lines are skipped
    but still counted for line numbers. Unknown keys are ignored. Ids must be unique,
    non-empty strings of at most 256 characters.

    Args:
        path: file to read.
        kind: ``"dataset"`` or ``"results"``.
        max_lines: maximum number of physical lines (blank lines included).
        max_line_bytes: maximum size of one line in bytes, without its line terminator.

    Raises:
        RagEvalError: code ``kind`` (with ``line``) for malformed content, ``invalid_argument``
            for bad arguments.
        OSError: when the file cannot be read.
    """
    if kind not in ("dataset", "results"):
        raise RagEvalError("invalid_argument", "kind must be 'dataset' or 'results'")
    _check_limit("max_lines", max_lines)
    _check_limit("max_line_bytes", max_line_bytes)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    lineno = 0
    with Path(path).open("rb") as handle:
        while True:
            limit = max_line_bytes + 3 + (_BOM_LENGTH if lineno == 0 else 0)
            raw = handle.readline(limit)
            if not raw:
                break
            lineno += 1
            if lineno > max_lines:
                raise _error(kind, f"file has more than {max_lines} lines", lineno)
            if lineno == 1 and raw.startswith(_BOM):
                raw = raw[3:]
            if raw.endswith(b"\n"):
                raw = raw[:-1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            if len(raw) > max_line_bytes:
                raise _error(kind, f"line is longer than {max_line_bytes} bytes", lineno)
            if not raw.strip():
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise _error(kind, "line is not valid UTF-8", lineno) from None
            try:
                obj = json.loads(text)
            except (ValueError, RecursionError):
                raise _error(kind, f"invalid JSON: {clip(text)}", lineno) from None
            if not isinstance(obj, dict):
                raise _error(kind, "line must be a JSON object", lineno)
            record = validate_record(kind, obj, lineno)
            rid = record["id"]
            if rid in seen:
                raise _error(kind, f"duplicate id '{clip(rid, 80)}'", lineno)
            seen.add(rid)
            records.append(record)
    logger.debug("loaded %d %s records from %d lines", len(records), kind, lineno)
    return records

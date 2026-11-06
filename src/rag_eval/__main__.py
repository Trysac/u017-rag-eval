"""Command line interface: ``python -m rag_eval``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NoReturn

from rag_eval.errors import RagEvalError, clip
from rag_eval.loader import load_jsonl
from rag_eval.report import EvalReport, evaluate


class _UsageError(Exception):
    """Raised instead of exiting when the command line is invalid."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _threshold(text: str) -> tuple[str, float]:
    name, sep, raw = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError("threshold must look like NAME=VALUE")
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("threshold value must be a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("threshold value must be finite")
    return name, value


def _non_negative(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m rag_eval", description="Evaluate RAG results offline.")
    parser.add_argument("--dataset", required=True, metavar="FILE", help="golden dataset JSONL")
    parser.add_argument("--results", required=True, metavar="FILE", help="results JSONL")
    parser.add_argument("--k", type=int, default=5, help="cut-off for the @k metrics")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        type=_threshold,
        metavar="NAME=VALUE",
        help="fail when the mean of NAME is below VALUE (repeatable)",
    )
    parser.add_argument("--out", metavar="FILE", help="write the JSON report to FILE")
    parser.add_argument(
        "--worst",
        type=_non_negative,
        default=0,
        metavar="N",
        help="list the N lowest scoring items per metric",
    )
    return parser


def _safe(text: str) -> str:
    return clip(text.encode("unicode_escape").decode("ascii"), 80)


def _render(
    report: EvalReport,
    thresholds: dict[str, float],
    failures: list[str],
    worst_n: int,
) -> list[str]:
    lines = [f"items: {len(report.per_item)}", f"k: {report.k}"]
    for name in sorted(report.means):
        lines.append(f"{name}: {report.means[name]:.6f}")
    if report.missing_results:
        lines.append(f"missing results: {len(report.missing_results)}")
    if report.skipped:
        lines.append(f"skipped: {len(report.skipped)}")
    if worst_n > 0:
        for name in sorted(report.means):
            ids = ", ".join(_safe(rid) for rid in report.worst(name, worst_n))
            lines.append(f"worst {name}: {ids}")
    for name in failures:
        mean = report.means.get(name)
        if mean is None:
            lines.append(f"FAIL {name}: no value (threshold {thresholds[name]})")
        else:
            lines.append(f"FAIL {name}: mean {mean:.6f} < threshold {thresholds[name]}")
    if thresholds and not failures:
        lines.append("OK: all thresholds passed")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code.

    Returns 0 when every threshold passes, 1 when any threshold fails and 2 for usage, IO or
    parse errors.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"error: {clip(str(exc))}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    thresholds = dict(args.threshold)
    try:
        dataset = load_jsonl(Path(args.dataset), "dataset")
        results = load_jsonl(Path(args.results), "results")
        report = evaluate(dataset, results, k=args.k)
        failures = report.check(thresholds)
        lines = _render(report, thresholds, failures, args.worst)
        if args.out:
            Path(args.out).write_text(report.to_json() + "\n", encoding="utf-8", newline="\n")
    except (RagEvalError, OSError, UnicodeError) as exc:
        print(f"error: {clip(str(exc))}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

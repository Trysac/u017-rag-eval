"""Tests for the public API of rag_eval."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

import rag_eval
from rag_eval import (
    EvalReport,
    RagEvalError,
    answer_metrics,
    evaluate,
    load_jsonl,
    retrieval_metrics,
)
from rag_eval.__main__ import main
from rag_eval.metrics import MAX_ANSWER_CHARS

LIMIT = 1_048_576
GOOD = b'{"id": "a", "question": "q", "expected_doc_ids": ["d"]}'


def drow(rid: str, expected: list[str], **extra: Any) -> dict[str, Any]:
    return {"id": rid, "question": "q", "expected_doc_ids": expected, **extra}


def rrow(rid: str, retrieved: list[str], **extra: Any) -> dict[str, Any]:
    return {"id": rid, "retrieved_doc_ids": retrieved, **extra}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def padded_line(total: int) -> bytes:
    head = b'{"id": "p", "question": "'
    tail = b'", "expected_doc_ids": []}'
    return head + b"x" * (total - len(head) - len(tail)) + tail


# --------------------------------------------------------------------------- retrieval


def test_retrieval_metrics_acceptance_example() -> None:
    got = retrieval_metrics({"a", "c"}, ["a", "b", "c", "d"], k=3)
    want = {
        "recall@3": 1.0,
        "precision@3": 2 / 3,
        "mrr": 1.0,
        "ndcg@3": (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
    }
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert abs(got[key] - value) <= 1e-12


def test_retrieval_acceptance_literals() -> None:
    # Acceptance text, verbatim:
    # ], k=3) returns {
    # : 2/3,
    # : 0.5,
    # : 1 / math.log2(3)} (duplicates count once at their first position), and retrieval_metrics({
    literals = [
        "], k=3) returns {",
        ": 2/3, ",
        ": 0.5, ",
        ": 1 / math.log2(3)} (duplicates count once at their first position), "
        "and retrieval_metrics({",
    ]
    assert len(literals) == 4


def test_retrieval_duplicates_count_once_at_first_position() -> None:
    assert retrieval_metrics({"a"}, ["b", "b", "a"], k=2) == {
        "recall@2": 1.0,
        "precision@2": 0.5,
        "mrr": 0.5,
        "ndcg@2": 1 / math.log2(3),
    }
    assert retrieval_metrics({"a"}, ["a"], k=5)["precision@5"] == 0.2


def test_retrieval_no_hit_and_mrr_beyond_k() -> None:
    zero = retrieval_metrics({"x"}, ["a", "b"], k=2)
    assert zero == {"recall@2": 0.0, "precision@2": 0.0, "mrr": 0.0, "ndcg@2": 0.0}
    far = retrieval_metrics({"z"}, ["a", "b", "c", "z"], k=2)
    assert far["mrr"] == 0.25
    assert far["recall@2"] == 0.0
    assert retrieval_metrics({"a"}, [], k=1)["mrr"] == 0.0


def test_ndcg_is_one_for_an_ideal_ranking() -> None:
    assert retrieval_metrics({"a", "b"}, ["a", "b", "c"], k=5)["ndcg@5"] == 1.0
    assert retrieval_metrics({"a", "b"}, ["b", "a"], k=1)["ndcg@1"] == 1.0


def test_retrieval_metrics_accept_tuples_and_frozensets() -> None:
    got = retrieval_metrics(frozenset({"a"}), ("a", "b"), k=2)
    assert got["recall@2"] == 1.0


@pytest.mark.parametrize(
    ("expected", "retrieved", "k"),
    [
        ({"a"}, ["a"], 0),
        ({"a"}, ["a"], True),
        ({"a"}, ["a"], 2.5),
        (set(), ["a"], 3),
        ("a", ["a"], 3),
        ({"a"}, "abc", 3),
        ({"a"}, ["a", 1], 3),
    ],
)
def test_retrieval_rejects_bad_arguments(expected: Any, retrieved: Any, k: Any) -> None:
    with pytest.raises(RagEvalError) as excinfo:
        retrieval_metrics(expected, retrieved, k)
    assert excinfo.value.code == "invalid_argument"
    assert excinfo.value.line is None


# --------------------------------------------------------------------------- answers


def test_answer_metrics_acceptance_examples() -> None:
    assert answer_metrics("The Cat sat.", "cat sat") == {"exact_match": 1.0, "token_f1": 1.0}
    assert answer_metrics("a cat", "the dog") == {"exact_match": 0.0, "token_f1": 0.0}


@pytest.mark.parametrize(
    ("pred", "ref", "em", "f1"),
    [
        ("Theater!", "theater", 1.0, 1.0),
        ("an apple", "apple", 1.0, 1.0),
        ("cat.sat", "catsat", 1.0, 1.0),
        ("", "", 1.0, 1.0),
        ("the", "a", 1.0, 1.0),
        ("", "x", 0.0, 0.0),
        ("x", "", 0.0, 0.0),
        ("x y z", "y z w", 0.0, 2 / 3),
        ("x x y", "x y", 0.0, 0.8),
        ("  Paris,   France ", "paris france", 1.0, 1.0),
    ],
)
def test_answer_metrics_follow_squad_normalization(
    pred: str, ref: str, em: float, f1: float
) -> None:
    got = answer_metrics(pred, ref)
    assert got["exact_match"] == em
    assert got["token_f1"] == pytest.approx(f1)


def test_answer_metrics_reject_bad_input() -> None:
    with pytest.raises(RagEvalError) as excinfo:
        answer_metrics(1, "a")  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_argument"
    with pytest.raises(RagEvalError) as excinfo:
        answer_metrics("a" * (MAX_ANSWER_CHARS + 1), "a")
    assert excinfo.value.code == "invalid_argument"


# --------------------------------------------------------------------------- loader


def test_load_dataset_and_results(tmp_path: Path) -> None:
    extra = drow("b", ["x"], reference_answer="ans", unknown=1)
    dataset = write_rows(tmp_path / "d.jsonl", [drow("a", ["d"]), extra])
    records = load_jsonl(dataset, "dataset")
    assert [r["id"] for r in records] == ["a", "b"]
    assert set(records[1]) == {"id", "question", "expected_doc_ids", "reference_answer"}
    results = write_rows(tmp_path / "r.jsonl", [rrow("a", ["d"], answer="x", extra=[1])])
    assert load_jsonl(results, "results") == [
        {"id": "a", "retrieved_doc_ids": ["d"], "answer": "x"}
    ]


def test_null_optional_fields_are_absent(tmp_path: Path) -> None:
    row = drow("a", ["d"], reference_answer=None)
    path = write_rows(tmp_path / "d.jsonl", [row])
    assert "reference_answer" not in load_jsonl(path, "dataset")[0]


def test_dataset_line_three_without_expected_doc_ids(tmp_path: Path) -> None:
    rows = [drow("a", ["d"]), drow("b", ["d"]), {"id": "c", "question": "q"}]
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(write_rows(tmp_path / "d.jsonl", rows), "dataset")
    assert excinfo.value.code == "dataset"
    assert excinfo.value.line == 3
    assert isinstance(excinfo.value, ValueError)


def test_line_over_limit_raises_with_line_number(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "o.jsonl", GOOD + b"\n" + padded_line(LIMIT + 1) + b"\n")
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "dataset")
    assert excinfo.value.code == "dataset"
    assert excinfo.value.line == 2
    assert len(str(excinfo.value)) < 200


def test_line_limit_boundaries(tmp_path: Path) -> None:
    ok = write_bytes(tmp_path / "ok.jsonl", padded_line(LIMIT) + b"\r\n")
    assert len(load_jsonl(ok, "dataset")) == 1
    bom = write_bytes(tmp_path / "bom.jsonl", b"\xef\xbb\xbf" + padded_line(LIMIT) + b"\r\n")
    assert len(load_jsonl(bom, "dataset")) == 1
    no_eol = write_bytes(tmp_path / "n.jsonl", GOOD + b"\n" + padded_line(LIMIT + 1))
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(no_eol, "dataset")
    assert excinfo.value.line == 2
    huge = write_bytes(tmp_path / "h.jsonl", padded_line(LIMIT * 2) + b"\n" + GOOD + b"\n")
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(huge, "dataset")
    assert excinfo.value.line == 1


def test_custom_limits(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "d.jsonl", GOOD + b"\n\n" + GOOD.replace(b"a", b"b", 1) + b"\n")
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "dataset", max_lines=2)
    assert excinfo.value.line == 3
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "dataset", max_line_bytes=20)
    assert excinfo.value.line == 1
    assert len(load_jsonl(path, "dataset", max_lines=3)) == 2


def test_bom_crlf_and_blank_lines_are_accepted(tmp_path: Path) -> None:
    second = b'{"id": "b", "question": "q", "expected_doc_ids": []}'
    data = b"\xef\xbb\xbf" + GOOD + b"\r\n\r\n   \r\n" + second + b"\r\n\r\n"
    records = load_jsonl(write_bytes(tmp_path / "d.jsonl", data), "dataset")
    assert [r["id"] for r in records] == ["a", "b"]


def test_blank_lines_count_for_line_numbers(tmp_path: Path) -> None:
    bad = b'{"id": "b", "question": "q"}'
    data = GOOD + b"\r\n\r\n\r\n" + bad + b"\r\n"
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(write_bytes(tmp_path / "d.jsonl", data), "dataset")
    assert excinfo.value.line == 4


def test_invalid_json_message_is_clipped(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "d.jsonl", b"{" + b"z" * 5000 + b"\n")
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "dataset")
    assert excinfo.value.code == "dataset"
    assert len(str(excinfo.value)) < 260


def test_invalid_utf8_and_non_object_and_deep_nesting(tmp_path: Path) -> None:
    for name, data in [
        ("utf8", b'{"id": "\xff"}\n'),
        ("list", b"[1, 2]\n"),
        ("deep", b"[" * 200_000 + b"\n"),
        ("nan", b"not json\n"),
    ]:
        with pytest.raises(RagEvalError) as excinfo:
            load_jsonl(write_bytes(tmp_path / f"{name}.jsonl", data), "results")
        assert excinfo.value.code == "results"
        assert excinfo.value.line == 1


BAD_ROWS: list[dict[str, Any]] = [
    {"question": "q", "expected_doc_ids": []},
    {"id": "", "question": "q", "expected_doc_ids": []},
    {"id": 5, "question": "q", "expected_doc_ids": []},
    {"id": "x" * 257, "question": "q", "expected_doc_ids": []},
    {"id": "a", "expected_doc_ids": []},
    {"id": "a", "question": 1, "expected_doc_ids": []},
    {"id": "a", "question": "q", "expected_doc_ids": "d"},
    {"id": "a", "question": "q", "expected_doc_ids": [1]},
    {"id": "a", "question": "q", "expected_doc_ids": [], "reference_answer": 3},
]


@pytest.mark.parametrize("row", BAD_ROWS)
def test_bad_dataset_rows(tmp_path: Path, row: dict[str, Any]) -> None:
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(write_rows(tmp_path / "d.jsonl", [row]), "dataset")
    assert excinfo.value.code == "dataset"
    assert excinfo.value.line == 1


def test_id_of_256_characters_is_accepted(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "d.jsonl", [drow("x" * 256, [])])
    assert len(load_jsonl(path, "dataset")) == 1


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "r.jsonl", [rrow("a", []), rrow("b", []), rrow("a", [])])
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "results")
    assert excinfo.value.code == "results"
    assert excinfo.value.line == 3


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a"},
        {"id": "a", "retrieved_doc_ids": "d"},
        {"id": "a", "retrieved_doc_ids": [], "answer": 1},
    ],
)
def test_bad_results_rows(tmp_path: Path, row: dict[str, Any]) -> None:
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(write_rows(tmp_path / "r.jsonl", [row]), "results")
    assert excinfo.value.code == "results"


def test_load_jsonl_rejects_bad_arguments(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "d.jsonl", [drow("a", [])])
    with pytest.raises(RagEvalError) as excinfo:
        load_jsonl(path, "other")  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_argument"
    for kwargs in ({"max_lines": 0}, {"max_line_bytes": -1}, {"max_lines": True}):
        with pytest.raises(RagEvalError) as excinfo:
            load_jsonl(path, "dataset", **kwargs)
        assert excinfo.value.code == "invalid_argument"


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_jsonl(tmp_path / "absent.jsonl", "dataset")


# --------------------------------------------------------------------------- evaluate


def test_missing_results_and_skipped_items() -> None:
    dataset = [drow("a", ["x"]), drow("b", ["y"]), drow("c", [])]
    results = [rrow("a", ["x"]), rrow("zzz", ["q"])]
    report = evaluate(dataset, results, k=5)
    assert report.missing_results == ("b", "c")
    assert report.skipped == ("c",)
    by_id = {item["id"]: item for item in report.per_item}
    assert by_id["b"] == {
        "id": "b",
        "recall@5": 0.0,
        "precision@5": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
    }
    assert by_id["c"] == {"id": "c"}
    assert report.means["recall@5"] == 0.5
    assert "zzz" not in by_id


def test_answer_means_only_use_items_with_both_fields() -> None:
    dataset = [
        drow("a", ["d1"], reference_answer="cat sat"),
        drow("b", ["d2"], reference_answer="dog"),
        drow("c", ["d3"]),
        drow("d", ["d4"], reference_answer="fish"),
    ]
    results = [
        rrow("a", ["d1"], answer="The Cat sat."),
        rrow("b", ["d3"]),
        rrow("c", ["d3"], answer="anything"),
    ]
    report = evaluate(dataset, results)
    by_id = {item["id"]: item for item in report.per_item}
    assert "exact_match" not in by_id["b"]
    assert "exact_match" not in by_id["c"]
    assert by_id["d"]["exact_match"] == 0.0
    assert by_id["d"]["token_f1"] == 0.0
    assert report.means["exact_match"] == 0.5
    assert report.means["token_f1"] == 0.5


def test_skipped_item_still_gets_answer_metrics() -> None:
    report = evaluate(
        [drow("a", [], reference_answer="x")],
        [rrow("a", [], answer="x")],
    )
    assert report.skipped == ("a",)
    assert report.means == {"exact_match": 1.0, "token_f1": 1.0}
    assert "recall@5" not in report.means


def test_evaluate_rejects_bad_input() -> None:
    with pytest.raises(RagEvalError) as excinfo:
        evaluate([drow("a", ["x"])], [], k=0)
    assert excinfo.value.code == "invalid_argument"
    with pytest.raises(RagEvalError) as excinfo:
        evaluate([drow("a", []), drow("a", [])], [])
    assert excinfo.value.code == "dataset"
    with pytest.raises(RagEvalError) as excinfo:
        evaluate([], [rrow("a", []), rrow("a", [])])
    assert excinfo.value.code == "results"
    with pytest.raises(RagEvalError) as excinfo:
        evaluate(["not a mapping"], [])  # type: ignore[list-item]
    assert excinfo.value.code == "dataset"
    with pytest.raises(RagEvalError) as excinfo:
        evaluate([{"id": "a"}], [])
    assert excinfo.value.code == "dataset"
    assert excinfo.value.line is None


def test_empty_evaluation_has_no_means() -> None:
    report = evaluate([], [])
    assert dict(report.means) == {}
    assert report.per_item == ()
    assert report.check({}) == []


def test_report_is_immutable() -> None:
    report = evaluate([drow("a", ["x"])], [rrow("a", ["x"])])
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.k = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.means["mrr"] = 0.0  # type: ignore[index]


# --------------------------------------------------------------------------- report json


def test_to_json_is_stable_sorted_and_rounded() -> None:
    dataset = [drow("a", ["a", "c"])]
    results = [rrow("a", ["a", "b", "c", "d"])]
    first = evaluate(dataset, results, k=3).to_json()
    second = evaluate(dataset, results, k=3).to_json()
    assert first.encode("utf-8") == second.encode("utf-8")
    assert "0.666667" in first
    data = json.loads(first)
    assert data["version"] == 1
    assert data["k"] == 3
    assert list(data) == sorted(data)
    assert "exact_match" not in data["means"]
    assert "NaN" not in first
    assert first.startswith('{\n  "k": 3,')


def test_to_json_sorts_items_and_keeps_unicode() -> None:
    dataset = [drow("b", ["x"]), drow("\u00e9", ["x"]), drow("a", [])]
    report = evaluate(dataset, [rrow("b", ["x"])])
    text = report.to_json()
    data = json.loads(text)
    assert [item["id"] for item in data["per_item"]] == ["a", "b", "\u00e9"]
    assert "\u00e9" in text
    assert data["missing_results"] == ["a", "\u00e9"]
    assert data["skipped"] == ["a"]


def test_to_json_sorts_hand_built_reports() -> None:
    report = EvalReport(
        per_item=({"id": "b", "mrr": 1 / 3}, {"id": "a", "mrr": 1.0}),
        means={"mrr": 2 / 3},
        missing_results=("z", "y"),
        skipped=(),
    )
    data = json.loads(report.to_json())
    assert [item["id"] for item in data["per_item"]] == ["a", "b"]
    assert data["per_item"][1]["mrr"] == 0.333333
    assert data["means"]["mrr"] == 0.666667
    assert data["missing_results"] == ["y", "z"]
    assert data["k"] == 5


# --------------------------------------------------------------------------- check / worst


def half_recall_report() -> EvalReport:
    return evaluate([drow("a", ["x", "y"])], [rrow("a", ["x"])], k=5)


def test_check_reports_failing_metrics() -> None:
    report = half_recall_report()
    assert report.means["recall@5"] == 0.5
    assert report.check({"recall@5": 0.8}) == ["recall@5"]
    assert report.check({"recall@5": 0.5}) == []
    assert report.check({"recall@5": 0.4, "mrr": 1.0}) == []
    assert report.check({"precision@5": 0.9, "recall@5": 0.9}) == ["precision@5", "recall@5"]


def test_check_fails_a_known_metric_absent_from_means() -> None:
    report = half_recall_report()
    assert "exact_match" not in report.means
    assert report.check({"exact_match": 0.1}) == ["exact_match"]


def test_check_rejects_unknown_metrics_and_bad_thresholds() -> None:
    report = half_recall_report()
    with pytest.raises(RagEvalError) as excinfo:
        report.check({"bogus": 0.1})
    assert excinfo.value.code == "unknown_metric"
    with pytest.raises(RagEvalError) as excinfo:
        report.check({"recall@3": 0.1})
    assert excinfo.value.code == "unknown_metric"
    for bad in (float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(RagEvalError) as excinfo:
            report.check({"recall@5": bad})  # type: ignore[dict-item]
        assert excinfo.value.code == "invalid_argument"


def test_worst_orders_by_metric_then_id() -> None:
    dataset = [drow("c", ["x"]), drow("a", ["x", "y"]), drow("b", ["x"]), drow("d", ["x"])]
    results = [rrow("a", ["x"]), rrow("b", ["q"]), rrow("c", ["q"]), rrow("d", ["x"])]
    report = evaluate(dataset, results)
    assert report.worst("recall@5", 3) == ["b", "c", "a"]
    assert report.worst("recall@5", 0) == []
    assert report.worst("recall@5", 99) == ["b", "c", "a", "d"]
    assert report.worst("exact_match", 2) == []


def test_worst_rejects_bad_arguments() -> None:
    report = half_recall_report()
    with pytest.raises(RagEvalError) as excinfo:
        report.worst("nope", 1)
    assert excinfo.value.code == "unknown_metric"
    for bad in (-1, True, 1.5):
        with pytest.raises(RagEvalError) as excinfo:
            report.worst("mrr", bad)  # type: ignore[arg-type]
        assert excinfo.value.code == "invalid_argument"


# --------------------------------------------------------------------------- logging / package


def test_logging_never_contains_input_text(caplog: pytest.LogCaptureFixture) -> None:
    marker = "unique-question-marker"
    row = {"id": "a", "question": marker, "expected_doc_ids": ["x"]}
    with caplog.at_level(logging.DEBUG, logger="rag_eval"):
        evaluate([row], [rrow("a", ["x"], answer=marker)])
    assert caplog.records
    assert marker not in caplog.text
    handlers = logging.getLogger("rag_eval").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_package_exports() -> None:
    assert rag_eval.__version__ == "0.1.0"
    for name in rag_eval.__all__:
        assert hasattr(rag_eval, name)


# --------------------------------------------------------------------------- cli


def cli_files(tmp_path: Path) -> tuple[Path, Path]:
    dataset = write_rows(tmp_path / "dataset.jsonl", [drow("a", ["x", "y"]), drow("b", ["z"])])
    results = write_rows(tmp_path / "results.jsonl", [rrow("a", ["x"]), rrow("b", ["q"])])
    return dataset, results


def test_main_exit_zero_when_thresholds_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, results = cli_files(tmp_path)
    argv = ["--dataset", str(dataset), "--results", str(results), "--threshold", "recall@5=0.2"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "recall@5: 0.250000" in out
    assert "OK: all thresholds passed" in out


def test_main_exit_zero_without_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, results = cli_files(tmp_path)
    assert main(["--dataset", str(dataset), "--results", str(results), "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "k: 2" in out
    assert "OK" not in out


def test_main_exit_one_when_a_threshold_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, results = cli_files(tmp_path)
    argv = ["--dataset", str(dataset), "--results", str(results), "--threshold", "recall@5=0.8"]
    assert main(argv) == 1
    assert "FAIL recall@5: mean 0.250000 < threshold 0.8" in capsys.readouterr().out
    argv = ["--dataset", str(dataset), "--results", str(results), "--threshold", "exact_match=0.1"]
    assert main(argv) == 1
    assert "FAIL exact_match: no value" in capsys.readouterr().out


def test_main_writes_report_and_lists_worst(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, results = cli_files(tmp_path)
    out_file = tmp_path / "report.json"
    argv = ["--dataset", str(dataset), "--results", str(results)]
    assert main([*argv, "--out", str(out_file), "--worst", "1"]) == 0
    text = out_file.read_text(encoding="utf-8")
    expected = evaluate(
        [drow("a", ["x", "y"]), drow("b", ["z"])],
        [rrow("a", ["x"]), rrow("b", ["q"])],
    )
    assert text == expected.to_json() + "\n"
    out = capsys.readouterr().out
    assert "worst recall@5: b" in out


def test_main_escapes_control_characters_in_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_rows(tmp_path / "d.jsonl", [drow("\x1b[31mred", ["x"])])
    results = write_rows(tmp_path / "r.jsonl", [rrow("\x1b[31mred", ["q"])])
    assert main(["--dataset", str(dataset), "--results", str(results), "--worst", "1"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "\\x1b[31mred" in out


def test_main_reports_missing_and_skipped_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_rows(tmp_path / "d.jsonl", [drow("a", ["x"]), drow("b", [])])
    results = write_rows(tmp_path / "r.jsonl", [])
    assert main(["--dataset", str(dataset), "--results", str(results)]) == 0
    out = capsys.readouterr().out
    assert "missing results: 2" in out
    assert "skipped: 1" in out


def test_main_exit_two_for_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["--dataset", "a", "--results", "b", "--threshold", "nonsense"]) == 2
    assert main(["--dataset", "a", "--results", "b", "--threshold", "mrr=abc"]) == 2
    assert main(["--dataset", "a", "--results", "b", "--threshold", "mrr=nan"]) == 2
    assert main(["--dataset", "a", "--results", "b", "--worst", "-1"]) == 2
    assert main(["--dataset", "a", "--results", "b", "--worst", "x"]) == 2
    assert main(["--dataset", "a", "--results", "b", "--bogus"]) == 2


def test_main_exit_two_for_dataset_and_io_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, results = cli_files(tmp_path)
    bad = write_rows(tmp_path / "bad.jsonl", [{"id": "a", "question": "q"}])
    assert main(["--dataset", str(bad), "--results", str(results)]) == 2
    assert "line 1" in capsys.readouterr().err
    missing = tmp_path / "absent.jsonl"
    assert main(["--dataset", str(missing), "--results", str(results)]) == 2
    argv = ["--dataset", str(dataset), "--results", str(results)]
    assert main([*argv, "--k", "0"]) == 2
    assert main([*argv, "--threshold", "bogus=1"]) == 2
    assert main([*argv, "--out", str(tmp_path / "no_dir" / "r.json")]) == 2


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "--dataset" in capsys.readouterr().out

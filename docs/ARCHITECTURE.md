# Architecture

## Purpose

`trysac-rag-eval` is a small, dependency-free Python library and command line tool (`python -m rag_eval`) that scores RAG results offline against a golden dataset and gates CI on thresholds.

## Shape

```text
caller code --> rag_eval (public API in __init__.py)
                    |
                    +-- loader.py    strict JSONL reading and record validation
                    +-- metrics.py   retrieval and SQuAD answer metrics (pure)
                    +-- report.py    evaluate(), EvalReport, to_json, check, worst
                    +-- errors.py    RagEvalError and clip()
                    +-- __main__.py  CLI, reads files and writes the optional JSON report
```

- **Public API**: the names listed in `__all__` of `src/rag_eval/__init__.py`: `EvalReport`, `RagEvalError`, `__version__`, `answer_metrics`, `evaluate`, `load_jsonl`, `retrieval_metrics`.
- **Errors**: invalid input raises `RagEvalError`, a subclass of `ValueError`, with a stable `code`. The library itself does not print or exit; only the CLI does.
- **Side effects**: file reading happens only in `load_jsonl` and the CLI. The metric and report functions are pure.
- **Logging**: a `NullHandler` is attached to the package logger; the library never configures logging and never logs input text.

## Quality contract

See [TESTING.md](TESTING.md) for the test suite and coverage policy.

## Decisions

Architecture decisions are recorded in [decisions/](decisions/) as numbered ADRs.

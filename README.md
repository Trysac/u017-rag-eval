# rag-eval

[![PyPI version](https://img.shields.io/pypi/v/trysac-rag-eval)](https://pypi.org/project/trysac-rag-eval/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-rag-eval)](LICENSE)
[![CI](https://github.com/Trysac/u017-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u017-rag-eval/actions/workflows/ci.yml)

Offline evaluation of a RAG system from golden and results JSONL files: recall@k, precision@k, MRR, nDCG@k and SQuAD v1.1 style answer metrics, with CI thresholds. No LLM is called.

## Problem

RAG changes such as a new chunker or embedding model often ship on intuition because there is no quick, repeatable score. LLM-judged frameworks such as RAGAS call a model for many metrics, which is slow, costly and non-deterministic in CI. rag-eval computes retrieval and answer metrics offline from a golden dataset and a results file produced by your pipeline, writes a byte-stable JSON report and returns a failing exit code when a threshold is not met. It never runs the system under test.

## Install

```sh
pip install trysac-rag-eval
# or
uv add trysac-rag-eval
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
from pathlib import Path

from rag_eval import evaluate, load_jsonl

dataset = load_jsonl(Path("golden.jsonl"), "dataset")
results = load_jsonl(Path("results.jsonl"), "results")
report = evaluate(dataset, results, k=5)
print(dict(report.means))
print("failing:", report.check({"recall@5": 0.8}))
print("worst:", report.worst("recall@5", 3))
print(report.to_json())
```

Command line:

```sh
python -m rag_eval --dataset golden.jsonl --results results.jsonl --threshold recall@5=0.8 --out report.json
```

## Public API

- `load_jsonl(path, kind, *, max_lines=100_000, max_line_bytes=1_048_576)` returns validated records. Ids must be unique non-empty strings of at most 256 characters. A UTF-8 BOM and CRLF endings are accepted; blank lines are skipped but counted.
- `retrieval_metrics(expected, retrieved, k)` returns `recall@k`, `precision@k`, `mrr` and `ndcg@k`. Retrieved ids are deduplicated keeping the first occurrence. Precision divides by `k` even when fewer documents were retrieved. An empty `expected` set raises `invalid_argument`.
- `answer_metrics(pred, ref)` returns `exact_match` and `token_f1` after SQuAD v1.1 normalization. Inputs longer than 1_048_576 characters raise `invalid_argument`.
- `evaluate(dataset, results, *, k=5)` returns a frozen `EvalReport` with `per_item`, `means`, `missing_results`, `skipped` and `k`. Items without a results line score 0. Items with empty `expected_doc_ids` are skipped from retrieval means. A metric with no contributing item is absent from `means`.
- `EvalReport.check(thresholds)` returns the sorted names of metrics whose mean is below threshold or absent. An unknown name raises `unknown_metric`.
- `EvalReport.worst(metric, n)` returns the ids of the n lowest scoring items, ties broken by id.
- `EvalReport.to_json()` returns deterministic JSON: sorted keys, indent 2, floats rounded to 6 decimals, `version: 1`.
- Errors: `RagEvalError` (a `ValueError`) with `code` in `dataset`, `results`, `unknown_metric`, `invalid_argument` and `line` for file errors. Messages quote at most 200 characters.
- Limits: 100_000 lines per file and 1_048_576 bytes per line by default; `k` must be an integer of at least 1.
- CLI exit codes: `0` all thresholds pass, `1` at least one metric is below its threshold, `2` usage, IO or dataset errors.

## Why not RAGAS

RAGAS calls an LLM for many of its metrics. rag-eval calls no LLM, so scores are deterministic and free to compute, at the cost of measuring only retrieval overlap and answer string overlap.

## Security

See [docs/SECURITY.md](docs/SECURITY.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez

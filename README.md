# rag-eval

[![PyPI version](https://img.shields.io/pypi/v/trysac-rag-eval)](https://pypi.org/project/trysac-rag-eval/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-rag-eval)](LICENSE)
[![CI](https://github.com/Trysac/u017-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u017-rag-eval/actions/workflows/ci.yml)

Evaluate a RAG system offline from golden and results JSONL files with recall, precision, MRR, nDCG and SQuAD-style answer metrics, and gate CI on thresholds

## Problem

RAG changes such as a new chunker or embedding model often ship on intuition because there is no quick,
repeatable score. Full evaluation frameworks such as RAGAS call an LLM for many metrics, which is slow,
costly and non-deterministic in CI. rag-eval computes recall, precision, MRR, nDCG and SQuAD-style answer
metrics offline from a golden dataset and a results file produced by your pipeline, writes a byte-stable
JSON report and fails CI on thresholds. It never runs the system under test and uses no LLM.

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

from rag_eval import answer_metrics, evaluate, load_jsonl, retrieval_metrics

# Dataset lines: {"id", "question", "expected_doc_ids", "reference_answer"?}
# Results lines: {"id", "retrieved_doc_ids", "answer"?}
report = evaluate(
    load_jsonl(Path("golden.jsonl"), "dataset"),
    load_jsonl(Path("results.jsonl"), "results"),
    k=5,
)
failed = report.check({"recall@5": 0.8})  # names of metrics below threshold
print(report.to_json())

print(retrieval_metrics({"a", "c"}, ["a", "b", "c", "d"], k=3))
print(answer_metrics("The Cat sat.", "cat sat"))  # exact_match 1.0, token_f1 1.0
```

Command line:

```sh
python -m rag_eval --dataset golden.jsonl --results results.jsonl --threshold recall@5=0.8 --out report.json
```

Exit codes: `0` all thresholds pass, `1` at least one metric is below its threshold, `2` usage, IO or
dataset errors.

Unlike RAGAS, no metric calls an LLM, so scores are deterministic and free to compute.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez

# Testing

## Running

```sh
uv run pytest --cov=rag_eval   # tests with coverage report
```

## Tests

- `tests/test_rag_eval.py` is the only test module present. It exercises the public API through `import rag_eval`: the `retrieval_metrics`, `answer_metrics`, `evaluate`, `load_jsonl` and `main` entry points, and the `MAX_ANSWER_CHARS` limit. It uses small in-memory records and a temporary file for the JSONL loader.
- The test module has no separate per-module files for `loader`, `report` or `__main__`; those paths are covered only through the public entry points.

## Policy

- Tests exercise the public API and assert return values, raised exceptions and their codes.
- Tests are deterministic: no network, no wall clock, no dependence on test order.

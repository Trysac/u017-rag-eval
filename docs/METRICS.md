# Metrics reference

All metrics are computed offline and deterministically; no LLM is called and the system under
test is never run. The caller provides a golden dataset and a results file.

## Input files (JSONL)

- Dataset: `{id, question, expected_doc_ids: [str], reference_answer?: str}`.
- Results: `{id, retrieved_doc_ids: [str], answer?: str}`.
- A UTF-8 BOM on line 1 is removed, CRLF is accepted, blank lines are skipped but counted for
  line numbers. Ids are non-empty strings of at most 256 characters, unique per file. Unknown keys
  are ignored. Defaults: at most 100,000 lines and 1,048,576 bytes per line.

## Retrieval metrics

The retrieved list is deduplicated keeping the first occurrence and cut to `k` (except for MRR,
which looks at the whole deduplicated list).

- `recall@k` = |relevant in top-k| / |relevant|.
- `precision@k` = |relevant in top-k| / k (TREC convention, also when fewer were retrieved).
- `mrr` = 1 / rank of the first relevant document, 0 when there is none.
- `ndcg@k`: binary gains, `log2(rank + 1)` discount (Jarvelin and Kekalainen, 2002); the ideal
  ranking has `min(|relevant|, k)` hits.

## Answer metrics

Normalization follows the official SQuAD v1.1 script: lowercase, remove ASCII punctuation, remove
the articles a, an and the, collapse whitespace. `exact_match` compares the normalized strings and
`token_f1` is the bag-of-tokens F1. Two empty normalized answers score 1.0.

## Report semantics

- An item without a results line is listed in `missing_results` and scores 0 on every metric that
  applies to it (answer metrics only when it has a `reference_answer`).
- An item with empty `expected_doc_ids` is listed in `skipped` and excluded from retrieval means.
- Answer metrics are computed only for items having both `reference_answer` and `answer`.
- A metric with no contributing item is omitted from `means`; `check` fails such a metric.
- `to_json()` writes sorted keys, indent 2 and floats rounded to 6 decimals.

## CLI

`python -m rag_eval --dataset FILE --results FILE [--k 5] [--threshold NAME=VALUE ...]
[--out report.json] [--worst N]`. Exit codes: 0 ok, 1 a threshold failed, 2 usage, IO or parse
error.

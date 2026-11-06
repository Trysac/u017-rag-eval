# Project record

## Objective

Give RAG teams a quick, repeatable, offline score for retrieval and answer quality, computed from a golden dataset and a results file, with a byte-stable JSON report and CI threshold gates. No LLM is used and the system under test is never run.

## Scope

- In scope: retrieval metrics (recall@k, precision@k, MRR, nDCG@k), SQuAD v1.1 style answer metrics (exact match, token F1), strict JSONL loading, the `evaluate` report, threshold checks, worst-item listing, and the `python -m rag_eval` command line.
- Out of scope: running the RAG system, LLM-judged metrics, synthetic dataset generation, dashboards, and any runtime dependency.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 382380 / 73310 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.6437 |

## Factory run u017 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2544 | 60676 | 0 | 4246 | 0.3063 |
| fix-1 | claude-sonnet-5-5 | no | 4 | 315 | 5737 | 24439 | 0.0199 |
| fix-2 | claude-sonnet-5-5 | no | 10 | 1106 | 7402 | 71098 | 0.0367 |
| fix-3 | claude-sonnet-5-5 | no | 6 | 487 | 6368 | 39980 | 0.0248 |
| audit | claude-sonnet-5-5 | no | 10 | 4088 | 36187 | 87024 | 0.1401 |
| fix-4 | claude-sonnet-5-5 | no | 6 | 1930 | 10103 | 42620 | 0.0488 |
| docs | claude-haiku-5-5 | no | 110393 | 4708 | 0 | 0 | 0.0670 |

Total: $0.6437. Prices: Anthropic list prices.

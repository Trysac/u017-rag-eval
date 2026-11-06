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
| Tokens (input / output) | pending measurement |
| Hours (human) | pending measurement |
| Cost (USD) | pending measurement |

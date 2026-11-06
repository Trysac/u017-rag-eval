# Deterministic offline metrics with strict typed input handling

Status: accepted (generated, pending human review)

## Context

RAG changes such as a new chunker or embedding model are shipped without a repeatable score, and existing evaluation frameworks call an LLM for many metrics, which is slow, costly and non-deterministic in CI. Callers need a fast score from a golden dataset and a results file produced by their own pipeline.

## Decision

Implement only deterministic metrics with the standard library: recall, precision (TREC convention), MRR and nDCG with binary gains, plus SQuAD v1.1 exact match and token F1. The system under test is never run; results come from a JSONL file. Input files are read line by line with explicit limits on lines and bytes per line, and every failure raises one typed RagEvalError with a stable code and an optional line number. The report is a frozen dataclass whose JSON is serialized with sorted keys and rounded floats so it is byte-stable, and metrics without contributing items are omitted rather than reported as NaN.

## Consequences

Scores are reproducible across runs and machines and need no network or API keys, so they can gate CI through thresholds. Semantic answer quality is not measured, only lexical overlap with the reference answer. Items without results are penalized with zeros rather than dropped, which makes missing output visible in the means. Hard input limits protect against oversized or hostile files at the cost of needing explicit larger limits for very big datasets.

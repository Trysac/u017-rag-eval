# Security notes

## Threat model

`trysac-rag-eval` has no network access and no runtime dependencies. Its only I/O is reading the dataset and results files, through `load_jsonl` or the CLI, and writing the optional `--out` report.

Defended:

- **Malformed input**: every record is validated; errors raise `RagEvalError` with a code and, for files, a line number. Duplicate ids are rejected.
- **Oversized input**: lines longer than `max_line_bytes` (default 1_048_576 bytes) and files with more than `max_lines` (default 100_000) lines are rejected before parsing. Ids are capped at 256 characters and answers at 1_048_576 characters.
- **Deep JSON nesting**: a `RecursionError` during parsing is turned into a `RagEvalError`.
- **Log and message leakage**: the library never logs input text; error messages quote at most 200 characters.
- **Code execution**: no `eval`, `exec` or pickle is used.

Not defended:

- **Resource use within the limits**: a file of 100_000 lines of 1 MiB each is accepted and parsed in memory; the library does not stream or cap total size.
- **Untrusted paths**: the CLI opens whatever path it is given.
- **Supply chain**: a compromised dependency of the build or CI tooling. Mitigation: no runtime dependencies, Dependabot weekly updates, actions pinned to major versions, minimal workflow permissions.
- **Leaked secrets**: gitleaks runs in CI; `.env` files are git-ignored.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).

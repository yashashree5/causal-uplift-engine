# Design Document: X5 download

## Overview

`ingestion/download_x5.py` reads `configs/x5/x5.yaml`, builds a per-table plan from the ordered sources, downloads each artifact to `<name>.part` with streaming reads, verifies it, renames it atomically, and updates `download_provenance.json`. Shared helpers (config loading with `extends:`, path resolution, hashing, atomic writes) live in `ingestion/x5_common.py` and are reused by every later X5 tool.

## Decisions

1. **Pinned revision.** The Hub revision is pinned in YAML so reruns fetch identical bytes; `main` may be set to follow the head.
2. **HTTP, not the Hub client.** Plain streaming HTTP with Range requests keeps resume, retry and checksum behaviour testable against a local fixture server (`tests/x5/http_fixture.py`).
3. **Never replace data.** A table that no source delivers fails the run. There is no synthetic fallback.
4. **Artifacts stay compressed.** Extraction is optional and derived; the `.csv.gz` file is the record. Plain CSV is only an optimisation for parallel DuckDB scans.
5. **Config layering.** `extends:` lets task YAML files inherit `x5.yaml`; repo-root-relative paths.

## Components

- `x5_common.py`: `load_config`, `resolve_path`, `config_get`, `ConfigError`, hashing, `atomic_write_text/json`.
- `download_x5.py`: plan builder, downloader with retry and resume, gzip check, provenance writer, CLI.

## Testing

`tests/x5/test_x5_download.py` (37 tests) and `test_x5_config.py` (9) run against the fixture HTTP server: success, resume, retry, fallback, all-sources-fail, corrupt gzip, `--force`, `--dry-run`, `--tables`, `--extract`, token handling.

## Verification status

Local tests passed. A real download of all five files from the pinned revision completed, with SHA-256 recorded. The S3 fallback returned 403 (documented).

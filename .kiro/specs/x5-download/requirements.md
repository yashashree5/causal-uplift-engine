# Requirements Document

## Introduction

Download the five X5 RetailHero tables (`uplift_train`, `uplift_test`, `clients`, `products`, `purchases`) from the Hugging Face mirror `dllllb/retailhero-uplift`, with a documented fallback, byte-for-byte integrity checks and a provenance record. One script anyone can rerun: `ingestion/download_x5.py`. Folded into Linear TEA-15 (raw landing); there is no separate issue. The status of each task is in `tasks.md`.

## Glossary

- **Downloader**: `ingestion/download_x5.py`.
- **Source**: an entry of `sources:` in `configs/x5/x5.yaml`; tried in order per table.
- **Artifact**: one downloaded file, stored unchanged under `<data_root>/downloads/`.
- **Provenance_Record**: `download_provenance.json` in the downloads directory.
- **Pinned_Revision**: the Hugging Face dataset commit `7744156a4f89f4828607921e8c9668f04801f4de`.

## Requirements

### Requirement 1: Fetch all five tables from the pinned source

**User Story:** As a pipeline owner, I want one command that fetches every table from a pinned revision, so that everyone works from the same bytes.

#### Acceptance Criteria

1. WHEN the Downloader runs without flags, THE Downloader SHALL download all five tables from the Hugging Face Source at the Pinned_Revision.
2. THE Downloader SHALL store each Artifact unchanged under `<data_root>/downloads/`.
3. WHERE `--tables` is given, THE Downloader SHALL download only the listed tables.
4. WHERE `--extract` is given, THE Downloader SHALL also write the decompressed CSV under `<data_root>/extracted/` and leave the Artifact untouched.
5. THE Downloader SHALL record the resolved repository id and revision (the id `dllllb/retailhero-uplift` redirects to `pytorch-lifestream/retailhero-uplift`).

### Requirement 2: Fallback and failure behaviour

**User Story:** As a pipeline owner, I want a failed source to fall back to the next source and never to fake data, so that a result always traces to the real dataset.

#### Acceptance Criteria

1. IF a Source fails for a table, THEN THE Downloader SHALL try the next Source in the configured order.
2. IF no Source can deliver a table, THEN THE Downloader SHALL exit with a nonzero exit code and SHALL NOT substitute toy or synthetic data.
3. THE Downloader SHALL retry transient errors with exponential backoff up to `download.max_retries`.
4. WHEN a leftover `.part` file exists, THE Downloader SHALL resume it with an HTTP Range request unless `--no-resume` is given.

### Requirement 3: Integrity and provenance

#### Acceptance Criteria

1. WHEN an Artifact is complete, THE Downloader SHALL verify that it is not empty or an error document, that its format matches the expected compression, and its size and SHA-256 against the Hub metadata (or response headers) when available, and SHALL record the computed size, MD5 and SHA-256 in the Provenance_Record.
2. WHEN an Artifact already exists and verifies, THE Downloader SHALL skip it; IF it fails verification, THEN THE Downloader SHALL refuse to replace it unless `--force` is given.
3. THE Provenance_Record SHALL name the Source used for each table.

### Requirement 4: Configuration and safety

#### Acceptance Criteria

1. THE Downloader SHALL take paths, sources, timeouts and retries from `configs/x5/x5.yaml`; precedence SHALL be CLI flag, then YAML, then built-in default.
2. THE Downloader SHALL read an optional `HF_TOKEN` from the environment and SHALL NOT store credentials in the repository or the Provenance_Record.
3. WHEN `--dry-run` is given, THE Downloader SHALL print the plan and write nothing; WITH `--check-remote` it SHALL probe sources using metadata or range-0 requests only.

### Requirement 5: Fallback source status

#### Acceptance Criteria

1. THE configuration SHALL keep the scikit-uplift S3 copy as the documented fallback.
2. THE documentation SHALL state that the fallback returned HTTP 403 AccessDenied on 2026-11-25 and that it has no `products` or `uplift_test` file.

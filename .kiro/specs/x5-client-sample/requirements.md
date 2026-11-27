# Requirements Document

## Introduction

Publish a reproducible 1% **client** sample: pick 1% of unique `client_id` values with a fixed seed and write those clients' rows from every table to `raw_sample/`, so teammates can start on real data without processing the 4 GB purchases file. Tool: `ingestion/sample_clients.py`. Linear TEA-14.

## Glossary

- **Sampler**: `ingestion/sample_clients.py`.
- **Selection_Rank**: `sha256("<seed>:<client_id>")`; clients are ranked by it ascending.
- **K**: `round(fraction x N)` where N is the number of unique client ids in `clients`.
- **Sample_Metadata**: `raw_sample/sample_metadata.json` (seed, fraction, counts, checksum of the selected ids, rules per table).

## Requirements

### Requirement 1: Client-level selection

**User Story:** As a downstream owner, I want whole clients in the sample, so that features computed on it behave like features on the full data.

#### Acceptance Criteria

1. THE Sampler SHALL select the K clients with the smallest Selection_Rank among the unique client ids of `clients`.
2. WHEN run twice with the same inputs, seed and fraction, THE Sampler SHALL produce byte-identical output files, including row order.
3. THE Sampler SHALL take seed, fraction, input and output directories from `configs/x5/sample.yaml`, overridable by CLI flags.
4. THE Sampler SHALL treat client ids as strings and preserve leading zeros.

### Requirement 2: Tables in the sample

#### Acceptance Criteria

1. THE Sampler SHALL write `clients`, `uplift_train`, `uplift_test` and `purchases` rows of the selected clients.
2. THE Sampler SHALL include every purchase row of a selected client (all rows of every transaction), so that no transaction is cut.
3. THE Sampler SHALL write the `products` rows referenced by the sampled purchases.
4. THE Sampler SHALL write `selected_client_ids.csv` and the Sample_Metadata with counts per table.
5. THE Sampler SHALL report purchase or label rows whose client is missing from `clients` instead of dropping them silently.

### Requirement 3: Safety and optional upload

#### Acceptance Criteria

1. WHERE `--dry-run` is given, THE Sampler SHALL print the plan and write nothing.
2. WHERE `--upload` is given, THE Sampler SHALL upload `raw_sample/` to `gs://<bucket>/raw_sample/` using Application Default Credentials and SHALL refuse to overwrite different remote content unless `--overwrite` is given.
3. THE Sampler SHALL read inputs with bounded memory (DuckDB) and SHALL NOT commit generated samples to Git.

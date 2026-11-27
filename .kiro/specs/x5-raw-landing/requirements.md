# Requirements Document

## Introduction

Write a manifest (file name, size, row count, MD5) for every downloaded X5 file and land the files unchanged in the team bucket under `raw/batch/<file>/ingest_date=YYYY-MM-DD/`, with the manifest next to them. Tools: `ingestion/manifest.py`, `ingestion/object_store.py`, `ingestion/land_raw.py`. The bucket is created by the infrastructure owner; this feature never provisions it. Linear TEA-15.

## Glossary

- **Landing_Tool**: `ingestion/land_raw.py`.
- **Manifest**: JSON record of one file: name, size, row count and MD5; its artifact record also holds the SHA-256 and its source record the resolved repository id and revision.
- **Object_Prefix**: `raw/batch/<file>/ingest_date=<YYYY-MM-DD>/`.
- **Expectations**: rounded work-plan row counts (45.8M purchases, 400k clients, 200k uplift_train) in `configs/x5/raw_landing.yaml`, with a tolerance.

## Requirements

### Requirement 1: Manifests

**User Story:** As a pipeline owner, I want a manifest that proves nothing was lost, so that raw can always be rebuilt and checked.

#### Acceptance Criteria

1. WHEN the Landing_Tool runs, THE Landing_Tool SHALL write one Manifest per table containing file name, size in bytes, row count and MD5, plus the SHA-256 and the download source.
2. THE row count SHALL count CSV records (not physical lines), so quoted newlines and blank lines do not change the result.
3. THE Landing_Tool SHALL compare each row count with the Expectations and report whether it is within tolerance; WHERE `--strict-expectations` is given it SHALL fail when a count is outside tolerance.
4. THE Manifest SHALL record that the uploaded object is the original artifact, byte for byte.

### Requirement 2: Landing in the bucket

#### Acceptance Criteria

1. WHEN uploading, THE Landing_Tool SHALL write each artifact unchanged to the Object_Prefix and the Manifest next to it.
2. IF a remote object exists with different content, THEN THE Landing_Tool SHALL refuse to replace it unless `--overwrite` is given; IF it has identical content, THEN it SHALL skip it.
3. THE Landing_Tool SHALL use Application Default Credentials only and SHALL NOT read or store key material in the repository.
4. THE bucket and project SHALL come from `--bucket`/`--project`, then `X5_GCS_BUCKET`/`GOOGLE_CLOUD_PROJECT`, then `configs/x5/raw_landing.yaml`.
5. IF the bucket or credentials are missing, THEN THE Landing_Tool SHALL fail with a message naming the missing setting, except in `--manifest-only` mode.

### Requirement 3: Safe local modes

#### Acceptance Criteria

1. WHERE `--manifest-only` is given, THE Landing_Tool SHALL write manifests and validate counts without touching any cloud service.
2. WHERE `--dry-run` is given, THE Landing_Tool SHALL list the objects it would write and write nothing remotely.
3. WHERE `--local-bucket-dir` is given, THE Landing_Tool SHALL rehearse the upload into a local directory, and the report SHALL state that this is not a cloud check.

### Requirement 4: Reporting

#### Acceptance Criteria

1. THE Landing_Tool SHALL write a landing report that states the mode (manifest-only, rehearsal or cloud), per-table counts and status.
2. THE documentation SHALL NOT describe a rehearsal or manifest-only run as a cloud verification.

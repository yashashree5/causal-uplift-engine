# Implementation Plan: raw landing and manifests

## Tasks

- [x] 1. Manifest builder (`ingestion/manifest.py`): MD5, SHA-256, size, CSV-aware row count
  - _Requirements: 1.1, 1.2, 1.4_
- [x] 2. Object-store interface with GCS and local-directory implementations (`ingestion/object_store.py`)
  - _Requirements: 2.2, 2.3, 3.3_
- [x] 3. Landing tool (`ingestion/land_raw.py`, `configs/x5/raw_landing.yaml`): plan, expectations, upload, report
  - _Requirements: 1.3, 2.1, 2.4, 2.5, 3.1, 3.2, 4.1_
- [x] 4. Tests (`tests/x5/test_x5_manifest.py`, `test_x5_land_raw.py`)
  - _Requirements: 1.1-1.3, 2.2, 3.1-3.3_
- [x] 5. Documentation (`ingestion/RAW_LANDING.md`)
  - _Requirements: 4.2_
- [x] 6. Manifests of the five real files (`--manifest-only`); counts within tolerance of the work plan
  - _Requirements: 1.1-1.3_
- [ ] 7. Land the real files in the team bucket and confirm manifest counts match the source
  - Blocked: needs the bucket, project and Application Default Credentials from the infrastructure owner. See spec `x5-cloud-verification`.
  - _Requirements: 2.1-2.5_

## Notes

- Branch: `ameya/x5-raw-manifest`, stacked on `ameya/x5-download`.

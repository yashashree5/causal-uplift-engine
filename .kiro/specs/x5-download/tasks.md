# Implementation Plan: X5 download

## Tasks

- [x] 1. Shared configuration helpers (`ingestion/x5_common.py`, `configs/x5/x5.yaml`)
  - Layered YAML with `extends:`, repo-root-relative paths, precedence CLI > env > YAML > default
  - _Requirements: 4.1_
- [x] 2. Downloader (`ingestion/download_x5.py`)
  - Ordered sources, retries with backoff, `.part` resume, gzip check, atomic rename, provenance record
  - _Requirements: 1.1-1.5, 2.1-2.4, 3.1-3.3, 4.2-4.3_
- [x] 3. Tests with a local fixture HTTP server (`tests/x5/test_x5_download.py`, `test_x5_config.py`)
  - _Requirements: 1.3, 2.1-2.4, 3.1-3.2, 4.1-4.3_
- [x] 4. README (`ingestion/README.md`) and requirements (`requirements-x5.txt`)
  - _Requirements: 5.2_
- [x] 5. Real download of all five tables at the pinned revision (verified; sha256 recorded)
  - _Requirements: 1.1, 1.2, 3.1_
- [x] 6. Probe the fallback source and document the result (HTTP 403 on 2026-11-25)
  - _Requirements: 5.1, 5.2_
- [ ] 7. Add the Linear issue ID to the branch and PR title
  - Folded into TEA-15 (raw landing). There is no separate Linear issue for download.

## Notes

- Branch: `ameya/x5-download`. First in the merge order.

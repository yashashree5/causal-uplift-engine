# Implementation Plan: 1% client sample

## Tasks

- [x] 1. Sampler (`ingestion/sample_clients.py`, `configs/x5/sample.yaml`): hash-rank selection, filtering of all tables
  - _Requirements: 1.1, 1.3, 1.4, 2.1-2.5_
- [x] 2. Deterministic, order-preserving writer (byte-identical across runs)
  - _Requirements: 1.2_
- [x] 3. Tests (`tests/x5/test_x5_sample.py`)
  - _Requirements: 1.1, 1.2, 2.2-2.5, 3.1_
- [x] 4. Documentation (`ingestion/SAMPLING.md`)
- [x] 5. Real 1% sample (4,002 clients, 447,380 purchase rows, 22,749 products) written to `raw_sample/`
  - _Requirements: 1.1, 2.1-2.4_
- [ ] 6. Upload `raw_sample/` to the bucket
  - Blocked: needs the bucket and credentials. The code path (`--upload`) is tested against a local directory only. See spec `x5-cloud-verification`.
  - _Requirements: 3.2_

## Notes

- Branch: `ameya/x5-client-sample`, stacked on `ameya/x5-raw-manifest`.

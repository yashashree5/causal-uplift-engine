# Implementation Plan: X5 EDA

## Tasks

- [x] 1. Settings and config (`configs/x5/eda.yaml`, `EdaSettings`, `Results`)
  - _Requirements: 3.3_
- [x] 2. Typed preparation with `TRY_CAST(NULLIF(...))` and cast-failure counting (`prepare`)
  - _Requirements: 3.1, 3.2_
- [x] 3. Sections 1-10 (`section_files` ... `section_semantics`)
  - _Requirements: 1.1-1.8, 2.1-2.3_
- [x] 4. Figures (`eda/x5_eda_plots.py`)
- [x] 5. CLI (`eda/run_eda.py`, exit codes 0/1/2)
  - _Requirements: 4.1_
- [x] 6. Notebook `eda/x5_eda.ipynb` and headless runner `eda/execute_notebook.py` (`--in-place`)
  - _Requirements: 3.4, 3.5_
- [x] 7. Tests (`tests/x5/test_x5_eda.py`) and `eda/README.md`
- [x] 8. Full-data run and notebook execution; outputs saved into the notebook
  - _Requirements: 3.4, 3.6_
- [ ] 9. Re-run the leakage check with the real campaign date
  - Blocked: the date is not published. Set `campaign.cutoff` (or pass `--cutoff`) and re-run; then regenerate `eda/FINDINGS.md`. See spec `x5-cloud-verification`.
  - _Requirements: 1.2, 1.3_

## Notes

- Branch: `ameya/x5-eda`, stacked on `ameya/x5-client-sample`.

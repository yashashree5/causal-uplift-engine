# Implementation Plan: FINDINGS

## Tasks

- [x] 1. Findings builder and renderer (`eda/make_findings.py`)
  - _Requirements: 1.1-1.4, 2.1-2.4_
- [x] 2. Tests (`tests/x5/test_x5_findings.py`)
  - _Requirements: 1.2, 1.3, 2.1_
- [x] 3. Generate `eda/FINDINGS.md` from the full-data outputs
  - _Requirements: 1.1, 2.2, 2.3_
- [x] 4. One-page summary with row references, length test (`build_summary`)
  - _Requirements: 3.1-3.3_
- [ ] 5. Review by the cleaning owner (definition of done)
  - Blocked: a teammate must review; the document says "not reviewed" until then. See spec `x5-cloud-verification`.
- [ ] 6. Regenerate after the campaign date is known
  - Blocked on the date. Findings 1, 2, 16 and the open questions change.

## Notes

- Branch: `ameya/x5-findings`, stacked on `ameya/x5-eda`.

# Requirements Document

## Introduction

`eda/FINDINGS.md`: findings of the full-data EDA, each with the action it implies for the cleaning owner. The file is generated from the EDA output files by `eda/make_findings.py`, never written by hand. Linear TEA-12.

## Glossary

- **Generator**: `eda/make_findings.py`.
- **Findings_Table**: table with columns Observation, Evidence, Recommended cleaning action, Caveat.
- **Summary**: the page-one list "what to change in cleaning".
- **Outputs**: files under `eda/outputs/full/`.

## Requirements

### Requirement 1: Generated from output files

#### Acceptance Criteria

1. THE Generator SHALL read every number from the Outputs and SHALL record in the document which files it read.
2. IF the Outputs come from a `sample` run, THEN THE Generator SHALL refuse to write findings and exit nonzero.
3. WHEN run twice on the same Outputs, THE Generator SHALL produce identical text.
4. THE document SHALL NOT claim that anyone has reviewed it; it SHALL state that the cleaning and baseline owners have not reviewed it.

### Requirement 2: Content

#### Acceptance Criteria

1. THE document SHALL contain a Findings_Table where every row has all four fields filled.
2. THE document SHALL cover: campaign date and leakage, client date columns, age validity, gender, repeated `purchase_sum` and points per product row, shared `transaction_id`, repeated product rows, zero amounts and quantities, sign of points, `trn_sum_*` columns, product attribute gaps, id formats, randomisation and uplift, uplift by segment, recency, `uplift_test` labels, table joins, and the length of the purchase history.
3. THE document SHALL list the row counts against the work-plan figures with their tolerance.
4. THE document SHALL list open questions the data cannot answer.

### Requirement 3: One-page summary

#### Acceptance Criteria

1. THE document SHALL start (after the status block) with a Summary of 10 to 14 one-line decisions, each pointing at its Findings_Table rows (`#n`).
2. THE status block and Summary together SHALL NOT exceed 450 words.
3. THE Summary SHALL be built from the same Outputs as the table, so no number in it is typed by hand.

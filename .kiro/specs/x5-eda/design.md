# Design Document: X5 EDA

## Overview

`x5_eda_lib.prepare` builds typed DuckDB tables from the text CSVs; one `section_*` function per notebook section writes scalar results to `summary.json` and tables to CSV; `x5_eda_plots.make_figures` draws figures from the result tables only. The notebook and `run_eda.py` call the same `run_eda`.

## Sections (notebook headings)

1 File statistics, nulls, duplicates, keys. 2 Timestamps. 3 Campaign cutoff and leakage. 4 Treatment and outcome. 5 Distributions. 6 Uplift by segment. 7 Product, category and id formats. 8 `uplift_test` labels. 9 Clients without purchases and unmatched ids. 10 Transaction semantics.

## Decisions

1. **Text first.** Everything is read as text; ids are never cast, so leading zeros survive.
2. **Transaction key `(client_id, transaction_id)`.** Grouping by `transaction_id` alone hid 28 shared ids and misattributed their rows; the typed-table fingerprint carries an `EDA_VERSION` so stale tables are rebuilt after such a change.
3. **Blocked, not passed.** With no published campaign date the leakage check reports `blocked`.
4. **Mode is a label.** A sample run describes the sample only; the notebook banner and `summary.json` say so.
5. **One notebook.** The committed notebook holds the outputs of the full-data run; no executed copies are kept.

## Testing

`tests/x5/test_x5_eda.py` (31): a hand-written dataset whose statistics are derived by hand in the tests; bad values, shared ids, gzip input, provenance, CLI exit codes, and headless execution of the notebook.

## Verification status

Local tests passed. Full-data run: 45,786,568 purchase rows, about 50 s, 6 GB DuckDB limit, 1.6 GB work database. The notebook ran without errors on the full data and its outputs are saved in place. Results are in `eda/outputs/full/`.

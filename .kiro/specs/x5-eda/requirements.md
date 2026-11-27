# Requirements Document

## Introduction

Explore the full X5 data before anyone builds features: one notebook, `eda/x5_eda.ipynb`, run top to bottom on the full data, backed by reusable headless code (`eda/x5_eda_lib.py`, `eda/run_eda.py`). Findings feed the cleaning rules; plots feed the report. Linear TEA-11.

## Glossary

- **EDA_Library**: `eda/x5_eda_lib.py`; `run_eda(settings)` is the entry point.
- **Notebook**: `eda/x5_eda.ipynb`, the only `.ipynb` of the EDA.
- **Mode**: `sample` (1% client sample) or `full` (all rows). A label recorded in every output; it does not change what is computed.
- **Outputs**: `summary.json`, `tables/*.csv`, `figures/*.png` and `work.duckdb` under `eda/outputs/<mode>/` (git-ignored).
- **Cutoff**: the SMS campaign date, `campaign.cutoff` in `configs/x5/eda.yaml`. Not published; unset.

## Requirements

### Requirement 1: Content of the analysis (work-plan list)

#### Acceptance Criteria

1. THE EDA_Library SHALL report row counts, columns, types, null rates and duplicate rows for each of the five tables.
2. THE EDA_Library SHALL report the purchase date range and invalid timestamps, and SHALL check whether any purchase falls after the Cutoff.
3. WHILE the Cutoff is unset, THE EDA_Library SHALL report the leakage check as `blocked` and SHALL NOT report it as passed; recency SHALL use a provisional anchor that is labelled provisional.
4. THE EDA_Library SHALL report treatment share, purchase rate per arm and the simple uplift with a 95% interval for `uplift_train`, and the standardised mean difference of pre-treatment covariates.
5. THE EDA_Library SHALL report distributions of spend per transaction, transactions per client, recency, age, gender, points use and stores per client.
6. THE EDA_Library SHALL report uplift by age band, spend tier, recency band and gender, and SHALL mark segments below the configured minimum arm size or cell count as sparse.
7. THE EDA_Library SHALL report whether product, category and client ids look hashed, and whether `uplift_test` carries labels.
8. THE EDA_Library SHALL report clients without purchases, unmatched ids and the join relationships between tables.

### Requirement 2: Transaction semantics

#### Acceptance Criteria

1. THE EDA_Library SHALL key a transaction by `(client_id, transaction_id)` and SHALL report ids shared by several clients.
2. THE EDA_Library SHALL aggregate `purchase_sum` and the points columns once per transaction and SHALL report the overstatement of a row-wise sum.
3. THE EDA_Library SHALL report the sign convention of the points columns and how `trn_sum_from_iss` relates to `purchase_sum`.

### Requirement 3: Reproducibility and bounds

#### Acceptance Criteria

1. THE EDA_Library SHALL read ids as text, cast numbers with `TRY_CAST(NULLIF(x, '') AS type)`, and count values that fail to cast instead of dropping them.
2. THE EDA_Library SHALL run with bounded memory (`duckdb.memory_limit`, `threads` from config) and SHALL reuse typed work tables only while inputs and settings are unchanged.
3. THE EDA_Library SHALL take all paths, thresholds and segment edges from `configs/x5/eda.yaml`; precedence CLI flag, then environment, then YAML.
4. THE Notebook SHALL run top to bottom without errors on the full data and SHALL print a banner naming the Mode.
5. THE Notebook SHALL be the only `.ipynb` of the EDA; `eda/execute_notebook.py` SHALL NOT write a second notebook, and `--in-place` SHALL save the cell outputs into the Notebook.
6. THE EDA SHALL NOT present sample or fixture statistics as full-data results.

### Requirement 4: Interfaces

#### Acceptance Criteria

1. `eda/run_eda.py` SHALL exit 0 on success, 1 on a runtime error and 2 on a usage or configuration error.
2. THE Outputs SHALL NOT be committed.

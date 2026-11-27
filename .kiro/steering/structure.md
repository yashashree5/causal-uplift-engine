---
inclusion: always
---

# Repository and storage layout

The repository already contains the nuPlan planner-regression work (`src/causal_planner/`, `tools/`, `dags/`, `infra/`, existing `.kiro/specs/`). The X5 uplift work lives **alongside** it. Do not edit the nuPlan files for X5 tasks.

## X5 folders (Amay)
```
configs/x5/          x5.yaml (shared), raw_landing.yaml, sample.yaml, eda.yaml, replay.yaml, ruff.toml
ingestion/           download_x5.py, manifest.py, object_store.py, land_raw.py, sample_clients.py (+ READMEs)
eda/                 x5_eda.ipynb (the only notebook), x5_eda_lib.py, x5_eda_plots.py, run_eda.py,
                     execute_notebook.py, make_findings.py, FINDINGS.md, README.md
streaming/producer/  replay.py, events.py, event_source.py, publishers.py, README.md
tests/x5/            test_x5_*.py
```
Other owners' folders from the work plan: `infra/` and `dags/` (Tejas), `streaming/sink/` and `streaming/consumer/` (Tejas), `features/` and `quality/` (Prathamesh), `baseline/` and `eval/` (Yashashree).

## Bucket layout `gs://<project-id>-x5/`
- `raw/batch/<file>/ingest_date=YYYY-MM-DD/` original CSV files untouched plus manifest (Amay)
- `raw_sample/` 1% client sample of every file (Amay)
- `raw/stream/transactions/dt=YYYY-MM-DD/` events as received from Pub/Sub (Tejas)
- `bronze/<table>/`, `silver/transactions/`, `gold/...`, `online/client_features/...`, `results/<model>/<run_id>/` (other owners)

The bucket, project and `x5-transactions` topic are created by Tejas. This repo never provisions them.

## Shared contracts (change only by pull request with all four owners reviewing)
- Stream event, `schema_version` 1: `event_id`, `client_id`, `tx_ts`, `purchase_sum`, `store_id`, `items[{product_id, quantity, amount}]`, `points{regular_received, express_received, regular_spent, express_spent}`, `schema_version`. One message per transaction, ordering key `client_id`.
- `silver.transactions`: one row per transaction (`client_id`, `transaction_id`, `tx_ts`, `purchase_sum`, `store_id`, `n_items`, `product_quantity`, `trn_sum_from_iss`, four points columns).
- Labels and folds: `client_id`, `treatment_flg`, `target`, `fold` 0 to 4 (stratified on treatment x target, seed 82718).
- No-leakage rule: every feature uses only purchases dated before the SMS campaign. The campaign date is **not published**; `campaign.cutoff` stays unset until the team confirms it.

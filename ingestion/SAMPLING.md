# Reproducible 1% client sample (`raw_sample/`)

`sample_clients.py` writes a small, deterministic copy of the X5 tables for development, CI-sized
runs and EDA "sample mode". The sampling unit is the **client**, never a row.

```bash
python ingestion/sample_clients.py --config configs/x5/sample.yaml \
    --data-root data/x5 --output-dir raw_sample          # reads data/x5/downloads/*.csv.gz
python ingestion/sample_clients.py --dry-run             # select IDs, print plan, write nothing
python ingestion/sample_clients.py --fraction 0.05 --seed 7
```

Precedence: CLI flag > environment (`X5_GCS_BUCKET`, `GOOGLE_CLOUD_PROJECT`) > YAML > default.
Fraction, seed, universe table, output dir and compression live in `configs/x5/sample.yaml`
(`sampling:`); DuckDB memory/thread limits in its `duckdb:` block.

## Method

1. Universe = unique non-empty `client_id` of `clients` (IDs are read as text; leading zeros survive).
2. Rank every ID by `(sha256("<seed>:<client_id>") hex, client_id)` and keep the first
   `K = round(fraction * N)`. The result does not depend on file order or duplicate rows.
3. Keep **all** rows of selected clients in `clients`, `uplift_train`, `uplift_test` and `purchases`
   (so every sampled transaction is complete), in source-file order.
4. `products` has no `client_id`: keep the products referenced by sampled purchases.
5. `uplift_test` stays unlabeled (source columns only).
6. Files are written to a temporary directory, validated (selected-ID count, schema preserved,
   every sampled client present in the sampled `clients`, no train/test overlap, no label columns in
   `uplift_test`, products exactly those referenced, source client IDs absent from `clients` reported),
   and only then swapped into place. Whole-transaction completeness is covered by the unit tests.

`sample_metadata.json` records fraction, seed, method, `K`, the checksum of the selected IDs, per-file
rows/bytes/md5, validation results, and the provenance of each source artifact
(sha256, source URL, revision, manifest row count). `selected_client_ids.csv` lists the IDs.

## Optional upload

`--upload --bucket <name> [--project <id>]` reuses the landing preflight/upload code to put the sample
under `gs://<bucket>/raw_sample/`. `--local-bucket-dir` rehearses it into a directory (not a cloud
check). Credentials are Application Default Credentials only. **Not run against a real bucket.**

## What was actually run

- Full data (`data/x5/downloads`, revision `7744156a…`), fraction 0.01, seed 82718, 40 s:
  4,002 of 400,162 clients; uplift_train 1,948; uplift_test 2,054; purchases 447,380 rows;
  products 22,749. Selected-ID sha256 `6b603fe4…54bc`.
- Two independent runs produced byte-identical outputs (all seven files).
- The sample is a *client* sample: its purchase rows are about 0.98% of the table, not exactly 1%.

## Tests

`tests/x5/test_x5_sample.py` (25 tests, tiny test-only fixtures from `tests/x5/x5_fixtures.py`).

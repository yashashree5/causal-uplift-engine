# X5 RetailHero uplift: data collection, 1% sample, EDA and findings

Branch `ameya/x5-findings`. This branch carries the whole data-collection and exploration workstream for the X5 RetailHero
uplift dataset: download, raw landing and manifests, a reproducible 1% client sample, full-data EDA, and a generated findings
report. It is the top of a stack of five branches, so it contains the work of all of them.

The repository's own `README.md` describes the separate nuPlan planner-regression project; the X5 work sits beside it and does not
edit it.

## What is in this branch

| Task | Branch it came from | Code | Detail |
|---|---|---|---|
| Download the five tables | `ameya/x5-download` | `ingestion/download_x5.py`, `ingestion/x5_common.py` | [ingestion/README.md](ingestion/README.md) |
| Raw landing and manifests | `ameya/x5-raw-manifest` | `ingestion/land_raw.py`, `manifest.py`, `object_store.py` | [ingestion/RAW_LANDING.md](ingestion/RAW_LANDING.md) |
| Reproducible 1% client sample | `ameya/x5-client-sample` | `ingestion/sample_clients.py` | [ingestion/SAMPLING.md](ingestion/SAMPLING.md) |
| Full-data EDA | `ameya/x5-eda` | `eda/x5_eda.ipynb`, `eda/run_eda.py`, `eda/x5_eda_lib.py`, `eda/x5_eda_plots.py`, `eda/execute_notebook.py` | [eda/README.md](eda/README.md) |
| Findings report | `ameya/x5-findings` | `eda/make_findings.py` and the generated `eda/FINDINGS.md` | [eda/FINDINGS.md](eda/FINDINGS.md) |

Settings for all of them live in [`configs/x5/`](configs/x5/) (`x5.yaml` plus one file per task). Dependencies are pinned in
[`requirements-x5.txt`](requirements-x5.txt), with task extras in `eda/requirements.txt` and `ingestion/requirements-gcs.txt`.
Tests are in [`tests/x5/`](tests/x5/).

Not in this branch: the replay producer (`ameya/x5-replay-producer`, depends only on the download branch) and the Kiro steering files
and specs (`ameya/x5-kiro-setup`).

## Dataset

- **Source:** Hugging Face `dllllb/retailhero-uplift`, which redirects to `pytorch-lifestream/retailhero-uplift`. Pinned revision
  `7744156a4f89f4828607921e8c9668f04801f4de`. Every downloaded file is verified against the Hub's SHA-256.
- **Fallback:** the scikit-uplift S3 copy is configured but returned HTTP 403 when tested, so Hugging Face is the only source used.
- **Tables (full data):** `purchases` 45,786,568 rows (about 0.6 GB compressed, 4.46 GB extracted), `clients` 400,162,
  `uplift_train` 200,039, `uplift_test` 200,123 (no labels), `products` 43,038.
- **Campaign date:** not published. The dataset description says `purchases` is history "before the SMS campaign", but gives no
  date and nothing in the data can confirm it. `campaign.cutoff` is therefore unset, and the leakage check reports `blocked`, not passed.

## Quick start

Run from the repository root with Python 3.11.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-x5.txt -r eda/requirements.txt

# 1. Download (about 0.6 GB; --extract also writes the plain CSVs, about 4.5 GB for purchases)
python ingestion/download_x5.py --extract

# 2. Manifests and landing report, local only (no cloud access needed)
python ingestion/land_raw.py --manifest-only

# 3. 1% client sample -> raw_sample/
python ingestion/sample_clients.py

# 4. EDA on the full data, then the findings report
python eda/run_eda.py --mode full --input-dir data/x5/extracted
python eda/make_findings.py

# 5. Notebook: run headlessly and save the outputs into eda/x5_eda.ipynb
python eda/execute_notebook.py --mode full --input-dir data/x5/extracted --in-place
```

Every script has `--dry-run` or `--help`. The first run of the EDA builds a typed working DuckDB file and reuses it while the inputs
and settings are unchanged.

## What each step produces

- **Download** writes the files exactly as served to `data/x5/downloads/`, plus `download_provenance.json` (source, revision,
  checksums, retrieval time). Interrupted downloads resume; reruns skip verified files; failures never fall back to synthetic data.
- **Raw landing** writes one `manifest.json` per table (file name, size, row count, MD5) and, when given a bucket, uploads the
  unchanged files to `gs://<bucket>/raw/batch/<table>/ingest_date=YYYY-MM-DD/`. It never creates buckets or IAM bindings.
- **Sample** keeps 1% of unique `client_id`s, chosen by hash rank with seed 82718, and **all** rows of each chosen client. The current
  sample has 4,002 of 400,162 clients and 447,380 purchase rows. Two independent runs produced identical files.
- **EDA** writes tables, figures and `summary.json` to `eda/outputs/<mode>/`. The notebook and the CLI run the same code. The
  notebook has 13 code cells and carries the outputs of the full-data run; it is the only `.ipynb` in `eda/`.
- **Findings** is generated, never edited by hand. Every number is read from the full-data output files, and the generator refuses
  outputs from a sample run.

## Headline findings

The full report, with evidence, a recommended cleaning action and a caveat for each, is [eda/FINDINGS.md](eda/FINDINGS.md). It opens with
a one-page summary of 12 proposed cleaning decisions. The main ones:

- A row-wise sum of `purchase_sum` overstates spend 10.34x, because the value repeats on every product row of a transaction.
  Aggregate once per (`client_id`, `transaction_id`).
- `transaction_id` alone is not unique: 28 ids belong to two clients, so a transaction key must include `client_id`.
- Points-spent columns are stored as non-positive numbers (585,800 negative values); received points are never negative.
- 3,190 clients (0.80%) have impossible ages; 11.2% of clients first redeem after the last purchase in the data.
- IDs are hashed hex strings, and some look numeric (6.2% of client ids start with `0`), so they must be read as strings.
- Treatment looks randomised (49.98% treated). Uplift is +3.32 pp (95% CI +2.90 to +3.75 pp) and varies by segment, from +1.34 pp
  (ages 16 to 24) to +5.90 pp (ages 65 and over).
- The history is short (117 days), so use daily or weekly windows.

## Configuration and precedence

- Paths, seeds, thresholds, bucket and project all come from YAML in `configs/x5/`. Relative paths resolve against the repository root.
- Precedence: CLI flag, then environment variable (where documented), then YAML, then built-in default.
- The GCP project and bucket are `null` until the infrastructure owner provides them. Credentials come from Application Default
  Credentials only and are never stored in the repository.

## Files that are never committed

`.gitignore` excludes `data/`, `raw_sample/`, `eda/outputs/`, `streaming/producer/outputs/` and `streaming/producer/checkpoints/`.
Raw data, samples, manifests with local paths, and generated outputs stay local.

## Tests and lint

```bash
pytest tests/x5 -q
ruff check --config configs/x5/ruff.toml ingestion eda tests/x5
ruff format --check --config configs/x5/ruff.toml ingestion eda tests/x5
```

166 tests across seven files: config 9, download 37, manifest 27, raw landing 24, sample 25, EDA 31, findings 13. They use small
hand-made fixtures and a local HTTP server, never the real network or the real data. The EDA tests execute the notebook headlessly,
so they start a Jupyter kernel. `make check` does not collect these pytest-style tests; run `pytest`.

## What has and has not been verified

| Check | Status |
|---|---|
| Download from Hugging Face, checksum verification, resume | Run on the real data (single observation: about 3 minutes for the remaining 603 MB plus extraction) |
| 1% sample on the real data | Run (40 s); two runs byte-identical |
| EDA and findings on the full data | Run on all 45.8M purchase rows (48.6 s on the last run; a single observation, not a benchmark) |
| Manifests | Written locally and checked against the work plan's rounded row counts (all within 1%) |
| Upload to a real bucket (raw files, manifests, sample) | **Not run.** No bucket or credentials were available |
| Cloud behaviour of the landing code | Rehearsed only against a local directory standing in for the bucket |
| Review of `eda/FINDINGS.md` by the cleaning owner | **Not done.** The report is marked "not reviewed" and its cleaning actions are proposals |

## Open decisions that these results depend on

1. **Campaign date.** Needed to run the leakage check and to build final features. Until then recency uses the provisional anchor
   `2019-03-18 23:40:03` (the last purchase).
2. **Points-spent sign.** Whether features use the stored non-positive value or its absolute value.
3. **`items.amount` for the event stream.** `trn_sum_from_iss` is assumed and unverified (finding 11 in the report).
4. **Review of the findings** by the cleaning owner.

## Merge order

`ameya/x5-download`, then `ameya/x5-raw-manifest`, `ameya/x5-client-sample`, `ameya/x5-eda`, `ameya/x5-findings`. Each branch is stacked
on the previous one, so open each pull request against its parent and retarget to `main` once the parent has merged.

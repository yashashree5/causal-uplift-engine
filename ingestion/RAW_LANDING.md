# Raw landing and manifests

`land_raw.py` (CLI) with `manifest.py` (scanning/validation) and `object_store.py` (GCS +
local stand-in) write one manifest per logical table and land the **unchanged** downloaded
artifacts in Cloud Storage:

```
gs://<bucket>/raw/batch/<logical_table>/ingest_date=YYYY-MM-DD/<artifact>      e.g. purchases.csv.gz
gs://<bucket>/raw/batch/<logical_table>/ingest_date=YYYY-MM-DD/manifest.json
```

`<logical_table>` is one of `uplift_train`, `uplift_test`, `clients`, `products`, `purchases`.
The bucket (`gs://<project-id>-x5`) is created by Tejas's infrastructure work; this code never
creates buckets, IAM bindings or any other cloud resource.

## Commands

```bash
# Local only: scan the artifacts, write manifests + a validation report. No cloud access at all.
python ingestion/land_raw.py --manifest-only

# Cloud upload rehearsal: also prints the exact gs:// objects that would be written. No bucket or
# credentials needed (paths use a placeholder bucket).
python ingestion/land_raw.py --dry-run

# Same, but look the objects up in the real bucket and predict upload / skip / conflict.
python ingestion/land_raw.py --dry-run --check-remote --bucket <project-id>-x5

# Real upload (artifacts first, manifests last).
python ingestion/land_raw.py --bucket <project-id>-x5 --ingest-date 2026-11-26

# Replace remote objects whose content differs (never the default).
python ingestion/land_raw.py --bucket <project-id>-x5 --overwrite

# Rehearse the whole upload path into a local directory standing in for the bucket (not a cloud check).
python ingestion/land_raw.py --local-bucket-dir /tmp/fake-gcs --bucket demo-x5
```

| Flag | Meaning |
| --- | --- |
| `--config` | default `configs/x5/raw_landing.yaml` |
| `--data-root`, `--tables`, `--ingest-date` | local root, table subset, `YYYY-MM-DD` (default: today in **UTC**) |
| `--bucket`, `--project` | bucket (bare name or `gs://name`) and GCP project |
| `--manifest-only` / `--dry-run` / `--check-remote` | see above; `--manifest-only` and `--dry-run` are mutually exclusive |
| `--overwrite` | allow replacing differing remote objects |
| `--strict-expectations` | a row count outside the configured tolerance becomes an error (exit 1) |
| `--reuse-manifests` | skip re-scanning an artifact whose local manifest is current (same name, size, MD5) |

**Precedence:** CLI flag > environment (`X5_GCS_BUCKET`, `GOOGLE_CLOUD_PROJECT`) > YAML (`gcp.bucket`,
`gcp.project`) > default. **Credentials:** Application Default Credentials only
(`gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS` pointing at a key file kept
outside the repository). Nothing credential-like is read from, or written to, the config, manifests or
reports.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | OK (manifests written / upload complete / dry run complete) |
| 1 | Validation failed: missing file, missing required column, zero rows, artifact differs from its download provenance, or (`--strict-expectations`) a count outside tolerance |
| 2 | Usage or configuration error |
| 3 | Remote conflict (object exists with different content and no `--overwrite`), precondition lost, or transfer-integrity failure |
| 4 | Cloud not configured: no bucket, no credentials, bucket missing, or permission denied |

## Manifest (`manifest.json`)

Deterministic: the same artifact and ingest date produce byte-identical JSON (there is no
generation timestamp), which is what makes reruns idempotent.

| Field | Meaning |
| --- | --- |
| `logical_table`, `ingest_date`, `dataset` | identity |
| `filename`, `bytes`, `md5`, `compression` | the uploaded **artifact** (e.g. `purchases.csv.gz`); `md5` is a content hash of its bytes |
| `row_count` | exact number of CSV **data records**, excluding the header (and blank lines, which are reported in `scan_warnings`) |
| `row_count_method` | how it was counted (below) |
| `columns` | header of the CSV |
| `artifact` | the same artifact plus `sha256` and the destination `object_name` |
| `member` | for compressed artifacts, the **CSV member** inside it: `filename`, `bytes`, `md5`, `sha256`, `row_count`, and `artifact` (the artifact it belongs to). `null` when the artifact itself is a CSV |
| `source` | from `download_provenance.json`: source name/URL, requested and resolved revision, retrieval timestamp |
| `download_provenance_check` | `matches_artifact`, `MISMATCH_with_artifact` (validation error) or `no_download_provenance` |

Counting rule: the decompressed stream is read in 1 MiB chunks; `\n` terminators **outside quoted
fields** are counted (RFC 4180 quoting, `""` escapes). Quoted newlines therefore do not inflate the count
(plain `wc -l` would). Memory use does not depend on file size (a unit test checks the peak with
`tracemalloc`). Bare `\r` line endings, an unterminated quote or an empty file are errors. The counter
agrees with Python's `csv` module on 600 random CSVs (including random chunk sizes of 1 byte), and with an
independent DuckDB `read_csv` count on all five real files (see below).

## Validation and expectations

* **Required files**: each configured table must have an artifact on disk.
* **Required columns**: `raw_landing.required_columns` (read from the real headers on 2026-11-25). A
  missing column is an error; extra or reordered columns are warnings.
* **Row-count expectations**: `raw_landing.expectations` holds the *approximate* numbers quoted in the work
  plan (purchases 45.8M, clients 400k, uplift_train 200k, uplift_test 200k, products about 43k) with a
  1% tolerance. They are only compared against the **measured** count, reported as
  `expectation_check`, and never copied into a manifest. A miss is a warning unless
  `--strict-expectations`.

## Storage transfer integrity vs content MD5

These are different checks and are recorded in different places:

* **Content MD5/SHA-256** — properties of the file; computed locally; stored in `manifest.json`.
* **Transfer integrity** — did Cloud Storage receive the same bytes? The upload uses
  `checksum="crc32c"` (client-side verification during the transfer), and afterwards the object's
  reported size, CRC32C and MD5 are compared with the local file. The result is `transfer_verified` per
  object in the local `landing_report_ingest_date=....json`, not in the manifest.

## Idempotency and overwrite policy

1. Every target object is looked up **before anything is written** (preflight).
2. Identical content (size + CRC32C/MD5) is skipped (`skipped_identical`).
3. Missing objects are created with `if_generation_match=0`, so a concurrent writer cannot be
   clobbered.
4. Differing content stops the run with exit code 3 and *nothing uploaded* (no partial landing). With
   `--overwrite` the object is replaced using `if_generation_match=<generation seen in preflight>`.
5. Artifacts are uploaded without a `Content-Encoding`, so Cloud Storage keeps and serves the exact
   gzip bytes (no decompressive transcoding). Manifests go last: a table's manifest present means its
   artifact landed.

## Local outputs (git-ignored, under `data/x5/manifests/`)

* `<table>/ingest_date=<date>/manifest.json` — the manifest that is uploaded.
* `landing_report_ingest_date=<date>.json` — validation result per table, expectation checks, scan seconds,
  and per-object upload status with transfer verification.

## What has and has not been verified

| Check | Status |
| --- | --- |
| Manifest scan of the real Hugging Face artifacts (all five tables) | **done 2026-11-26** — see below |
| Row counts vs an independent DuckDB `read_csv` count | **done**, all five equal |
| Upload logic against a directory-backed store and a mocked `google-cloud-storage` client | unit tests |
| Upload to a real bucket, remote verification | **not done** — no bucket or credentials available to this work |

Measured on the real files (`--manifest-only`, one run; `purchases.csv.gz` scan took 83.7 s on a laptop,
which is a single observation and not a benchmark): uplift_train 200,039 rows; uplift_test 200,123;
clients 400,162; products 43,038; purchases 45,786,568. All five are within the 1% tolerance of the work
plan's approximate figures. Regenerate them with the command above; they are not hard-coded anywhere.

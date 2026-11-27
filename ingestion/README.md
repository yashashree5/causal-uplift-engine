# X5 RetailHero ingestion

Scripts that collect the X5 RetailHero uplift data (one randomized SMS experiment run by the
retailer X5 in 2019). Everything here is the **batch collection** side owned by Amay; bronze
conversion (`ingestion/to_bronze.py`) belongs to Tejas.

| Step | Code | Status |
| --- | --- | --- |
| Download the five tables | `download_x5.py` (this document) | implemented |
| Land raw files + manifests in GCS | `land_raw.py`, `manifest.py`, `object_store.py` ([RAW_LANDING.md](RAW_LANDING.md)) | implemented |
| Reproducible 1% client sample | `sample_clients.py` ([SAMPLING.md](SAMPLING.md)) | implemented |

Shared settings live in [`configs/x5/x5.yaml`](../configs/x5/x5.yaml); shared helpers in
[`x5_common.py`](x5_common.py).

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-x5.txt
```

Run commands from the repository root. Paths in the config are relative to the repository
root, so the working directory does not matter.

## Source (verified 2026-11-25)

| Fact | Evidence |
| --- | --- |
| Hugging Face id `dllllb/retailhero-uplift` is a redirect (HTTP 307) to `pytorch-lifestream/retailhero-uplift`. The script records both ids. | `GET /api/datasets/dllllb/retailhero-uplift` |
| Pinned revision `7744156a4f89f4828607921e8c9668f04801f4de` (last modified 2024-02-19). | Hub API `sha` field |
| Files: `data/{uplift_train,uplift_test,clients,products,purchases}.csv.gz`, each a single gzip stream (not an archive). A sixth file, `uplift_sample_submission.csv.gz`, exists in the repo and is **not** downloaded. | Hub API file list, `README.md` `configs:` block of the dataset |
| The Hub publishes a SHA-256 (LFS object id) and size per file; the script verifies every download against them. | Hub API `siblings[].lfs.sha256` |
| The dataset's own `description.txt` says `purchases.csv` is purchase history "before the SMS campaign". It publishes **no campaign date**. | Hugging Face repo `description.txt` |
| Fallback (scikit-uplift S3, from `sklift/datasets/datasets.py::fetch_x5`): only `uplift_train`, `clients`, `purchases`, plus their MD5 values. It has **no `products` or `uplift_test` file**. | scikit-uplift source |
| On 2026-11-25 the S3 bucket answered HTTP 403 `AccessDenied` for all five anonymous requests, so the fallback is currently unusable; it is kept because it is the documented fallback. | `--dry-run --check-remote` output |

The Hugging Face `.gz` bytes are **not** the S3 bytes: the MD5 values of the downloaded
Hugging Face artifacts differ from the MD5 values published in scikit-uplift's loader. Whether the
two mirrors contain identical *rows* has not been verified. Provenance therefore records which
source produced each artifact, and a run that mixes sources prints a warning.

## Usage

```bash
# 1. See what would happen. No network, nothing written:
python ingestion/download_x5.py --dry-run

# 2. Same plan, plus metadata/range-0 probes of each source (no dataset bytes):
python ingestion/download_x5.py --dry-run --check-remote

# 3. Download all five tables (about 0.6 GB compressed) into data/x5/downloads/:
python ingestion/download_x5.py

# 4. Also write the decompressed CSVs (about 4.5 GB for purchases) into data/x5/extracted/:
python ingestion/download_x5.py --extract

# Subsets / overrides:
python ingestion/download_x5.py --tables uplift_train,clients --data-root /big/disk/x5
python ingestion/download_x5.py --source huggingface --max-retries 8 --read-timeout 120
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | YAML config (default `configs/x5/x5.yaml`). |
| `--data-root DIR` | Override `paths.data_root`. |
| `--tables a,b` | Subset of the five logical tables. |
| `--source NAME` | Use only this configured source (repeatable). |
| `--dry-run` | Print the plan; write and download nothing. |
| `--check-remote` | With `--dry-run`: probe sources with metadata / `Range: bytes=0-0` requests. |
| `--force` | Replace an existing artifact that fails verification (otherwise it is never overwritten). |
| `--extract` / `--no-extract` | Also write (or skip) extracted CSV members. |
| `--no-resume` | Delete leftover `.part` files instead of resuming them. |
| `--max-retries`, `--connect-timeout`, `--read-timeout` | Override the `download:` block. |

**Precedence:** CLI flag > YAML config > built-in default. The only environment variable the
download reads is the optional Hugging Face token named by `auth_token_env` (`HF_TOKEN`); it is sent to
`huggingface.co` only and is dropped by `requests` on redirects to the CDN host. No credentials are
stored in the config or in the provenance file.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Every requested table is present and verified (downloaded, skipped or adopted). |
| 1 | A table could not be downloaded: network failure after retries, HTTP 4xx/5xx, no source provides the table, extraction error. |
| 2 | Usage or configuration error (unknown table/source, bad YAML, bad flag combination). |
| 3 | Integrity failure: size/SHA-256/MD5 mismatch, wrong file format, or an existing artifact that cannot be trusted. |

Failures never fall back to toy or synthetic data; a failed table simply has no artifact.

## Behaviour guarantees

* **Bytes are kept as served.** The artifact `purchases.csv.gz` is exactly what the server sent.
  It is written to `purchases.csv.gz.part` and renamed (`os.replace`) only after the size, format and
  checksums pass, so a crash can never leave a truncated file under the final name.
* **Artifacts and members are different things.** `--extract` writes the decompressed CSV to
  `extracted/purchases.csv`. The provenance file keeps the artifact record (`bytes`, `md5`, `sha256`
  of the `.csv.gz`) and, separately under `extracted_members`, the member record (`kind:
  extracted_csv_member`, its own `bytes`/`md5`, and the artifact SHA-256 it came from). Hashes of the
  two are never interchangeable.
* **Compression.** The format is detected from magic bytes: gzip, zip, tar, bzip2 and zstd are
  recognised; an HTML/XML/JSON body (a typical error page served with HTTP 200) is rejected.
  gzip, bzip2, zip (exactly one `.csv` member) and tar (exactly one `.csv` member) can be extracted. Archive member
  paths are reduced to their base name (no zip-slip).
* **Retries.** Timeouts, connection errors, short transfers, HTTP 429 and 5xx are retried with
  exponential backoff (`backoff_initial_s * backoff_factor^n`, capped at `backoff_max_s`; a `Retry-After`
  header raises the delay up to the cap). HTTP 4xx other than 429 is permanent and moves on to the next source.
* **Resume.** After a dropped connection the next attempt sends `Range: bytes=<size of .part>-`. A
  server that ignores `Range` causes a clean restart. If a resumed file fails verification it is deleted
  and fetched once more from byte 0. A `.part` file left by a killed process is resumed on the next run
  (use `--no-resume` to discard it).
* **Safe reruns.** If the artifact exists and matches its provenance record, it is skipped without any
  network request. An existing file with no record is adopted only if it matches the independent
  Hub SHA-256. Anything else is refused unless `--force` is given, which replaces the file atomically.
* **Provenance** (`data/x5/downloads/download_provenance.json`): per logical table the artifact name and
  relative path, source name, URL, redirect-resolved repo id and revision, retrieval timestamp (UTC),
  bytes, MD5, SHA-256, detected compression, each verification check (expected vs actual and where the
  expected value came from), and any extracted members. Query strings of CDN redirects (signed URLs) are
  not stored.

## What was actually run

Full download on 2026-11-25 from Hugging Face revision `7744156a…`, with `--extract`, on a home
network (the runtime is a single observation, not a benchmark):

* All five artifacts verified against the Hub SHA-256 (`independently_verified: true`).
* The first process was killed after 6.3 MB of `purchases.csv.gz`; a rerun resumed the `.part` file
  (`resumed_partial: true`) and the final SHA-256 matched.
* The remaining 603 MB of `purchases.csv.gz` plus its 4.46 GB extraction took under 3 minutes.

## Tests

```bash
python -m pytest tests/x5 -q
ruff check --config configs/x5/ruff.toml ingestion tests/x5
ruff format --config configs/x5/ruff.toml --check ingestion tests/x5
```

The download tests use a local HTTP server (`tests/x5/http_fixture.py`) that can fail, truncate,
redirect, ignore `Range` and mimic the Hub metadata endpoint. They never touch the real network.
`make check` (the repository's unittest runner) does not collect these pytest-style tests; run `pytest`.

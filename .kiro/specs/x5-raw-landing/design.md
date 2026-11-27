# Design Document: raw landing and manifests

## Overview

`manifest.py` computes content hashes and counts CSV records in one streaming pass over the gzip file. `object_store.py` hides the destination behind a small interface with two implementations: Google Cloud Storage (lazy import of `google-cloud-storage`, listed in `ingestion/requirements-gcs.txt`) and a local directory used for rehearsals and tests. `land_raw.py` composes them: plan, manifests, validation, upload, report.

## Decisions

1. **Two hashes.** MD5 is what the work plan asks for; SHA-256 is the integrity record used in provenance.
2. **Record counting.** Rows are counted with a CSV-aware counter, not `wc -l`, because quoted values may contain newlines.
3. **No overwrite by default.** A second run against the same date is a no-op for identical content and an error for different content.
4. **Cloud code is optional.** The GCS client is imported only when a cloud run is requested, so local work needs no cloud package or credentials.
5. **Honest modes.** The report names the mode so a rehearsal can never be mistaken for a cloud run.

## Testing

`test_x5_manifest.py` (21) and `test_x5_land_raw.py` (24): counts with quoted newlines and blank lines, hash correctness, expectations tolerance, idempotent upload, refusal to overwrite, dry-run, missing bucket and credentials, local-directory rehearsal.

## Verification status

Local tests passed. Manifests for all five real files were produced with `--manifest-only`; counts match the work plan within 1%. **No real upload has happened**: no bucket or credentials exist yet.

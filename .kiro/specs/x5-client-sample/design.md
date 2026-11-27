# Design Document: 1% client sample

## Overview

The Sampler ranks unique client ids by `sha256("<seed>:<client_id>")`, takes the first K, and filters every table with a DuckDB semi-join on that id set. Output files are gzip CSV with the source column order.

## Decisions

1. **Hash ranking instead of a random generator.** The selection depends only on the seed and the id, never on file order, library versions or threads, and the selected set grows consistently if the fraction changes.
2. **Sample unit is the client.** Whole clients are kept so no transaction is truncated and per-client features are valid.
3. **Deterministic row order.** Semi-joins and parallel scans return rows in arbitrary order; the writer numbers source rows (`row_number() OVER ()`) and orders by that number so repeated runs are byte-identical.
4. **Sample is not the full data.** `sample_metadata.json` records the fraction and seed; the EDA labels runs on it as sample runs, and results from it must never be quoted as full-data results.

## Testing

`tests/x5/test_x5_sample.py` (25): selection by rank, determinism and byte identity, whole-client transactions, products subset, leading zeros, orphan reporting, dry-run, local-bucket upload rehearsal.

## Verification status

Local tests passed. Real run: 4,002 clients, 447,380 purchase rows and 22,749 products in about 40 s. The upload path was rehearsed against a local directory only.

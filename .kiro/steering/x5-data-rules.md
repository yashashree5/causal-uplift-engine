---
inclusion: fileMatch
fileMatchPattern: "{ingestion,eda,streaming}/**/*.py"
---

# X5 data rules (learned from the real data; see `eda/FINDINGS.md`)

- **IDs are strings.** `client_id`, `transaction_id`, `product_id`, `store_id` are 10-character hex strings. 6.2% of client ids start with `0` and 2,967 look like scientific notation. Read them as text (`all_varchar`, explicit dtype) and never cast them.
- **Transaction key is `(client_id, transaction_id)`.** `transaction_id` alone is shared by two clients for 28 ids. Never group by it alone.
- **Never sum `purchase_sum` or the points columns over product rows.** They repeat on every row of a transaction (a row-wise sum is 10.34x too large). Aggregate once per transaction (`max`).
- **DuckDB casts.** Use `TRY_CAST(NULLIF(col, '') AS type)`. A bare `TRY_CAST` on a CSV-scan column is pushed into the reader and raises on the first bad value.
- **Deterministic output.** Joins and parallel scans return rows in nondeterministic order: add an explicit `ORDER BY` (and a `row_number()` when order must follow the source). Multi-threaded float sums are not bit-stable: compare with a tolerance.
- **Points spent are stored non-positive.** Documented in `eda/FINDINGS.md`; the cleaning owner decides whether features take the absolute value. The producer publishes the stored value unless configured otherwise.
- **`items.amount` source is unverified.** The producer uses `trn_sum_from_iss` by assumption; it is configurable (`replay.event.item_amount_column`).
- **Sample is not full data.** A sample or fixture result must never be presented as a full-data result. The campaign date is unknown, so the leakage check is `blocked`, not `passed`.
- **Never fabricate results**, counts, runtimes or cloud checks. Report what was actually run.

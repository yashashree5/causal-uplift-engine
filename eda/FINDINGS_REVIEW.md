# Review of `eda/FINDINGS.md` — cleaning/feature owner (Prathmesh)

Reviewer: Prathmesh (cleaning and feature extraction).
Subject: `eda/FINDINGS.md`, generated from the full-data EDA run.
Purpose: the work plan requires the cleaning/feature owner to review the EDA
findings; FINDINGS.md states its cleaning actions are *proposals* for me to
accept or change. This document records that decision for each one, so silver and
the gold feature library have an agreed rule set to build against.

## Verdict

**Accepted, with the amendments and blockers noted below.** The findings are
reproducible (generator + provenance recorded), internally consistent with
`configs/x5/eda.yaml`, and the row counts match the work plan within tolerance.
I checked the config-dependent claims (age range, segment edges) against
`configs/x5/eda.yaml` and they agree. The cleaning proposals are sound; most are
accepted as written, a few are tightened below.

## Decisions on each proposed cleaning action

Numbering follows the "Summary: what to change in cleaning" list in FINDINGS.md.

| # | Proposal | Decision | Note |
|---|---|---|---|
| 1 | Campaign date unknown; leakage check blocked; give every time feature a cutoff; use provisional anchor `2019-03-18 23:40:03` | **Accept** | Done in `features/aggregates.py`: `cutoff` is a required parameter, transactions are filtered strictly before it, recency is measured from it. The anchor stays provisional and is **not** hard-coded in code. Blocker B1 below. |
| 2 | Do not use `first_redeem_date`; it is leakage-suspect | **Accept** | Excluded from the feature set. Not used as an input. |
| 3 | Treat `first_issue_date` as noisy; do not use it to filter/order purchases; tenure allowed but flag "issued after first purchase" | **Accept with amendment** | v1 orders purchases by `transaction_datetime` only, never by issue date. v1 tenure uses observed first/last purchase (`tenure_days`), not the card issue date, so no issue-date flag is needed in v1. Revisit if we add a card-tenure feature. |
| 4 | Ages outside [16, 100] → missing + `age_invalid`; keep clients; do not impute before split | **Accept** | Implemented exactly: out-of-range or missing age → `age = NaN`, `age_invalid = 1`; no imputation in the library. |
| 5 | Keep gender `U`; keep missing product attributes as explicit 'unknown' | **Accept** | Gender is one-hot `gender_F/M/U`; `U` is a real level. Product-attribute 'unknown' handling applies to sequences/summaries (later tasks), noted there. |
| 6 | Aggregate `purchase_sum` and points once per (client_id, transaction_id); never row-sum | **Accept — belongs to silver** | The feature library assumes transaction-level input (one row per transaction). The once-per-transaction collapse is `features/silver.py`'s job (blocked on bronze, B2). The library does not re-aggregate product rows. |
| 7 | Key a transaction by (client_id, transaction_id) | **Accept** | Will be the silver key and the replay `event_id` basis. Encoded as an assumption in the feature docstring. |
| 8 | Read every ID as a string | **Accept** | Feature code treats `store_id`/`client_id` as strings. Silver and all readers must set explicit string dtypes. |
| 9 | Keep zero-quantity / zero-sum rows, flag them; null `trn_sum_from_red` = 0 + indicator | **Accept with scope note** | `zero_qty_tx_count` feature counts zero-quantity transactions (kept, not dropped). Zero-sum handling and the `trn_sum_from_red` indicator live in silver. |
| 10 | Points 'spent' stored non-positive → take absolute value | **Accept — belongs to silver** | The feature library assumes points-spent arrive as non-negative magnitudes and sums them directly. Silver must apply `abs()`. Documented in `DICTIONARY.md`. |
| 11 | `uplift_test` has no labels; tables join cleanly; keep as assertions | **Accept** | Join assertions belong in silver/quality. Labels + folds come only from `uplift_train`. |
| 12 | History short (117 days); use daily/weekly windows, not calendar months | **Accept with amendment** | v1 uses 30/60/90-day rolling windows relative to the cutoff rather than calendar months. See open question Q1 on window sizes given the short history. |

(The remaining summary points 10–12 in FINDINGS map to findings #14–#19, all
descriptive; no cleaning action is needed beyond what is listed above.)

## How this maps to what is already built (TEA-18)

The feature library, data dictionary, and quality checks delivered in TEA-18
already encode the accepted decisions:

- `features/aggregates.py` — cutoff parameter (#1), age rule (#4), gender `U` as a
  level (#5), zero-quantity flag (#9), points-spent treated as magnitudes (#10),
  `first_issue_date`/`first_redeem_date` excluded (#2, #3).
- `quality/checks.py` — `no_purchase_after_cutoff` **skips** rather than passes
  when the cutoff is unset (#1), per-fold treatment share in [0.45, 0.55] (#14),
  non-negative `purchase_sum` (#9), labeled clients present, no null `client_id`.
- `features/DICTIONARY.md` — records every feature's formula, unit, and the
  cleaning rule it depends on.

## Blockers this review surfaces for downstream work

- **B1 — campaign date.** Features and the leakage check cannot be finalised
  until `campaign.cutoff` is set in `configs/x5/x5.yaml` from an authoritative
  source. Until then everything time-based is provisional and the leakage check
  reports "skipped", by design. (FINDINGS open question #1.)
- **B2 — bronze tables.** Silver and the gold builders depend on Tejas's typed
  bronze Parquet, which does not exist yet. The pure feature library was built
  and tested without it; the rest waits on bronze.
- **B3 — `items.amount` source.** Whether `items.amount` in the stream contract
  comes from `trn_sum_from_iss` is unverified (FINDINGS #11, open question #2).
  This is a contract decision for all four owners, not something I set
  unilaterally; until resolved, the replay producer should carry `purchase_sum`
  as the transaction value and leave `items.amount` provisional.

## Open questions back to the EDA owner (Amya)

1. **Window sizes.** With only 117 days of history and partial first/last months
   (#16, #19), are 30/60/90-day windows the right granularity, or should v1 also
   expose weekly counts? I defaulted to 30/60/90 for M0 parity; happy to add
   weekly if the EDA suggests more signal there.
2. **Recency under the provisional anchor (#16).** Every client's last purchase
   is within ~22 days of the anchor, so `days_since_last_tx` has little spread
   until the real cutoff is known. Confirm we accept near-constant recency in v1.
3. **28 shared transaction ids (#7).** Accepted keying by (client_id,
   transaction_id); flagging in case the source is later corrected.

## Sign-off

FINDINGS.md is **accepted** as the basis for cleaning and feature rules, subject
to blockers B1–B3. The accepted decisions are already reflected in the TEA-18
feature library and quality checks; silver and the gold builders will apply the
silver-scoped rules (#6, #8, #9, #10, #11) once bronze and the campaign date are
available.

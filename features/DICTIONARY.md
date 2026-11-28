# Data dictionary: gold features (X5 uplift)

Every column produced by the feature library (`features/aggregates.py`), with its
formula and unit. This is the contract `gold.client_features` writes against; one
row per client, plus `client_id` and `feature_version`.

## Conventions

- **Cutoff.** The campaign date is not published (EDA finding #1), so every
  time-based feature takes a `cutoff`. Only transactions with `tx_ts` **strictly
  before** the cutoff contribute. Recency and the 30/60/90-day windows are
  measured backward from the cutoff, not from "now".
- **Inputs are transaction-level.** `features/silver.py` collapses product rows
  to one row per `(client_id, transaction_id)` before these functions run:
  `purchase_sum` and the points columns are taken once per transaction (EDA #6),
  points-spent columns are stored as absolute magnitudes (EDA #10), and all IDs
  are strings (EDA #13).
- **No purchases before the cutoff.** Such a client gets zero counts/sums,
  `has_purchases = 0`, and `NaN` for recency/tenure and the undefined statistics
  (see each row). This matches the work-plan rule "clients with no purchases get
  zero counts, a has_purchases flag and maximum recency".
- **Units.** "currency" is the dataset's `purchase_sum` unit (unspecified by the
  source); "points" are loyalty points; "days" are calendar days as floats.

`feature_version` is currently **v1** (`FEATURE_VERSION` in `features/aggregates.py`).

## Spend

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `total_spend` | sum of `purchase_sum` over the client's pre-cutoff transactions | currency | 0 |
| `mean_tx_spend` | `total_spend / n_transactions` | currency | 0 |
| `max_tx_spend` | max `purchase_sum` | currency | 0 |
| `std_tx_spend` | population std (ddof=0) of `purchase_sum`; 0 for a single transaction | currency | NaN |
| `median_tx_spend` | median `purchase_sum` | currency | 0 |

## Frequency

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `n_transactions` | count of pre-cutoff transactions | count | 0 |
| `n_active_days` | number of distinct calendar days with a transaction | count | 0 |
| `tx_per_active_day` | `n_transactions / n_active_days` | ratio | 0 |
| `avg_days_between_tx` | (last − first) in days / (`n_transactions` − 1); NaN for a single transaction | days | NaN |

## Recency and tenure

All measured against the cutoff.

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `days_since_last_tx` | (cutoff − last transaction timestamp) | days | NaN |
| `days_since_first_tx` | (cutoff − first transaction timestamp) | days | NaN |
| `tenure_days` | (last − first transaction timestamp) | days | NaN |

## Windowed spend and counts

Each window is `[cutoff − N days, cutoff)`.

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `spend_30d` | sum of `purchase_sum` in the last 30 days before the cutoff | currency | 0 |
| `spend_60d` | sum of `purchase_sum` in the last 60 days | currency | 0 |
| `spend_90d` | sum of `purchase_sum` in the last 90 days | currency | 0 |
| `tx_count_30d` | transaction count in the last 30 days | count | 0 |
| `tx_count_60d` | transaction count in the last 60 days | count | 0 |
| `tx_count_90d` | transaction count in the last 90 days | count | 0 |

## Points

Spent columns are magnitudes (EDA #10 found them stored non-positive; silver takes
the absolute value).

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `regular_points_received` | sum of `regular_points_received` | points | 0 |
| `express_points_received` | sum of `express_points_received` | points | 0 |
| `regular_points_spent` | sum of `regular_points_spent` (magnitude) | points | 0 |
| `express_points_spent` | sum of `express_points_spent` (magnitude) | points | 0 |
| `net_points` | (received regular + express) − (spent regular + express) | points | 0 |
| `any_points_spent` | 1 if total points spent > 0 else 0 | flag (0/1) | 0 |

## Stores

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `n_unique_stores` | number of distinct `store_id` | count | 0 |
| `top_store_share` | transactions at the most-used store / `n_transactions` | ratio | 0 |
| `n_tx_at_top_store` | transaction count at the most-used store | count | 0 |

## Items

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `total_items` | sum of `n_items` across transactions | count | 0 |
| `mean_items_per_tx` | `total_items / n_transactions` | ratio | 0 |
| `zero_qty_tx_count` | transactions with `product_quantity == 0` (kept and flagged, EDA #9) | count | 0 |

## Profile and flags

From the `clients` table; independent of the cutoff.

| Column | Formula | Unit | Empty-client value |
|---|---|---|---|
| `age` | client age if in [16, 100], else NaN (EDA #4) | years | from profile |
| `age_invalid` | 1 if age missing or outside [16, 100] else 0 (EDA #4) | flag (0/1) | from profile |
| `gender_F` | 1 if gender == "F" else 0 | flag (0/1) | from profile |
| `gender_M` | 1 if gender == "M" else 0 | flag (0/1) | from profile |
| `gender_U` | 1 if gender == "U" else 0 (kept as a real level, EDA #5) | flag (0/1) | from profile |
| `has_purchases` | 1 if the client has ≥ 1 pre-cutoff transaction else 0 | flag (0/1) | 0 |

## Open items

- `age`/`age_invalid`/gender flags depend on the `clients` profile passed in by
  the batch builder; missing profile means `age_invalid = 1` and all gender flags
  0.
- Window boundaries and the recency anchor are provisional until the real
  `campaign.cutoff` is set in `configs/x5/x5.yaml` (EDA #1, #16).
- `first_issue_date` / `first_redeem_date` are deliberately **not** used as
  features (EDA #2, #3: leakage-suspect and noisy).

"""Pure per-client feature functions for the X5 uplift gold tables.

Contract (from the Semester 1 work plan, "Shared contracts" and the Prathmesh
tasks): ``gold.client_features`` is one row per client holding 36 aggregate
features (spend, frequency, recency, 30/60/90-day windows, points, stores, plus
profile features) alongside ``client_id`` and ``feature_version``.

Design rules this module follows:

- **Pure functions, no I/O.** Everything operates on an in-memory transaction
  table and a profile mapping, so the Airflow batch task and the streaming
  consumer call the same code and must produce identical values.
- **Cutoff is always a parameter** (work-plan no-leakage rule and EDA finding
  #1). Only transactions strictly before ``cutoff`` contribute to features.
  The campaign date is unpublished, so callers pass the agreed anchor; nothing
  is hard-coded here.
- **Transaction-level inputs.** The caller (``features.silver``) has already
  collapsed product rows to one row per (client_id, transaction_id) with
  ``purchase_sum`` and the points columns taken once per transaction (EDA
  finding #6), points-spent stored as absolute values (#10), and IDs as strings
  (#13). This module does not re-aggregate product rows.

The transaction DataFrame passed in uses the ``silver.transactions`` schema:
``client_id, transaction_id, tx_ts, purchase_sum, store_id, n_items,
product_quantity, trn_sum_from_iss, regular_points_received,
express_points_received, regular_points_spent, express_points_spent``.
``tx_ts`` must be timezone-naive ``datetime64[ns]``; points-spent columns are
non-negative magnitudes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

# Feature schema version. Bump when the set or definition of features changes;
# it is written into gold as ``feature_version`` and into the data dictionary.
FEATURE_VERSION = "v1"

# The 36 feature names, in a fixed order. Any change here is a contract change
# and must be mirrored in features/DICTIONARY.md and the quality checks.
FEATURE_NAMES: list[str] = [
    # Spend (5)
    "total_spend",
    "mean_tx_spend",
    "max_tx_spend",
    "std_tx_spend",
    "median_tx_spend",
    # Frequency (4)
    "n_transactions",
    "n_active_days",
    "tx_per_active_day",
    "avg_days_between_tx",
    # Recency / tenure (3)
    "days_since_last_tx",
    "days_since_first_tx",
    "tenure_days",
    # Windowed spend and counts, relative to the cutoff (6)
    "spend_30d",
    "spend_60d",
    "spend_90d",
    "tx_count_30d",
    "tx_count_60d",
    "tx_count_90d",
    # Points (6); spent values are magnitudes (EDA #10)
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "net_points",
    "any_points_spent",
    # Stores (3)
    "n_unique_stores",
    "top_store_share",
    "n_tx_at_top_store",
    # Items (3)
    "total_items",
    "mean_items_per_tx",
    "zero_qty_tx_count",
    # Profile and flags (6)
    "age",
    "age_invalid",
    "gender_F",
    "gender_M",
    "gender_U",
    "has_purchases",
]

assert len(FEATURE_NAMES) == 36, "the gold contract fixes the feature count at 36"

# Columns the transaction frame must provide.
_REQUIRED_TX_COLUMNS = (
    "tx_ts",
    "purchase_sum",
    "store_id",
    "n_items",
    "product_quantity",
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
)

# EDA finding #4: ages outside this inclusive range are treated as invalid.
AGE_MIN = 16
AGE_MAX = 100

_SECONDS_PER_DAY = 86_400.0


def _to_timestamp(cutoff: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Normalise the cutoff to a timezone-naive pandas Timestamp."""
    ts = pd.Timestamp(cutoff)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None) if ts.tz is not None else ts.tz_localize(None)
    return ts


def _profile_features(profile: Mapping | None) -> dict:
    """Age (with validity flag) and one-hot gender from the client profile.

    EDA finding #4: ages outside [16, 100] become NaN with ``age_invalid`` set.
    EDA finding #5: gender ``U`` is a real level, kept as its own one-hot column;
    anything else (including missing) leaves all three gender flags at 0.
    """
    profile = profile or {}

    raw_age = profile.get("age")
    age: float
    age_invalid: int
    if raw_age is None or (isinstance(raw_age, float) and math.isnan(raw_age)):
        age = float("nan")
        age_invalid = 1
    else:
        age_val = float(raw_age)
        if AGE_MIN <= age_val <= AGE_MAX:
            age = age_val
            age_invalid = 0
        else:
            age = float("nan")
            age_invalid = 1

    gender = profile.get("gender")
    gender = gender.strip().upper() if isinstance(gender, str) else gender
    return {
        "age": age,
        "age_invalid": age_invalid,
        "gender_F": int(gender == "F"),
        "gender_M": int(gender == "M"),
        "gender_U": int(gender == "U"),
    }


def empty_client_features(profile: Mapping | None = None) -> dict:
    """Feature row for a client with no purchases before the cutoff.

    Work-plan rule: clients with no purchases get zero counts, a
    ``has_purchases`` flag of 0, and maximum recency. "Maximum recency" has no
    finite reference without a purchase, so the recency/tenure fields are NaN
    and callers/consumers treat them with the missing-value handling the model
    expects. Spend, count, points, store and item features are all 0.
    """
    row = dict.fromkeys(FEATURE_NAMES, 0.0)

    # Counts and sums are a true zero for a client with no history.
    row["std_tx_spend"] = float("nan")
    row["median_tx_spend"] = 0.0
    row["tx_per_active_day"] = 0.0
    row["avg_days_between_tx"] = float("nan")

    # Recency and tenure are undefined without any purchase.
    row["days_since_last_tx"] = float("nan")
    row["days_since_first_tx"] = float("nan")
    row["tenure_days"] = float("nan")

    row["top_store_share"] = 0.0
    row["mean_items_per_tx"] = 0.0

    row["any_points_spent"] = 0
    row["has_purchases"] = 0

    row.update(_profile_features(profile))
    return row


def client_features_from_tx(
    transactions: pd.DataFrame,
    *,
    cutoff: datetime | pd.Timestamp | str,
    profile: Mapping | None = None,
) -> dict:
    """Compute the 36 gold features for one client.

    Parameters
    ----------
    transactions:
        One client's transactions in the ``silver.transactions`` schema, one row
        per (client_id, transaction_id). May be empty.
    cutoff:
        Only transactions with ``tx_ts`` strictly before this moment are used
        (no-leakage rule, EDA #1). Recency is measured from this moment too, so
        batch and stream agree on the same anchor.
    profile:
        Optional mapping with ``age`` and ``gender`` from the ``clients`` table.

    Returns
    -------
    dict
        A mapping with exactly the keys in :data:`FEATURE_NAMES`.
    """
    cutoff_ts = _to_timestamp(cutoff)

    if transactions is None or len(transactions) == 0:
        return empty_client_features(profile)

    missing = [c for c in _REQUIRED_TX_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")

    tx = transactions.copy()
    tx["tx_ts"] = pd.to_datetime(tx["tx_ts"])
    # Apply the cutoff: strictly before, so a purchase exactly at the campaign
    # moment never leaks into pre-campaign features.
    tx = tx.loc[tx["tx_ts"] < cutoff_ts]

    if len(tx) == 0:
        return empty_client_features(profile)

    tx = tx.sort_values("tx_ts")
    spend = tx["purchase_sum"].astype(float)
    n_tx = int(len(tx))

    row: dict = {}

    # --- Spend ---
    row["total_spend"] = float(spend.sum())
    row["mean_tx_spend"] = float(spend.mean())
    row["max_tx_spend"] = float(spend.max())
    # Population std is 0 for a single transaction; ddof=0 keeps it finite.
    row["std_tx_spend"] = float(spend.std(ddof=0))
    row["median_tx_spend"] = float(spend.median())

    # --- Frequency ---
    days = tx["tx_ts"].dt.normalize()
    n_active_days = int(days.nunique())
    row["n_transactions"] = float(n_tx)
    row["n_active_days"] = float(n_active_days)
    row["tx_per_active_day"] = float(n_tx / n_active_days) if n_active_days else 0.0
    if n_tx >= 2:
        span_days = (tx["tx_ts"].iloc[-1] - tx["tx_ts"].iloc[0]).total_seconds() / _SECONDS_PER_DAY
        row["avg_days_between_tx"] = float(span_days / (n_tx - 1))
    else:
        row["avg_days_between_tx"] = float("nan")

    # --- Recency / tenure, measured against the cutoff ---
    last_ts = tx["tx_ts"].iloc[-1]
    first_ts = tx["tx_ts"].iloc[0]
    row["days_since_last_tx"] = float((cutoff_ts - last_ts).total_seconds() / _SECONDS_PER_DAY)
    row["days_since_first_tx"] = float((cutoff_ts - first_ts).total_seconds() / _SECONDS_PER_DAY)
    row["tenure_days"] = float((last_ts - first_ts).total_seconds() / _SECONDS_PER_DAY)

    # --- Windowed spend and counts relative to the cutoff ---
    for window in (30, 60, 90):
        lower = cutoff_ts - pd.Timedelta(days=window)
        in_window = tx["tx_ts"] >= lower
        row[f"spend_{window}d"] = float(spend.loc[in_window].sum())
        row[f"tx_count_{window}d"] = float(int(in_window.sum()))

    # --- Points (spent columns are already magnitudes, EDA #10) ---
    reg_recv = float(tx["regular_points_received"].astype(float).sum())
    exp_recv = float(tx["express_points_received"].astype(float).sum())
    reg_spent = float(tx["regular_points_spent"].astype(float).sum())
    exp_spent = float(tx["express_points_spent"].astype(float).sum())
    row["regular_points_received"] = reg_recv
    row["express_points_received"] = exp_recv
    row["regular_points_spent"] = reg_spent
    row["express_points_spent"] = exp_spent
    row["net_points"] = (reg_recv + exp_recv) - (reg_spent + exp_spent)
    row["any_points_spent"] = int((reg_spent + exp_spent) > 0)

    # --- Stores ---
    store_counts = tx["store_id"].astype("string").value_counts()
    row["n_unique_stores"] = float(int(store_counts.size))
    top_count = int(store_counts.iloc[0])
    row["n_tx_at_top_store"] = float(top_count)
    row["top_store_share"] = float(top_count / n_tx)

    # --- Items ---
    items = tx["n_items"].astype(float)
    row["total_items"] = float(items.sum())
    row["mean_items_per_tx"] = float(items.mean())
    # EDA #9: zero-quantity rows are kept and flagged, not dropped.
    row["zero_qty_tx_count"] = float(int((tx["product_quantity"].astype(float) == 0).sum()))

    # --- Profile and presence flag ---
    row.update(_profile_features(profile))
    row["has_purchases"] = 1

    # Guarantee the exact contract: all 36 keys, nothing extra.
    assert set(row.keys()) == set(FEATURE_NAMES), "feature row must match FEATURE_NAMES exactly"
    return {name: row[name] for name in FEATURE_NAMES}

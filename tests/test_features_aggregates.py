"""Unit tests for the pure feature library (features.aggregates)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from features.aggregates import (
    FEATURE_NAMES,
    client_features_from_tx,
    empty_client_features,
)

CUTOFF = pd.Timestamp("2019-03-19 00:00:00")


def _tx(rows: list[dict]) -> pd.DataFrame:
    """Build a silver-schema transaction frame from partial rows with defaults."""
    defaults = {
        "client_id": "c1",
        "transaction_id": "t1",
        "tx_ts": CUTOFF - pd.Timedelta(days=1),
        "purchase_sum": 100.0,
        "store_id": "s1",
        "n_items": 1,
        "product_quantity": 1.0,
        "trn_sum_from_iss": 100.0,
        "regular_points_received": 0.0,
        "express_points_received": 0.0,
        "regular_points_spent": 0.0,
        "express_points_spent": 0.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_feature_names_are_exactly_36_unique():
    assert len(FEATURE_NAMES) == 36
    assert len(set(FEATURE_NAMES)) == 36


def test_output_keys_match_contract_exactly():
    tx = _tx([{"transaction_id": "t1"}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert list(feats.keys()) == FEATURE_NAMES  # order is stable


def test_empty_client_has_zero_counts_and_flag_off():
    feats = empty_client_features(profile={"age": 40, "gender": "F"})
    assert feats["has_purchases"] == 0
    assert feats["n_transactions"] == 0.0
    assert feats["total_spend"] == 0.0
    assert feats["n_unique_stores"] == 0.0
    # Recency/tenure undefined without any purchase.
    assert math.isnan(feats["days_since_last_tx"])
    assert math.isnan(feats["tenure_days"])
    # Profile still flows through.
    assert feats["age"] == 40.0
    assert feats["gender_F"] == 1
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_no_transactions_before_cutoff_falls_back_to_empty():
    # Only transaction is after the cutoff, so nothing counts.
    tx = _tx([{"tx_ts": CUTOFF + pd.Timedelta(days=1)}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF, profile={"age": 30, "gender": "M"})
    assert feats["has_purchases"] == 0
    assert feats["n_transactions"] == 0.0
    assert feats["gender_M"] == 1


def test_cutoff_is_strict_and_prevents_leakage():
    # One transaction exactly at the cutoff must be excluded (strictly before).
    tx = _tx(
        [
            {"transaction_id": "t_before", "tx_ts": CUTOFF - pd.Timedelta(hours=1), "purchase_sum": 50.0},
            {"transaction_id": "t_at", "tx_ts": CUTOFF, "purchase_sum": 999.0},
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["n_transactions"] == 1.0
    assert feats["total_spend"] == 50.0  # the 999 at the cutoff is excluded


def test_spend_statistics_known_values():
    tx = _tx(
        [
            {"transaction_id": "a", "tx_ts": CUTOFF - pd.Timedelta(days=10), "purchase_sum": 100.0},
            {"transaction_id": "b", "tx_ts": CUTOFF - pd.Timedelta(days=5), "purchase_sum": 300.0},
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["total_spend"] == 400.0
    assert feats["mean_tx_spend"] == 200.0
    assert feats["max_tx_spend"] == 300.0
    assert feats["median_tx_spend"] == 200.0
    # population std of {100, 300} = 100
    assert feats["std_tx_spend"] == pytest.approx(100.0)


def test_single_transaction_std_is_zero_and_gap_is_nan():
    tx = _tx([{"transaction_id": "only", "purchase_sum": 123.0}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["std_tx_spend"] == 0.0
    assert math.isnan(feats["avg_days_between_tx"])


def test_recency_and_tenure_measured_from_cutoff():
    tx = _tx(
        [
            {"transaction_id": "first", "tx_ts": CUTOFF - pd.Timedelta(days=20)},
            {"transaction_id": "last", "tx_ts": CUTOFF - pd.Timedelta(days=4)},
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["days_since_last_tx"] == pytest.approx(4.0)
    assert feats["days_since_first_tx"] == pytest.approx(20.0)
    assert feats["tenure_days"] == pytest.approx(16.0)


def test_windowed_spend_and_counts():
    tx = _tx(
        [
            {"transaction_id": "d5", "tx_ts": CUTOFF - pd.Timedelta(days=5), "purchase_sum": 10.0},
            {"transaction_id": "d45", "tx_ts": CUTOFF - pd.Timedelta(days=45), "purchase_sum": 20.0},
            {"transaction_id": "d75", "tx_ts": CUTOFF - pd.Timedelta(days=75), "purchase_sum": 40.0},
            {"transaction_id": "d120", "tx_ts": CUTOFF - pd.Timedelta(days=120), "purchase_sum": 80.0},
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["spend_30d"] == 10.0
    assert feats["spend_60d"] == 30.0
    assert feats["spend_90d"] == 70.0
    assert feats["tx_count_30d"] == 1.0
    assert feats["tx_count_60d"] == 2.0
    assert feats["tx_count_90d"] == 3.0
    assert feats["total_spend"] == 150.0  # the 120-day-old one still counts in totals


def test_points_absolute_values_and_net():
    tx = _tx(
        [
            {
                "transaction_id": "p",
                "regular_points_received": 10.0,
                "express_points_received": 2.0,
                "regular_points_spent": 4.0,  # magnitude, already absolute (EDA #10)
                "express_points_spent": 1.0,
            }
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["regular_points_received"] == 10.0
    assert feats["regular_points_spent"] == 4.0
    assert feats["net_points"] == pytest.approx((10.0 + 2.0) - (4.0 + 1.0))
    assert feats["any_points_spent"] == 1


def test_any_points_spent_zero_when_none_spent():
    tx = _tx([{"transaction_id": "p", "regular_points_received": 5.0}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["any_points_spent"] == 0


def test_store_loyalty_features():
    tx = _tx(
        [
            {"transaction_id": "a", "store_id": "s1"},
            {"transaction_id": "b", "store_id": "s1"},
            {"transaction_id": "c", "store_id": "s1"},
            {"transaction_id": "d", "store_id": "s2"},
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["n_unique_stores"] == 2.0
    assert feats["n_tx_at_top_store"] == 3.0
    assert feats["top_store_share"] == pytest.approx(0.75)


def test_item_features_and_zero_quantity_flag():
    tx = _tx(
        [
            {"transaction_id": "a", "n_items": 3, "product_quantity": 3.0},
            {"transaction_id": "b", "n_items": 1, "product_quantity": 0.0},  # zero-qty, kept
        ]
    )
    feats = client_features_from_tx(tx, cutoff=CUTOFF)
    assert feats["total_items"] == 4.0
    assert feats["mean_items_per_tx"] == 2.0
    assert feats["zero_qty_tx_count"] == 1.0


def test_age_outside_range_is_invalid_and_nan():
    tx = _tx([{"transaction_id": "a"}])
    old = client_features_from_tx(tx, cutoff=CUTOFF, profile={"age": 150, "gender": "U"})
    assert old["age_invalid"] == 1
    assert math.isnan(old["age"])
    assert old["gender_U"] == 1

    young = client_features_from_tx(tx, cutoff=CUTOFF, profile={"age": 10, "gender": "F"})
    assert young["age_invalid"] == 1
    assert math.isnan(young["age"])


def test_age_within_range_is_valid():
    tx = _tx([{"transaction_id": "a"}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF, profile={"age": 16, "gender": "M"})
    assert feats["age_invalid"] == 0
    assert feats["age"] == 16.0
    feats100 = client_features_from_tx(tx, cutoff=CUTOFF, profile={"age": 100})
    assert feats100["age_invalid"] == 0
    assert feats100["age"] == 100.0


def test_missing_profile_leaves_gender_zeroed_and_age_invalid():
    tx = _tx([{"transaction_id": "a"}])
    feats = client_features_from_tx(tx, cutoff=CUTOFF, profile=None)
    assert feats["gender_F"] == 0
    assert feats["gender_M"] == 0
    assert feats["gender_U"] == 0
    assert feats["age_invalid"] == 1


def test_missing_required_column_raises():
    tx = _tx([{"transaction_id": "a"}]).drop(columns=["store_id"])
    with pytest.raises(ValueError, match="store_id"):
        client_features_from_tx(tx, cutoff=CUTOFF)


def test_string_cutoff_accepted():
    tx = _tx([{"transaction_id": "a", "tx_ts": pd.Timestamp("2019-03-10")}])
    feats = client_features_from_tx(tx, cutoff="2019-03-19")
    assert feats["has_purchases"] == 1

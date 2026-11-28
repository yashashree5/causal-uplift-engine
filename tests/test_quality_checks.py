"""Unit tests for the gold data-quality checks (quality.checks)."""

from __future__ import annotations

import pandas as pd
import pytest

from quality.checks import (
    QualityCheckError,
    assert_gold_quality,
    check_fold_treatment_balance,
    check_labeled_clients_present,
    check_no_null_client_id,
    check_no_purchase_after_cutoff,
    check_nonnegative_purchase_sum,
    check_row_count_matches_bronze,
    run_gold_checks,
)

CUTOFF = pd.Timestamp("2019-03-19 00:00:00")


def _silver(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n)],
            "transaction_id": [f"t{i}" for i in range(n)],
            "tx_ts": [CUTOFF - pd.Timedelta(days=i + 1) for i in range(n)],
            "purchase_sum": [10.0 * (i + 1) for i in range(n)],
        }
    )


def _labels(fold_shares: dict[int, float] | None = None) -> pd.DataFrame:
    # Default: four folds, 50% treated each.
    rows = []
    cid = 0
    for fold in range(4):
        for t in (0, 1):
            for _ in range(10):
                rows.append({"client_id": f"c{cid}", "treatment_flg": t, "target": 0, "fold": fold})
                cid += 1
    return pd.DataFrame(rows)


def test_row_count_matches_bronze_pass_and_fail():
    s = _silver(5)
    assert check_row_count_matches_bronze(s, 5).passed is True
    assert check_row_count_matches_bronze(s, 6).passed is False


def test_no_null_client_id_detects_null_and_empty():
    good = _silver(2)
    assert check_no_null_client_id(good).passed is True

    bad_null = good.copy()
    bad_null.loc[0, "client_id"] = None
    assert check_no_null_client_id(bad_null).passed is False

    bad_empty = good.copy()
    bad_empty.loc[0, "client_id"] = ""
    assert check_no_null_client_id(bad_empty).passed is False


def test_nonnegative_purchase_sum():
    good = _silver(3)
    assert check_nonnegative_purchase_sum(good).passed is True
    bad = good.copy()
    bad.loc[0, "purchase_sum"] = -1.0
    assert check_nonnegative_purchase_sum(bad).passed is False


def test_labeled_clients_present():
    labels = pd.DataFrame({"client_id": ["c0", "c1", "c2"]})
    gold_full = pd.DataFrame({"client_id": ["c0", "c1", "c2", "c3"]})
    assert check_labeled_clients_present(labels, gold_full).passed is True

    gold_missing = pd.DataFrame({"client_id": ["c0", "c1"]})
    res = check_labeled_clients_present(labels, gold_missing)
    assert res.passed is False
    assert "c2" in res.detail


def test_no_purchase_after_cutoff_pass_fail_and_skip():
    good = _silver(3)  # all before the cutoff
    assert check_no_purchase_after_cutoff(good, CUTOFF).passed is True

    bad = good.copy()
    bad.loc[0, "tx_ts"] = CUTOFF + pd.Timedelta(days=1)
    assert check_no_purchase_after_cutoff(bad, CUTOFF).passed is False

    # A transaction exactly at the cutoff counts as leakage (on-or-after).
    at = good.copy()
    at.loc[0, "tx_ts"] = CUTOFF
    assert check_no_purchase_after_cutoff(at, CUTOFF).passed is False

    skipped = check_no_purchase_after_cutoff(good, None)
    assert skipped.skipped is True
    assert skipped.failed is False


def test_fold_treatment_balance_pass():
    assert check_fold_treatment_balance(_labels()).passed is True


def test_fold_treatment_balance_fail_when_fold_skewed():
    labels = _labels()
    # Flip all treated clients in fold 0 to control -> share 0.0, out of bounds.
    mask = labels["fold"] == 0
    labels.loc[mask, "treatment_flg"] = 0
    res = check_fold_treatment_balance(labels)
    assert res.passed is False
    assert "0" in res.detail


def test_run_and_assert_all_pass():
    silver = _silver(80)
    labels = _labels()
    gold = pd.DataFrame({"client_id": labels["client_id"]})
    results = run_gold_checks(
        silver=silver,
        gold_features=gold,
        labels=labels,
        bronze_row_count=80,
        cutoff=CUTOFF,
    )
    # Nothing should raise.
    assert_gold_quality(results)
    assert all(r.passed is not False for r in results)


def test_assert_raises_on_failure():
    silver = _silver(3)
    labels = _labels()
    gold = pd.DataFrame({"client_id": labels["client_id"]})
    results = run_gold_checks(
        silver=silver,
        gold_features=gold,
        labels=labels,
        bronze_row_count=999,  # wrong on purpose
        cutoff=CUTOFF,
    )
    with pytest.raises(QualityCheckError, match="row_count_matches_bronze"):
        assert_gold_quality(results)


def test_skipped_cutoff_does_not_raise():
    silver = _silver(80)
    labels = _labels()
    gold = pd.DataFrame({"client_id": labels["client_id"]})
    results = run_gold_checks(
        silver=silver,
        gold_features=gold,
        labels=labels,
        bronze_row_count=80,
        cutoff=None,  # campaign date unknown
    )
    assert_gold_quality(results)  # must not raise on a skipped check
    assert any(r.skipped for r in results)

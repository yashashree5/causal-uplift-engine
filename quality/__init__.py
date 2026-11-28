"""Data-quality checks for the X5 gold tables (Prathmesh workstream).

The checks are pure functions over in-memory tables. Each returns a
:class:`CheckResult`; :func:`run_gold_checks` aggregates them and
:func:`assert_gold_quality` raises loudly on any failure so the Airflow DAG
fails instead of shipping bad gold data.
"""

from quality.checks import (
    CheckResult,
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

__all__ = [
    "CheckResult",
    "QualityCheckError",
    "assert_gold_quality",
    "check_fold_treatment_balance",
    "check_labeled_clients_present",
    "check_no_null_client_id",
    "check_no_purchase_after_cutoff",
    "check_nonnegative_purchase_sum",
    "check_row_count_matches_bronze",
    "run_gold_checks",
]

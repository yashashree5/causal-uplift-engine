"""Pure data-quality checks for the X5 silver/gold tables.

These implement the checks named in the work plan (Prathmesh, "Data-quality
checks"):

- row counts vs bronze,
- no null ``client_id``,
- ``purchase_sum`` >= 0,
- every labeled client present in gold,
- no purchase dated on or after the campaign cutoff (no-leakage rule),
- treatment share within [0.45, 0.55] in every fold.

Each check is a pure function that returns a :class:`CheckResult`; nothing reads
files or the network, so the same functions run in a notebook, a unit test, and
inside Tejas's Airflow DAG. :func:`assert_gold_quality` turns any failure into a
raised :class:`QualityCheckError` so the pipeline fails loudly.

Note on the cutoff: the campaign date is unpublished (EDA finding #1). The
leakage check therefore takes the cutoff as an argument and returns a skipped
(non-failing but non-passing) result when it is ``None``, rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# Work-plan bound on treatment share per fold; the experiment is ~50/50
# (EDA #14 reports 49.98% treated).
FOLD_TREATMENT_SHARE_MIN = 0.45
FOLD_TREATMENT_SHARE_MAX = 0.55


class QualityCheckError(AssertionError):
    """Raised when one or more gold data-quality checks fail."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single data-quality check.

    ``passed`` is ``None`` for a skipped check (for example the leakage check
    when no cutoff is configured). ``passed`` is ``True``/``False`` otherwise.
    """

    name: str
    passed: bool | None
    detail: str

    @property
    def skipped(self) -> bool:
        return self.passed is None

    @property
    def failed(self) -> bool:
        return self.passed is False


def check_row_count_matches_bronze(silver: pd.DataFrame, bronze_row_count: int) -> CheckResult:
    """Silver transaction count must equal the bronze transaction count.

    Bronze has one row per transaction after Tejas's typed conversion, and silver
    keeps one row per (client_id, transaction_id), so the counts must match; a
    mismatch means rows were silently dropped or duplicated.
    """
    actual = int(len(silver))
    ok = actual == int(bronze_row_count)
    return CheckResult(
        name="row_count_matches_bronze",
        passed=ok,
        detail=f"silver rows={actual}, bronze rows={int(bronze_row_count)}",
    )


def check_no_null_client_id(table: pd.DataFrame, column: str = "client_id") -> CheckResult:
    """No null/empty ``client_id`` in the table."""
    if column not in table.columns:
        return CheckResult("no_null_client_id", False, f"missing column {column!r}")
    col = table[column]
    n_null = int(col.isna().sum())
    # IDs are strings (EDA #13); an empty string is as bad as a null.
    n_empty = int((col.astype("string").str.len() == 0).fillna(False).sum())
    ok = n_null == 0 and n_empty == 0
    return CheckResult(
        name="no_null_client_id",
        passed=ok,
        detail=f"null={n_null}, empty={n_empty}",
    )


def check_nonnegative_purchase_sum(
    silver: pd.DataFrame, column: str = "purchase_sum"
) -> CheckResult:
    """``purchase_sum`` must be >= 0 (EDA #9 found no negative sums)."""
    if column not in silver.columns:
        return CheckResult("nonnegative_purchase_sum", False, f"missing column {column!r}")
    vals = silver[column].astype(float)
    n_negative = int((vals < 0).sum())
    n_null = int(vals.isna().sum())
    ok = n_negative == 0 and n_null == 0
    return CheckResult(
        name="nonnegative_purchase_sum",
        passed=ok,
        detail=f"negative={n_negative}, null={n_null}",
    )


def check_labeled_clients_present(
    labels: pd.DataFrame,
    gold_features: pd.DataFrame,
    client_column: str = "client_id",
) -> CheckResult:
    """Every labeled client must have a row in gold client features."""
    if client_column not in labels.columns or client_column not in gold_features.columns:
        return CheckResult("labeled_clients_present", False, f"missing {client_column!r}")
    labeled = set(labels[client_column].astype("string"))
    present = set(gold_features[client_column].astype("string"))
    missing = labeled - present
    ok = len(missing) == 0
    sample = sorted(missing)[:5]
    return CheckResult(
        name="labeled_clients_present",
        passed=ok,
        detail=f"labeled={len(labeled)}, missing={len(missing)}, sample={sample}",
    )


def check_no_purchase_after_cutoff(
    silver: pd.DataFrame,
    cutoff: datetime | pd.Timestamp | str | None,
    ts_column: str = "tx_ts",
) -> CheckResult:
    """No transaction may be dated on or after the campaign cutoff.

    Returns a skipped result when ``cutoff`` is ``None`` (campaign date not yet
    known, EDA #1) so the pipeline does not silently pass a check it cannot run.
    """
    if cutoff is None:
        return CheckResult(
            name="no_purchase_after_cutoff",
            passed=None,
            detail="skipped: campaign cutoff is not set (EDA #1)",
        )
    if ts_column not in silver.columns:
        return CheckResult("no_purchase_after_cutoff", False, f"missing column {ts_column!r}")
    cutoff_ts = pd.Timestamp(cutoff)
    ts = pd.to_datetime(silver[ts_column])
    n_after = int((ts >= cutoff_ts).sum())
    ok = n_after == 0
    return CheckResult(
        name="no_purchase_after_cutoff",
        passed=ok,
        detail=f"cutoff={cutoff_ts.isoformat()}, on_or_after={n_after}",
    )


def check_fold_treatment_balance(
    labels: pd.DataFrame,
    treatment_column: str = "treatment_flg",
    fold_column: str = "fold",
) -> CheckResult:
    """Treatment share must be within [0.45, 0.55] in every fold."""
    for col in (treatment_column, fold_column):
        if col not in labels.columns:
            return CheckResult("fold_treatment_balance", False, f"missing column {col!r}")
    shares = labels.groupby(fold_column)[treatment_column].mean()
    offending = {
        int(fold): round(float(share), 4)
        for fold, share in shares.items()
        if not (FOLD_TREATMENT_SHARE_MIN <= float(share) <= FOLD_TREATMENT_SHARE_MAX)
    }
    ok = len(offending) == 0
    return CheckResult(
        name="fold_treatment_balance",
        passed=ok,
        detail=(
            f"bounds=[{FOLD_TREATMENT_SHARE_MIN}, {FOLD_TREATMENT_SHARE_MAX}], "
            f"per_fold_share={ {int(f): round(float(s), 4) for f, s in shares.items()} }, "
            f"offending={offending}"
        ),
    )


def run_gold_checks(
    *,
    silver: pd.DataFrame,
    gold_features: pd.DataFrame,
    labels: pd.DataFrame,
    bronze_row_count: int,
    cutoff: datetime | pd.Timestamp | str | None,
) -> list[CheckResult]:
    """Run every gold check and return the results in a stable order."""
    return [
        check_row_count_matches_bronze(silver, bronze_row_count),
        check_no_null_client_id(silver),
        check_no_null_client_id(gold_features),
        check_nonnegative_purchase_sum(silver),
        check_labeled_clients_present(labels, gold_features),
        check_no_purchase_after_cutoff(silver, cutoff),
        check_fold_treatment_balance(labels),
    ]


def assert_gold_quality(results: list[CheckResult]) -> None:
    """Raise :class:`QualityCheckError` if any check failed.

    Skipped checks (``passed is None``) do not raise, but their names are
    included in the error message when something else fails so the operator sees
    the full picture.
    """
    failures = [r for r in results if r.failed]
    if failures:
        lines = [f"  - {r.name}: {r.detail}" for r in failures]
        skipped = [r.name for r in results if r.skipped]
        msg = "gold data-quality checks failed:\n" + "\n".join(lines)
        if skipped:
            msg += f"\n(skipped: {', '.join(skipped)})"
        raise QualityCheckError(msg)

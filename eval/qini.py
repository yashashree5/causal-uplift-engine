"""Qini curve and score, ported line-for-line from the CausalPFN authors' code.

Source: https://github.com/vdblm/CausalPFN/blob/7da4afa/src/causalpfn/evaluation.py (Apache-2.0).
Keep this identical to the reference so our numbers are comparable with the paper's tables.
tests/test_qini.py checks it against the reference when `causalpfn` is installed.
"""

from __future__ import annotations

import numpy as np


def get_qini_curve(
    rct_treatments: np.ndarray,
    rct_outcomes: np.ndarray,
    estimated_cate: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Return (qini_curve, qini_score). Units are ranked by estimated_cate, highest first."""
    sorted_indices = np.argsort(estimated_cate)[::-1]

    t = rct_treatments[sorted_indices]
    y = rct_outcomes[sorted_indices]
    treated_size = t.cumsum()
    control_size = (1 - t).cumsum()
    treated_outcomes = (y * t).cumsum()
    control_outcomes = (y * (1 - t)).cumsum()

    with np.errstate(divide="ignore", invalid="ignore"):
        qini_curve = np.where(
            control_size != 0, treated_outcomes - control_outcomes * (treated_size / control_size), 0
        )
    if normalize:
        qini_curve = qini_curve / (np.abs(qini_curve[-1]) + 1e-8)

    qini_score = float(np.sum(qini_curve) / len(qini_curve))
    return qini_curve, qini_score


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Divide each method's score by the best score in the column, as in paper Tables 2 and 7."""
    best = max(scores.values())
    return {method: score / best for method, score in scores.items()}

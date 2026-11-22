"""Import the CausalPFN authors' code from the pinned submodule (third_party/CausalPFN).

We load `benchmarks.hillstrom`, `benchmarks.criteo`, and `benchmarks.baselines.econml` without running
the packages' __init__ files. Those import every benchmark (ACIC, IHDP, RealCause, BART, CATENet, GRF)
and would pull in R, JAX, and other dependencies we don't need for the uplift reproduction.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "CausalPFN"
PINNED_COMMIT = "7da4afa"


def _namespace(name: str, path: Path) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _setup() -> None:
    if not (ROOT / "benchmarks").exists():
        raise SystemExit("third_party/CausalPFN is missing. Run: git submodule update --init --recursive")
    _namespace("benchmarks", ROOT / "benchmarks")
    _namespace("benchmarks.baselines", ROOT / "benchmarks" / "baselines")
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def datasets():
    """Return (HillstromDataset, CriteoDataset) from the authors' code."""
    _setup()
    from benchmarks.criteo import CriteoDataset
    from benchmarks.hillstrom import HillstromDataset

    return HillstromDataset, CriteoDataset


def learners() -> dict:
    """The four comparison methods in paper Tables 2 and 7, as in notebooks/qini.ipynb (hpo=False)."""
    _setup()
    from benchmarks.baselines.econml import (
        DALearnerBaseline,
        SLearnerBaseline,
        TLearnerBaseline,
        XLearnerBaseline,
    )

    return {
        "t_learner": TLearnerBaseline(hpo=False),
        "s_learner": SLearnerBaseline(hpo=False),
        "x_learner": XLearnerBaseline(hpo=False),
        "da_learner": DALearnerBaseline(hpo=False),
    }


def cate_estimator():
    """CausalPFN's CATEEstimator from the pinned source, not the PyPI package."""
    _setup()
    from causalpfn import CATEEstimator

    return CATEEstimator

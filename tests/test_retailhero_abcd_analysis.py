"""Regression checks for reading the archived ABCD notebook summaries."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.retailhero_abcd.analysis import (
    ablation_comparisons,
    compute_stages,
    config_metadata,
    curve_metric_summary,
    load_results,
    registered_comparisons,
    seed_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archived_root(tmp_path):
    """Copy only published text summaries; no training data or model artifacts."""
    names = ["results.json", "compute.json", "decision-table.csv", "comparisons.csv",
             "factorial-effects.csv", "seed-policies.csv"]
    sources = [ROOT / "reports" / "retailhero_abcd" / name for name in names]
    sources.append(ROOT / "experiments" / "retailhero_abcd" / "design.json")
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
    if missing:
        pytest.skip("Local ABCD artifact integration coverage requires: " + ", ".join(missing))
    report = tmp_path / "reports" / "retailhero_abcd"
    report.mkdir(parents=True)
    for name in names:
        shutil.copyfile(ROOT / "reports" / "retailhero_abcd" / name, report / name)
    design = tmp_path / "experiments" / "retailhero_abcd"
    design.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments" / "retailhero_abcd" / "design.json", design / "design.json")
    return tmp_path


def change_csv(root, name, change):
    path = root / "reports" / "retailhero_abcd" / name
    frame = pd.read_csv(path)
    change(frame).to_csv(path, index=False)


def test_archived_scores_and_registered_decisions(archived_root):
    bundle = load_results(archived_root)
    assert bundle.decisions.shape[0] == 2 * 18 * 4
    registered = registered_comparisons(bundle)
    assert len(registered) == 7
    assert not registered.supported_gain.any()
    assert set(registered.reach) == {.3}
    assert bundle.results["selection"] == "A"
    # Exact duplicate secondary/post-hoc rows in the archive are harmless.
    assert not bundle.comparisons.duplicated(["challenger", "reference", "reach", "family"]).any()


def test_configuration_features_and_control_counts(archived_root):
    metadata = config_metadata(load_results(archived_root).design).set_index("config")
    assert metadata.loc["Original", "input_columns"] == 6
    assert metadata.loc["ABCD", "input_columns"] == 28
    assert metadata.loc["ABCD", list("ABCD")].all()
    assert metadata.loc["ABCD", "component_count"] == 4
    for control in ("D_random", "D_shuffled"):
        assert metadata.loc[control, "kind"] == "control"
        assert metadata.loc[control, "input_columns"] == 10
        assert not metadata.loc[control, list("ABCD")].any()


def test_ablation_uses_saved_paired_intervals(archived_root):
    bundle = load_results(archived_root)
    frame = ablation_comparisons(bundle)
    assert len(frame) == 8
    assert set(frame.loc[frame.component == "D", "family"]) == {"registered"}
    assert set(frame.loc[frame.component != "D", "family"]) == {"exploratory_ablation"}
    row = frame[(frame.host == "ptonet") & (frame.component == "D")].iloc[0]
    saved = registered_comparisons(bundle).query("challenger == 'ptonet_ABCD' and reference == 'ptonet_ABC'")
    assert row.low_95 == saved.iloc[0].low_95
    assert row.high_95 == saved.iloc[0].high_95
    assert len(ablation_comparisons(bundle, .5)) == 8
    with pytest.raises(ValueError, match="Unavailable reach"):
        ablation_comparisons(bundle, .4)


def test_seed_metrics_are_not_repeated_per_reach(archived_root):
    bundle = load_results(archived_root)
    metrics = seed_metrics(bundle)
    assert len(metrics) == 74
    assert not metrics.duplicated(["policy", "seed"]).any()
    assert len(metrics[metrics.host == "ptonet"]) == 18 * 3
    assert len(metrics[metrics.host == "causalpfn"]) == 18


def test_curve_summary_averages_training_seeds_without_counting_reaches(archived_root):
    bundle = load_results(archived_root)
    summary = curve_metric_summary(bundle).set_index(["host", "config"])
    seeds = bundle.seeds.query("policy == 'ptonet_ABCD' and reach == .3")
    row = summary.loc[("ptonet", "ABCD")]
    assert row.policies == 3
    assert row.qini_mean == pytest.approx(seeds.qini.mean())
    assert row.qini_seed_sd == pytest.approx(seeds.qini.std(ddof=1))
    assert row.auuc_mean == pytest.approx(seeds.auuc.mean())
    assert row.auuc_seed_sd == pytest.approx(seeds.auuc.std(ddof=1))
    assert summary.loc[("causalpfn", "Original"), "policies"] == 1
    standalone = summary.loc[("standalone", "B")]
    assert standalone.policies == 1
    assert pd.isna(standalone.qini_seed_sd)
    assert len(summary) == 2 * 18 + 2


def test_compute_charges_shared_stages_once_and_keeps_probes(archived_root):
    bundle = load_results(archived_root)
    stages = compute_stages(bundle).set_index("stage").gpu_hours
    assert stages.sum() == pytest.approx(bundle.compute["total_gpu_hours"])
    assert stages["Probes"] > 0
    assert stages["D"] == 0
    assert "PTONet" not in stages.index
    assert stages["C"] == pytest.approx(7.511854081948599)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.iloc[1:], "missing or unexpected keys"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate keys"),
    (lambda d: d.assign(gain=d.gain + 1), "gain must equal"),
    (lambda d: d.assign(input_columns=d.input_columns + 1), "Input column counts"),
    (lambda d: d.drop(columns=["low_95"]), "missing columns"),
    (lambda d: d.assign(low_95=np.inf), "non-finite"),
])
def test_rejects_corrupt_decision_tables(archived_root, mutation, match):
    change_csv(archived_root, "decision-table.csv", mutation)
    with pytest.raises(ValueError, match=match):
        load_results(archived_root)


def test_rejects_conflicting_duplicate_contrasts(archived_root):
    def conflicting(frame):
        row = frame.iloc[[0]].copy()
        row["low_95"] -= 1
        return pd.concat([frame, row])
    change_csv(archived_root, "comparisons.csv", conflicting)
    with pytest.raises(ValueError, match="duplicate contrast/family"):
        load_results(archived_root)


def test_rejects_registered_csv_json_disagreement(archived_root):
    def mismatch(frame):
        frame.loc[frame.family == "registered", "p_holm_7"] = .01
        return frame
    change_csv(archived_root, "comparisons.csv", mismatch)
    with pytest.raises(ValueError, match="disagree with results.json"):
        load_results(archived_root)


def test_rejects_missing_registered_comparison(archived_root):
    def missing(frame):
        registered = frame.index[frame.family == "registered"][0]
        return frame.drop(index=registered)
    change_csv(archived_root, "comparisons.csv", missing)
    with pytest.raises(ValueError, match="registered contrasts has missing"):
        load_results(archived_root)


def test_rejects_seed_score_and_arm_count_mismatches(archived_root):
    def mismatch(frame):
        frame.loc[0, "value"] += 1
        return frame
    change_csv(archived_root, "seed-policies.csv", mismatch)
    with pytest.raises(ValueError, match="Per-seed mean"):
        load_results(archived_root)


@pytest.mark.parametrize("policy", ["causalpfn_Original", "standalone_B"])
def test_rejects_reach_dependent_qini(archived_root, policy):
    def mismatch(frame):
        row = frame.index[frame.policy == policy][0]
        frame.loc[row, "qini"] += 1
        return frame
    change_csv(archived_root, "seed-policies.csv", mismatch)
    with pytest.raises(ValueError, match="Qini/AUUC must be constant"):
        load_results(archived_root)


def test_rejects_inconsistent_gpu_ledger_total(archived_root):
    path = archived_root / "reports" / "retailhero_abcd" / "compute.json"
    data = json.loads(path.read_text())
    data["total_gpu_hours"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU ledger does not sum"):
        load_results(archived_root)


def test_missing_artifact_explains_required_path(archived_root):
    path = archived_root / "reports" / "retailhero_abcd" / "compute.json"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="ABCD analysis requires saved artifact:.*compute.json"):
        load_results(archived_root)

"""Read and validate saved ABCD summaries for CPU-only notebook analysis.

This module never imports the experiment runner or opens customer-level arrays.
Confidence intervals in the decision table describe *paired gains*, not scores.
"""

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ResultBundle:
    design: dict
    results: dict
    compute: dict
    decisions: pd.DataFrame
    comparisons: pd.DataFrame
    effects: pd.DataFrame
    seeds: pd.DataFrame
    project_root: Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _columns(frame, names, source):
    missing = set(names) - set(frame.columns)
    _require(not missing, f"{source} is missing columns: {sorted(missing)}")


def _keys(frame, names, expected, source):
    _require(not frame.duplicated(names).any(), f"{source} contains duplicate keys: {names}")
    actual = set(frame[names].itertuples(index=False, name=None))
    missing, extra = expected - actual, actual - expected
    _require(not missing and not extra, f"{source} has missing or unexpected keys: "
             f"missing={sorted(missing)}, unexpected={sorted(extra)}")


def _close(left, right, message):
    _require(np.allclose(left, right, rtol=1e-8, atol=1e-8, equal_nan=True), message)


def config_metadata(design):
    """Describe the original six fields, optional blocks, and replacement controls.

    Control rows have no active A/B/C/D block: random and shuffled values replace D.
    Their input column count still includes four formula-control columns.
    """
    records = []
    for kind, key in (("core", "core_configurations"), ("control", "formula_controls")):
        for config in design[key]:
            active = {c: kind == "core" and config != "Original" and c in config for c in "ABCD"}
            count = sum(active.values())
            extra = sum(design["component_columns"][c] for c in "ABCD" if active[c])
            if kind == "control":
                extra = design["component_columns"]["D"]
            records.append(dict(config=config, **active, input_columns=design["baseline_columns"] + extra,
                                kind=kind, component_count=count))
    return pd.DataFrame(records)


def load_results(project_root=None):
    """Load a complete published run and fail clearly on inconsistent summaries."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    report = root / "reports" / "retailhero_abcd"

    def read_json(path):
        if not path.is_file():
            raise FileNotFoundError(f"ABCD analysis requires saved artifact: {path}")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def read_csv(name):
        path = report / name
        if not path.is_file():
            raise FileNotFoundError(f"ABCD analysis requires saved artifact: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    bundle = ResultBundle(
        design=read_json(root / "experiments" / "retailhero_abcd" / "design.json"),
        results=read_json(report / "results.json"), compute=read_json(report / "compute.json"),
        decisions=read_csv("decision-table.csv"), comparisons=read_csv("comparisons.csv"),
        effects=read_csv("factorial-effects.csv"), seeds=read_csv("seed-policies.csv"), project_root=root,
    )
    # The publisher emits identical secondary/post-hoc rows through both its
    # registered and full-grid loops. Collapse exact repeats, never conflicting ones.
    bundle.comparisons = bundle.comparisons.drop_duplicates().reset_index(drop=True)
    d, c, e, s = bundle.decisions, bundle.comparisons, bundle.effects, bundle.seeds
    comparison_columns = ["challenger", "reference", "reach", "family", "challenger_value", "reference_value",
                          "gain", "low_95", "high_95", "p_centered", "p_holm_7"]
    _columns(d, [*comparison_columns, "host", "config", "input_columns", "validation_mean", "seed_sd"],
             "decision-table.csv")
    _columns(c, comparison_columns, "comparisons.csv")
    _columns(e, ["host", "component", "reach", "matched_pairs", "gain", "low_95", "high_95"],
             "factorial-effects.csv")
    _columns(s, ["policy", "seed", "reach", "value", "qini", "auuc", "selected", "sms_count",
                 "control_count", "sms_buyers", "control_buyers"], "seed-policies.csv")
    design, result = bundle.design, bundle.results
    configs = design["core_configurations"] + design["formula_controls"]
    _require(len(configs) == len(set(configs)), "Design contains duplicate configurations")
    _keys(d, ["host", "config", "reach"], set(product(design["hosts"], configs, result["reaches"])),
          "decision-table.csv")
    _require(not c.duplicated(["challenger", "reference", "reach", "family"]).any(),
             "comparisons.csv contains duplicate contrast/family keys")
    _keys(e, ["host", "component", "reach"], set(product(design["hosts"], "ABCD", result["reaches"])),
          "factorial-effects.csv")
    _require(not s.duplicated(["policy", "seed", "reach"]).any(), "seed-policies.csv contains duplicate keys")
    metadata = config_metadata(design).set_index("config")
    _close(d["input_columns"], d["config"].map(metadata["input_columns"]),
           "Input column counts disagree with design")
    _require((d["challenger"] == d["host"] + "_" + d["config"]).all(),
             "Decision policy names disagree with host/config")
    _require((d["reference"] == d["host"] + "_Original").all(),
             "Decision reference must be same-host Original")
    for frame, name in ((d, "decision-table.csv"), (c, "comparisons.csv")):
        numeric = frame[["challenger_value", "reference_value", "gain", "low_95", "high_95"]].to_numpy()
        _require(np.isfinite(numeric).all(), f"{name} contains missing or non-finite scores/intervals")
        _close(frame["gain"], frame["challenger_value"] - frame["reference_value"],
               f"{name} gain must equal challenger minus reference")
        _require((frame["low_95"] <= frame["high_95"]).all(), f"{name} contains reversed intervals")
    expected_seed_keys = {
        (f"{host}_{config}", seed, reach)
        for host in design["hosts"] for config in configs for seed in design["host_seeds"][host]
        for reach in result["reaches"]
    }
    grid_seeds = s[s["policy"].isin(d["challenger"])]
    _keys(grid_seeds, ["policy", "seed", "reach"], expected_seed_keys, "seed-policies.csv host grid")
    seed_mean = grid_seeds.groupby(["policy", "reach"])["value"].mean()
    lookup = pd.MultiIndex.from_frame(d[["challenger", "reach"]])
    _close(seed_mean.reindex(lookup), d["challenger_value"],
           "Per-seed mean does not reproduce decision score")
    _require((s["sms_count"] + s["control_count"] == s["selected"]).all(),
             "Per-seed arm counts do not sum to selected customers")
    _require(((s["sms_count"] > 0) & (s["control_count"] > 0)).all(), "Per-seed policy has an empty arm")
    values = 1000 * (s["sms_buyers"] / s["sms_count"] - s["control_buyers"] / s["control_count"])
    _close(values, s["value"], "Per-seed arm purchase rates do not reproduce policy value")
    baseline = d[d["config"] == "Original"].set_index(["host", "reach"])["challenger_value"]
    _close(baseline.reindex(pd.MultiIndex.from_frame(d[["host", "reach"]])), d["reference_value"],
           "Decision reference value disagrees with Original")
    registered_comparisons(bundle)
    seed_metrics(bundle)
    compute_stages(bundle)
    return bundle


def registered_comparisons(bundle):
    """Return the seven registered primary contrasts, preserving their Holm p-values."""
    frame = bundle.comparisons.loc[bundle.comparisons["family"] == "registered"].copy()
    design = bundle.design
    expected = {(f"{host}_{challenger}", f"{host}_{reference}", design["primary_reach"])
                for host in design["hosts"] for challenger, reference in design["primary_contrasts_per_host"]}
    expected.add((f"ptonet_{bundle.results['selection']}", "causalpfn_Original", design["primary_reach"]))
    _require(len(expected) == 7, "Registered family must contain seven planned contrasts")
    _keys(frame, ["challenger", "reference", "reach"], expected, "registered contrasts")
    _require(bundle.results.get("complete_holm_family") is True, "Saved Holm family is incomplete")
    _require(frame["p_holm_7"].between(0, 1).all(),
             "Registered Holm p-values are missing or invalid")
    recorded = pd.DataFrame(bundle.results["registered"])
    _keys(recorded, ["challenger", "reference", "reach"], expected, "results.json registered contrasts")
    keys = ["challenger", "reference", "reach"]
    numeric = ["challenger_value", "reference_value", "gain", "low_95", "high_95", "p_centered", "p_holm_7"]
    _close(frame.set_index(keys)[numeric].sort_index(), recorded.set_index(keys)[numeric].sort_index(),
           "Registered CSV contrasts disagree with results.json")
    frame["supported_gain"] = (frame["low_95"] > 0) & (frame["p_holm_7"] < .05)
    return frame.reset_index(drop=True)


def ablation_comparisons(bundle, reach=.3):
    """Return full ABCD minus each one-block-removed policy, without deriving intervals."""
    _require(reach in bundle.results["reaches"], f"Unavailable reach: {reach}")
    rows = []
    for host in bundle.design["hosts"]:
        for component in "ABCD":
            reduced = "ABCD".replace(component, "")
            matches = bundle.comparisons.loc[
                (bundle.comparisons["challenger"] == f"{host}_ABCD")
                & (bundle.comparisons["reference"] == f"{host}_{reduced}")
                & (bundle.comparisons["reach"] == reach)
            ]
            _require(len(matches) == 1,
                     f"Expected one saved ablation contrast for {host}/{component}/{reach}")
            rows.append(dict(matches.iloc[0], host=host, component=component))
    return pd.DataFrame(rows)


def seed_metrics(bundle):
    """Return Qini and AUUC once per policy/seed, as these metrics span all reaches."""
    grouped = bundle.seeds.groupby(["policy", "seed"], dropna=False)[["qini", "auuc"]].nunique(dropna=False)
    _require((grouped == 1).all().all(), "Qini/AUUC must be constant across reaches for a policy/seed")
    frame = bundle.seeds[["policy", "seed", "qini", "auuc"]].drop_duplicates(["policy", "seed"]).copy()
    frame[["host", "config"]] = frame["policy"].str.split("_", n=1, expand=True)
    return frame.reset_index(drop=True)


def curve_metric_summary(bundle):
    """Average full-curve metrics over training seeds, counting each policy once.

    The count uses policy rows so seedless standalone diagnostics count as one.
    Sample SD is undefined (NaN) when only one policy is available.
    """
    return seed_metrics(bundle).groupby(["host", "config"], as_index=False).agg(
        policies=("policy", "size"),
        qini_mean=("qini", "mean"),
        qini_seed_sd=("qini", "std"),
        auuc_mean=("auuc", "mean"),
        auuc_seed_sd=("auuc", "std"),
    )


def compute_stages(bundle):
    """Group charged GPU ledger time once; retries and probes remain included.

    PTONet was CPU-only. Serving measurements are separate and are not added here.
    """
    hours = dict.fromkeys(["A", "B", "C", "D", "CausalPFN", "Probes", "Other"], 0.)
    for entry in bundle.compute["entries"]:
        label = entry["label"]
        duration = entry["end"] - entry["start"]
        _require(np.isfinite(duration) and duration >= 0, f"Invalid GPU ledger duration: {label}")
        parts = label.split("/")
        if label.startswith("causalpfn_"):
            stage = "CausalPFN"
        elif parts[-1] == "probe":
            stage = "Probes"
        elif len(parts) >= 3 and parts[1] in "ABCD" and len(parts[1]) == 1:
            stage = parts[1]
        else:
            stage = "Other"
        hours[stage] += duration / 3600
    _close(sum(hours.values()), bundle.compute["total_gpu_hours"],
           "GPU ledger does not sum to recorded total")
    return pd.DataFrame([dict(stage=stage, gpu_hours=value) for stage, value in hours.items()])

"""Compare PTONet forecast representations on the X5 validation customers only.

The main score averages three separately ranked seed policies. The ensemble diagnostic
ranks their mean prediction once. Paired intervals condition on the chosen checkpoints;
the same validation outcomes were used for checkpoint selection.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run import DESIGN, ROOT, checked, rank_policy

CONFIGS = ("Original", "C", "C_plus_no_sms")
SEEDS = DESIGN["host_seeds"]["ptonet"]
REACH = DESIGN["primary_reach"]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261123


def load_validation(data_root):
    prepared = data_root / "outputs/retailhero_removal/prepared.npz"
    manifest = json.loads(prepared.with_name("data_manifest.json").read_bytes())
    checked(prepared, manifest["output_sha256"][prepared.name])
    with np.load(prepared, allow_pickle=False) as archive:
        index = archive["validation_index"]
        ids, t, y = (archive[key][index] for key in ("client_id", "t", "y"))
    if (len(np.unique(ids)) != len(ids) or set(np.unique(t)) != {0, 1}
            or set(np.unique(y)) != {0, 1} or len(ids) != manifest["split_counts"]["validation"]):
        raise ValueError("Invalid validation cohort")
    return ids, t, y


def load_predictions(experiment, config, seed, ids):
    if config == "C_plus_no_sms":
        folder = experiment / "forecast_ablation" / f"ptonet_{seed}"
    else:
        folder = experiment / "full/models" / f"ptonet_{seed}_{config}"
    marker = json.loads((folder / "complete.json").read_bytes())
    identity = marker["cohort"], marker["host"], marker["config"], marker["seed"]
    if identity != ("full", "ptonet", config, seed):
        raise ValueError(f"Mismatched fit identity in {folder}")
    path = folder / "predictions.npz"
    checked(path, marker["files"][path.name])
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["id_validation"], ids)
        return archive["validation"], marker


def counts_and_score(t, y, selected):
    sms = t[selected] == 1
    purchases = y[selected]
    sms_count = int(sms.sum())
    control_count = len(selected) - sms_count
    if min(sms_count, control_count) == 0:
        raise ValueError("Targeting rule selected only one treatment arm")
    sms_buyers = int(purchases[sms].sum())
    control_buyers = int(purchases[~sms].sum())
    value = 1000 * (sms_buyers / sms_count - control_buyers / control_count)
    return sms_count, control_count, sms_buyers, control_buyers, value


def paired_differences(orders, t, y, limit, replicates, seed):
    """Resample within each arm and apply every ranking to the same sampled customers."""
    rng = np.random.default_rng(seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    # Rows for each condition are the three seed rankings followed by their ensemble ranking.
    rankings = np.stack([orders["C"], orders["C_plus_no_sms"]])
    scores = np.empty(rankings.shape[:2])
    differences = np.empty((replicates, 2))
    for draw in range(replicates):
        sampled = np.concatenate([arm[rng.integers(len(arm), size=len(arm))] for arm in arms])
        weights = np.bincount(sampled, minlength=len(t))
        for condition, policies in enumerate(rankings):
            for i, order in enumerate(policies):
                selected = np.repeat(order, weights[order])[:limit]
                scores[condition, i] = counts_and_score(t, y, selected)[-1]
        gain = scores[1] - scores[0]
        differences[draw] = gain[:-1].mean(), gain[-1]
    return differences


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(data_root, output_dir):
    ids, t, y = load_validation(data_root)
    limit = round(REACH * len(ids))
    experiment = data_root / "outputs/retailhero_abcd"
    orders, values, details = {}, {}, []
    for config in CONFIGS:
        seeds, predictions, rankings = [], [], []
        for seed in SEEDS:
            score, marker = load_predictions(experiment, config, seed, ids)
            order = rank_policy(score, ids)
            sms, control, sms_buy, control_buy, value = counts_and_score(t, y, order[:limit])
            np.testing.assert_allclose(value, marker["validation_value"], rtol=0, atol=1e-9)
            details.append(dict(config=config, seed=seed, reach=REACH, selected=limit, sms_count=sms,
                                control_count=control, sms_buyers=sms_buy, control_buyers=control_buy,
                                value=value, epoch=marker["epoch"], learning_rate=marker["rate"],
                                run_wall_seconds=marker["runtime_seconds"]))
            predictions.append(score)
            rankings.append(order)
            seeds.append(value)
        ensemble = rank_policy(np.mean(predictions, axis=0, dtype=np.float64), ids)
        orders[config] = np.stack([*rankings, ensemble])
        values[config] = (np.mean(seeds), counts_and_score(t, y, ensemble[:limit])[-1])
    draws = paired_differences(orders, t, y, limit, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    summary = []
    for i, policy in enumerate(("mean_seed_value", "mean_prediction_ensemble")):
        for config in CONFIGS:
            challenger = config == "C_plus_no_sms"
            summary.append(dict(policy=policy, config=config, split="validation", validation_rows=len(ids),
                                reach=REACH, selected=limit, seeds=len(SEEDS), value=values[config][i],
                                gain_vs_C=(values[config][i] - values["C"][i]) if challenger else "",
                                paired_low_95=lower[i] if challenger else "",
                                paired_high_95=upper[i] if challenger else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scores.csv", summary)
    write_csv(output_dir / "per-seed.csv", details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Repository with prepared X5 inputs.")
    parser.add_argument("--output-dir", type=Path, help="Directory for the aggregate validation tables.")
    args = parser.parse_args()
    destination = args.output_dir or args.data_root / "outputs/retailhero_abcd/forecast_ablation/summary"
    compare(args.data_root, destination)

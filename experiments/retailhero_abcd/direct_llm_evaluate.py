"""Evaluate a completed direct-Qwen policy and the archived X5 policies on exactly the same customers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import hypergeom
from sklift.metrics import qini_auc_score, uplift_auc_score

if __package__:
    from .run import DESIGN, ROOT, checked, id_hash, rank_policy, sha
else:
    from run import DESIGN, ROOT, checked, id_hash, rank_policy, sha

DIRECT = "direct_qwen"
RANDOM = "random_targeting"
ARCHIVED = ("standalone_C", "ptonet_Original", "ptonet_ABCD", "causalpfn_Original", "causalpfn_ABCD")
REACHES = (0.1, 0.2, 0.3, 0.5)
REPS = 5000
SEED = 20261123


def load_policies(data_root, predictions):
    """Check the frozen prompt, scored cohort and historical predictions before opening outcomes."""
    cohort = predictions.parent.name
    if cohort not in {"pilot", "full"}:
        raise ValueError("Expected the completed pilot or full prediction archive")
    terminal = json.loads((predictions.parent / "complete.json").read_bytes())
    checked(predictions, terminal["predictions_sha256"])
    registration_path = predictions.parent.parent / "registration.json"
    checked(registration_path, terminal["registration_sha256"])
    registration = json.loads(registration_path.read_bytes())
    if terminal["scope"] != cohort or terminal["method"] != registration["method"]:
        raise ValueError("Prediction lock and prompt registration disagree")
    frozen = json.loads((ROOT / "reports/retailhero_abcd/results.json").read_bytes())
    history = data_root / "outputs/retailhero_abcd/full"
    lock_path = history / "policies_locked.json"
    checked(lock_path, frozen["lock_sha256"])
    lock = json.loads(lock_path.read_bytes())
    archived_path = history / "locked_predictions.npz"
    checked(archived_path, lock["predictions_sha256"])
    with np.load(predictions, allow_pickle=False) as direct:
        ids, score, greedy = (direct[k] for k in ("row_id", "score", "greedy"))
        probabilities = direct["digit_probabilities"]
    ladder = np.asarray(registration["ladder_percentage_points"], dtype=np.float64)
    if probabilities.shape != (len(ids), len(ladder)) or not np.all(
        (probabilities >= 0) & (probabilities <= 1)
    ):
        raise ValueError("Invalid digit probabilities")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(score, probabilities @ ladder, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(greedy, ladder[probabilities.argmax(axis=1)])
    if (
        score.shape != ids.shape
        or not np.isfinite(score).all()
        or ids.ndim != 1
        or len(np.unique(ids)) != len(ids)
        or greedy.shape != ids.shape
        or not np.isfinite(greedy).all()
    ):
        raise ValueError("Invalid direct customer scores")
    if not (
        len(ids) == terminal["rows"] == registration["rows"][cohort]
        and id_hash(ids) == terminal["roster_sha256"] == registration["rosters"][cohort]
    ):
        raise ValueError("Direct predictions differ from the registered cohort")
    policies = {DIRECT: score[None, :]}
    with np.load(archived_path, allow_pickle=False) as archive:
        entire = archive["row_id"]
        if id_hash(entire) != lock["evaluation_ids"]:
            raise ValueError("Historical evaluation roster changed")
        if cohort == "full":
            take = np.arange(len(entire))
        else:
            # Fixed outcome-independent pilot rule. Preserve canonical evaluation order for paired draws.
            def pilot_key(i):
                return hashlib.sha256(b"x5-direct-qwen-20261125|" + entire[i].encode("ascii")).digest()

            take = np.sort(sorted(range(len(entire)), key=pilot_key)[:2048])
        np.testing.assert_array_equal(entire[take], ids)
        for name in ARCHIVED:
            policies[name] = archive[name][:, take]
    seeds = {name: [record["seed"] for record in lock["records"].get(name, [])] for name in ARCHIVED}
    seeds[DIRECT], seeds["standalone_C"] = [None], [None]
    if any(len(seeds[name]) != len(matrix) for name, matrix in policies.items()):
        raise ValueError("Missing or extra seed-level predictions")
    for host in DESIGN["hosts"]:
        if any(seeds[f"{host}_{config}"] != DESIGN["host_seeds"][host] for config in ("Original", "ABCD")):
            raise ValueError(f"Historical {host} seeds differ from the registered comparison")
    ties = dict(
        unique_scores=len(np.unique(score)),
        unique_greedy_values=len(np.unique(greedy)),
        soft_score_percentage_points=dict(
            min=float(score.min()), median=float(np.median(score)), max=float(score.max())
        ),
        greedy_bin_counts=[
            dict(percentage_points=float(x), customers=int((greedy == x).sum())) for x in ladder
        ],
    )
    provenance = dict(
        cohort=cohort,
        customers=len(ids),
        roster_sha256=id_hash(ids),
        predictions_sha256=terminal["predictions_sha256"],
        registration_sha256=terminal["registration_sha256"],
        reference_lock_sha256=frozen["lock_sha256"],
        reference_predictions_sha256=lock["predictions_sha256"],
        labels_sha256=frozen["labels_sha256"],
        **ties,
    )
    return ids, entire, take, policies, seeds, greedy, provenance


def greedy_sensitivity(greedy, t, y, reaches=REACHES):
    """Average the purchase-rate difference over uniform ways of resolving the cutoff tie."""
    rows = []
    treated = t == 1
    for reach in reaches:
        k = round(len(greedy) * reach)
        cutoff = np.partition(greedy, -k)[-k]
        above, tied = greedy > cutoff, greedy == cutoff
        left = k - int(above.sum())
        # The number of treated customers selected from the tied group is hypergeometric.
        m, a = int(tied.sum()), int((tied & treated).sum())
        x = np.arange(max(0, left - (m - a)), min(left, a) + 1)
        sms = int((above & treated).sum()) + x
        both = (sms > 0) & (sms < k)
        if not both.any():
            raise ValueError("Greedy ranking cannot yield a comparison of both treatment arms")
        sms, x = sms[both], x[both]
        weights = hypergeom.pmf(x, m, a, left)
        p1 = y[tied & treated].mean() if a else 0.0
        p0 = y[tied & ~treated].mean() if a < m else 0.0
        sms_buyers = y[above & treated].sum() + x * p1
        control_buyers = y[above & ~treated].sum() + (left - x) * p0
        expected = np.average(sms_buyers / sms - control_buyers / (k - sms), weights=weights)
        rows.append(
            dict(
                reach=reach,
                selected=k,
                cutoff_percentage_points=float(cutoff),
                customers_in_cutoff_bin=m,
                slots_from_cutoff_bin=left,
                value=float(1000 * expected),
            )
        )
    return rows


@njit
def weighted_counts(weights, orders, treatment, outcome, limits):
    """Count selected SMS, control and buyers in a treatment-stratified customer bootstrap draw."""
    counts = np.zeros((len(orders), len(limits), 4), dtype=np.int64)
    for policy, order in enumerate(orders):
        used = remain = pos = customer = 0
        sms = sms_buyers = control_buyers = 0
        for j, limit in enumerate(limits):
            while used < limit:
                if remain == 0:
                    customer = order[pos]
                    remain = weights[customer]
                    pos += 1
                step = min(limit - used, remain)
                used += step
                remain -= step
                if treatment[customer] == 1:
                    sms += step
                    sms_buyers += step * outcome[customer]
                else:
                    control_buyers += step * outcome[customer]
            counts[policy, j] = np.array([sms, limit - sms, sms_buyers, control_buyers])
    return counts


def values(counts):
    sms, control, sms_buyers, control_buyers = counts.transpose(2, 0, 1)
    if np.any(sms == 0) or np.any(control == 0):
        raise ValueError("A selected targeting group lacks one treatment arm")
    return 1000.0 * (sms_buyers / sms - control_buyers / control)


def evaluate(ids, policies, seeds, t, y, cohort, replicates=REPS):
    """Calculate common-draw intervals conditional on the fixed scores and training seeds."""
    t, y = np.asarray(t, dtype=np.int64), np.asarray(y, dtype=np.int64)
    if len(ids) != len(t) or t.shape != y.shape or set(np.unique(t)) != {0, 1} or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid treatment or purchase outcomes")
    names = [*policies, RANDOM]
    limits = np.array([round(len(ids) * q) for q in REACHES])
    orders, curves, slices = [], [], []
    for matrix in policies.values():
        start = len(orders)
        for score in matrix:
            orders.append(rank_policy(score, ids))
            curves.append((qini_auc_score(y, score, t), uplift_auc_score(y, score, t)))
        slices.append(slice(start, len(orders)))
    orders = np.stack(orders)
    arm = [np.flatnonzero(t == i) for i in (0, 1)]

    def scenarios(counts, weights):
        per_seed = values(counts)
        means = [per_seed[s].mean(0) for s in slices]
        # A uniformly targeted customer has the population-average treatment effect at every reach.
        ate = sum(sign * np.dot(weights[a], y[a]) / len(a) for sign, a in zip((-1, 1), arm, strict=True))
        return np.vstack([*means, np.full(len(limits), 1000 * ate)])

    weights = np.ones(len(ids), dtype=np.int64)
    observed = weighted_counts(weights, orders, t, y, limits)
    point = scenarios(observed, weights)
    rng = np.random.default_rng(SEED)
    draws = np.empty((replicates, len(names), len(limits)))
    for r in range(replicates):
        for a in arm:
            weights[a] = np.bincount(rng.integers(len(a), size=len(a)), minlength=len(a))
        draws[r] = scenarios(weighted_counts(weights, orders, t, y, limits), weights)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    differences = np.quantile(draws[:, :1] - draws[:, 1:], [0.025, 0.975], axis=0)
    scores, comparisons, detail = [], [], []
    raw_values = values(observed)
    for i, name in enumerate(names):
        summary = np.mean(curves[slices[i]], axis=0) if name in policies else ("", "")
        for j, reach in enumerate(REACHES):
            scores.append(
                dict(
                    cohort=cohort,
                    customers=len(ids),
                    policy=name,
                    reach=reach,
                    selected=limits[j],
                    seeds=len(policies[name]) if name in policies else 0,
                    value=point[i, j],
                    low_95=intervals[0, i, j],
                    high_95=intervals[1, i, j],
                    qini=summary[0],
                    auuc=summary[1],
                )
            )
            if i:
                comparisons.append(
                    dict(
                        cohort=cohort,
                        reach=reach,
                        challenger=DIRECT,
                        reference=name,
                        gain=point[0, j] - point[i, j],
                        low_95=differences[0, i - 1, j],
                        high_95=differences[1, i - 1, j],
                    )
                )
            if name in policies:
                for k, seed in enumerate(seeds[name], slices[i].start):
                    sms, control, sms_buyers, control_buyers = observed[k, j]
                    detail.append(
                        dict(
                            policy=name,
                            seed=seed,
                            reach=reach,
                            selected=limits[j],
                            sms_count=sms,
                            control_count=control,
                            sms_buyers=sms_buyers,
                            control_buyers=control_buyers,
                            value=raw_values[k, j],
                            qini=curves[k][0],
                            auuc=curves[k][1],
                        )
                    )
    return {"scores.csv": scores, "contrasts.csv": comparisons, "seed-policies.csv": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Terminal {pilot,full}/predictions.npz"
    )
    parser.add_argument("--output-dir", type=Path, help="Destination for aggregate summaries")
    args = parser.parse_args()
    ids, entire, take, policies, seeds, greedy, provenance = load_policies(args.data_root, args.predictions)
    labels = args.data_root / "outputs/retailhero_removal/test_labels.npz"
    checked(labels, provenance["labels_sha256"])
    with np.load(labels, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["client_id"], entire)
        t, y = archive["t"][take], archive["y"][take]
    tables = evaluate(ids, policies, seeds, t, y, provenance["cohort"])
    provenance["greedy_uniform_tie_sensitivity"] = greedy_sensitivity(greedy, t, y)
    output = args.output_dir or args.predictions.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance.update(
        reaches=REACHES,
        primary_reach=0.30,
        bootstrap_seed=SEED,
        bootstrap_replicates=REPS,
        score_unit="incremental buyers per 1,000 SMS",
        classifier_metrics="scikit-uplift normalized",
        comparison="direct Qwen minus reference; exploratory pointwise 95% intervals",
        table_sha256={name: sha(output / name) for name in tables},
    )
    (output / "evaluation.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote matched {provenance['cohort']} summaries for {len(ids):,} customers to {output}", flush=True
    )


if __name__ == "__main__":
    main()

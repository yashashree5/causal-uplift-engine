"""Build component B with nested, treatment-stratified TabPFN nuisance fits."""

import argparse
import gc
import os
import sys
from pathlib import Path
from time import perf_counter

DEPS = Path(__file__).resolve().parents[2] / "outputs/retailhero_abcd/dependencies"
sys.path[:0] = [str(p) for p in (DEPS / "tabpfn", DEPS / "extras", *sorted(DEPS.glob("extra[0-9]*")))]
os.environ["SKB_DATA_DIRECTORY"] = str(DEPS / "skrub_data")
os.environ["MPLCONFIGDIR"] = str(DEPS / "matplotlib_config")
os.environ["HF_HUB_OFFLINE"] = "1"

# Import the numerical stack after installing the pinned search paths and offline settings.
from run import HERE, OUT, PARTS, SEED, checked, cohort, gpu_lane, id_hash, impute, read, save, sha, write  # noqa: E402

# isort: split
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

MODEL = DEPS / "tabpfn-v3.5-20260909.safetensors"
REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
DIGEST = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
CONTEXT = 2048
BATCH = 1024


def context(rows, y, seed):
    from sklearn.model_selection import train_test_split

    if len(rows) <= CONTEXT:
        return rows
    selected, _ = train_test_split(rows, train_size=CONTEXT, random_state=seed, stratify=y[rows])
    return selected


def outcome_fit(clf, train, eligible, test, held_ids, seed, folder):
    import torch

    x, y, ids = (train[key] for key in ("x", "y", "id"))
    arm_fit = context(eligible, y, seed)
    assert not np.intersect1d(ids[arm_fit], held_ids).size
    path, marker = folder / "probabilities.npz", folder / "complete.json"
    identity = dict(fit_ids=id_hash(ids[arm_fit]), allowed_ids=id_hash(ids[eligible]),
                    query_ids=id_hash(held_ids),
                    revision=REVISION, context=CONTEXT, package="10.0.0", code=sha(HERE / "tabular.py"),
                    estimators=1, batch=BATCH)
    if marker.exists():
        record = read(marker)
        assert record["identity"] == identity
        checked(path, record["sha256"])
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["id"], held_ids)
            np.testing.assert_array_equal(z["fit_id"], ids[arm_fit])
            return z["probability"], record
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model = clf(model_path=MODEL, device="cuda", n_estimators=1, fit_mode="fit_with_cache", random_state=seed)
    model.fit(impute(x[arm_fit], x[arm_fit]), y[arm_fit])
    torch.cuda.synchronize()
    fit_seconds = perf_counter()-started
    started = perf_counter()
    blocks = [model.predict_proba(impute(x[arm_fit], test[i:i+BATCH]))[:, 1]
              for i in range(0, len(test), BATCH)]
    prob = np.concatenate(blocks).astype(np.float64)
    torch.cuda.synchronize()
    predict_seconds = perf_counter()-started
    peak = torch.cuda.max_memory_reserved()
    assert np.isfinite(prob).all() and np.all((0 <= prob) & (prob <= 1))
    save(path, id=held_ids, fit_id=ids[arm_fit], probability=prob)
    record = dict(identity=identity, sha256=sha(path), fit_rows=len(arm_fit), query_rows=len(prob),
                  fit_seconds=fit_seconds, predict_seconds=predict_seconds, reserved_bytes=peak)
    write(marker, record)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("B", folder.parent.name, folder.name, len(arm_fit), len(prob),
          round(fit_seconds, 2), round(predict_seconds, 2), flush=True)
    return prob, record


def dr_fit(classifier, train, permitted, query, ids, folder, seed):
    import joblib

    t, y, x = (train[key] for key in ("t", "y", "x"))
    assert not np.intersect1d(train["id"][permitted], ids).size
    nuisances = np.full((len(permitted), 2), np.nan, dtype=np.float64)
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    fit_logs = []
    for fold, (fit, held) in enumerate(inner.split(x[permitted], t[permitted])):
        fit, held_rows = permitted[fit], permitted[held]
        for arm in (0, 1):
            eligible = fit[t[fit] == arm]
            model_folder = folder / f"inner_{fold}" / f"arm_{arm}"
            nuisances[held, arm], record = outcome_fit(classifier, train, eligible, x[held_rows],
                                                      train["id"][held_rows], seed+fold*2+arm, model_folder)
            fit_logs.append(dict(**record, inner_fold=fold, arm=arm))
    assert np.isfinite(nuisances).all()
    propensity = float(t[permitted].mean())
    control, sms = nuisances.T
    obs_y, obs_t = y[permitted], t[permitted]
    target = sms-control + obs_t*(obs_y-sms)/propensity - (1-obs_t)*(obs_y-control)/(1-propensity)
    estimator = HistGradientBoostingRegressor(max_iter=100, max_depth=2, learning_rate=.05,
                                             early_stopping=False, random_state=seed)
    x_fit = x[permitted]
    started = perf_counter()
    estimator.fit(impute(x_fit, x_fit), target)
    training_seconds = perf_counter()-started
    query = impute(x_fit, query)
    started = perf_counter()
    score = estimator.predict(query)
    serving_seconds = perf_counter()-started
    joblib.dump(estimator, folder / "second_stage.joblib")
    digest = sha(folder / "second_stage.joblib")
    save(folder / "doubly_robust.npz", permitted_id=train["id"][permitted], nuisance=nuisances, target=target,
         query_id=ids, score=score, imputer=impute(x_fit, np.full((1, x.shape[1]), np.nan))[0])
    return score, dict(permitted_rows=len(permitted), fit_id_hash=id_hash(train["id"][permitted]),
                       query_rows=len(ids), query_id_hash=id_hash(ids), propensity=propensity,
                       training_seconds=training_seconds, serving_seconds=serving_seconds,
                       model_sha256=digest, inner=fit_logs)


def run(name, probe):
    import torch
    from tabpfn import TabPFNClassifier

    torch.set_num_threads(1)
    checked(MODEL, DIGEST)
    sources = {str(p): sha(p) for p in (HERE / "tabular.py", HERE / "run.py", HERE / "design.json")}
    data = cohort(name)
    train = data["train"]
    folds = train["fold"]
    folder = OUT / name / ("probes/B" if probe else "features/B")
    complete = folder / "complete.json"
    if complete.exists():
        meta = read(complete)
        assert meta["sources"] == sources and meta["checkpoint_sha256"] == DIGEST
        if not probe:
            checked(folder / "values.npz", meta["sha256"])
        return
    settings = dict(sources=sources, checkpoint_sha256=DIGEST, checkpoint_revision=REVISION,
                    checkpoint=MODEL.name, tabpfn_version="10.0.0", context_budget=CONTEXT, estimators=1,
                    batch=BATCH, fit_mode="fit_with_cache",
                    sampling="seeded within each arm, stratified by purchase")
    with gpu_lane(f"{name}/B/{'probe' if probe else 'bank'}"):
        if probe:
            fit = np.flatnonzero((folds != 0) & (train["t"] == 1))
            held = np.flatnonzero(folds == 0)[:1024]
            _, record = outcome_fit(TabPFNClassifier, train, fit, train["x"][held],
                                    train["id"][held], SEED, folder)
            write(complete, dict(**settings, probe=record))
            return
        values = np.full((len(folds), 1), np.nan, dtype=np.float32)
        logs = {}
        for fold in range(3):
            permitted = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            estimate, record = dr_fit(TabPFNClassifier, train, permitted, train["x"][held], train["id"][held],
                                      folder / f"outer_{fold}", SEED+fold*100)
            values[held, 0] = estimate
            logs[f"outer_{fold}"] = record
        query = np.vstack([data[part]["x"] for part in ("validation", "test")])
        query_ids = np.concatenate([data[part]["id"] for part in ("validation", "test")])
        estimate, logs["full"] = dr_fit(TabPFNClassifier, train, np.arange(len(folds)), query, query_ids,
                                       folder / "full", SEED+300)
    n_val = len(data["validation"]["id"])
    assert np.isfinite(values).all() and np.isfinite(estimate).all()
    path = folder / "values.npz"
    save(path, train=values, validation=estimate[:n_val, None].astype(np.float32),
         test=estimate[n_val:, None].astype(np.float32),
         **{"id_"+part: data[part]["id"] for part in PARTS}, fold_train=folds)
    write(complete, dict(**settings, folds=logs, sha256=sha(path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "build"))
    parser.add_argument("--cohort", choices=("development", "full"), default="development")
    args = parser.parse_args()
    run(args.cohort, args.action == "probe")

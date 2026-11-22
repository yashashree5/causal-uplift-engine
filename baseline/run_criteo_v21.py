"""Run the pinned Table 7 methods on the frozen Criteo v2.1 sample."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from baseline import authors, evaluate, run
from baseline.criteo_v21_faiss import install
from baseline.prepare_criteo_v21 import DEFAULT_OUTPUT_DIR, sha256, verify_cohort
from baseline.tasks import SEED

REPO = Path(__file__).resolve().parents[1]
TASK = "criteo_full"
VARIANT = "criteo_v2.1_stratified_2500000"
METHODS = ("causalpfn", "t_learner", "s_learner", "x_learner", "da_learner")
N_ROWS = 2_500_000
N_FOLDS = 5
TOTAL = len(METHODS) * N_FOLDS
DEVICE = "cuda:0"
PINNED_UPSTREAM = "7da4afa35affea0f1fb58893e9f41e87dafdc7eb"
EXPECTED_CHECKPOINT = "4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d"
RNG_META = b"post_fold_random_state"
FINGERPRINT = b"run_fingerprint"


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def progress(output_dir, stage, completed, **detail):
    write_json(
        output_dir / "progress.json",
        {
            "stage": stage,
            "completed_folds": completed,
            "total_folds": TOTAL,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **detail,
        },
    )


def freeze_configuration(output_dir, cohort_manifest, checkpoint):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    actual_upstream = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO / "third_party" / "CausalPFN",
        text=True,
    ).strip()
    if actual_upstream != PINNED_UPSTREAM or sha256(checkpoint) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Unexpected CausalPFN source revision or checkpoint checksum")
    config = {
        "variant": VARIANT,
        "cohort_manifest_sha256": sha256(cohort_manifest),
        "team_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "upstream_commit": actual_upstream,
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "runner_sha256": sha256(Path(__file__)),
        "retrieval_wrapper_sha256": sha256(Path(__file__).with_name("criteo_v21_faiss.py")),
        "methods": list(METHODS),
        "folds": N_FOLDS,
        "root_seed": SEED,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "max_context": 50_000,
        "max_query_batch": 4096,
        "neighbors_per_arm": 1024,
        "worker_limits": {
            name: os.environ.get(name)
            for name in (
                "LOKY_MAX_CPU_COUNT",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "scikit-learn",
                "faiss-cpu",
                "econml",
                "pyarrow",
                "pandas",
            )
        },
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = output_dir / "run_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise RuntimeError("This output directory contains a different benchmark configuration")
    else:
        write_json(manifest, {"fingerprint": fingerprint, "config": config})
    return fingerprint.encode()


def capture_rng():
    name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    return json.dumps(
        {
            "numpy": [name, keys.tolist(), position, has_gaussian, cached_gaussian],
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": torch.cuda.get_rng_state(0).tolist(),
        }
    ).encode()


def restore_rng(raw):
    saved = json.loads(raw)
    name, keys, position, has_gaussian, cached_gaussian = saved["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gaussian, cached_gaussian))
    version, state, gaussian = saved["python"]
    random.setstate((version, tuple(state), gaussian))
    torch.set_rng_state(torch.tensor(saved["torch_cpu"], dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(saved["torch_cuda"], dtype=torch.uint8), 0)


def fold_path(output_dir, method, fold):
    return output_dir / "predictions" / method / TASK / f"seed{SEED}_fold{fold}.parquet"


def save_predictions(path, method, fold, indices, source_ids, treatment, visit, effect, fingerprint):
    data = pa.Table.from_pydict(
        {
            "row_id": indices,
            "source_row_id": source_ids[indices],
            "fold": np.full(len(indices), fold, dtype=np.uint8),
            "seed": np.full(len(indices), SEED, dtype=np.int32),
            "task": [TASK] * len(indices),
            "model": [method] * len(indices),
            "t": treatment[indices].astype(np.int8),
            "y": visit[indices].astype(np.int8),
            "uplift_pred": effect,
        }
    ).replace_schema_metadata({FINGERPRINT: fingerprint, RNG_META: capture_rng()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(data, temporary)
    temporary.replace(path)


def validate_predictions(path, indices, source_ids, treatment, visit, fingerprint):
    file = pq.ParquetFile(path)
    metadata = file.schema_arrow.metadata
    if metadata.get(FINGERPRINT) != fingerprint or RNG_META not in metadata:
        raise RuntimeError(f"Fold artifact belongs to a different benchmark: {path}")
    data = file.read(columns=["row_id", "source_row_id", "t", "y", "uplift_pred"])
    for column, expected in (
        ("row_id", indices),
        ("source_row_id", source_ids[indices]),
        ("t", treatment[indices]),
        ("y", visit[indices]),
    ):
        np.testing.assert_array_equal(data[column].to_numpy(), expected, err_msg=str(path))
    if not np.isfinite(data["uplift_pred"].to_numpy()).all():
        raise ValueError(f"Nonfinite fold predictions in {path}")
    return metadata[RNG_META]


def write_report(output_dir):
    table = evaluate.score_task(output_dir, TASK)
    if set(table["method"]) != set(METHODS) or not table["n_test_rows"].eq(N_ROWS).all():
        raise RuntimeError("Each Table 7 method must have 2,500,000 held-out predictions before scoring")
    table["data_variant"] = VARIANT
    directory = output_dir / "reports"
    directory.mkdir(exist_ok=True)
    table.to_csv(directory / "baseline_criteo_2p5m_v21.csv", index=False)
    header = (
        "# Criteo v2.1, stratified 2.5 million rows\n\n"
        "This benchmark uses a new frozen sample drawn from the verified official Criteo v2.1 archive. "
        "The sample preserves the proportions of the four treatment-by-visit groups. Methods share five "
        "shuffled held-out folds, with each row scored once. Paper scores are reference values for the "
        "authors' cohort. The Table 7 difference measures proximity across cohorts.\n\n"
    )
    report = evaluate.to_markdown("criteo", table).split("## ", 1)[1]
    (directory / "baseline_criteo_2p5m_v21.md").write_text(header + "## " + report, encoding="utf-8")


def log_stratifier_progress(estimator_class):
    module = sys.modules[estimator_class.__module__]
    boosting_model = module.GradientBoostingRegressor

    class LoggedStratifier(boosting_model):
        def fit(self, X, y, sample_weight=None, monitor=None):
            started = time.perf_counter()

            def log_stage(index, estimator, local_variables):
                stage = index + 1
                if stage == 1 or stage % 10 == 0 or stage == estimator.n_estimators:
                    print(
                        f"Stratifier trees {stage}/{estimator.n_estimators} "
                        f"in {time.perf_counter() - started:.1f}s.",
                        flush=True,
                    )
                if monitor is not None:
                    return monitor(index, estimator, local_variables)
                return False

            return super().fit(X, y, sample_weight=sample_weight, monitor=log_stage)

    module.GradientBoostingRegressor = LoggedStratifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Published causalpfn_v0.pt checkpoint")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Prepared benchmark directory"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cohort = output_dir / "cohort"
    fingerprint = freeze_configuration(output_dir, verify_cohort(cohort), checkpoint)
    features = np.load(cohort / "X.npy", mmap_mode="r", allow_pickle=False)
    treatment = np.load(cohort / "t.npy", mmap_mode="r", allow_pickle=False)
    visit = np.load(cohort / "y.npy", mmap_mode="r", allow_pickle=False)
    source_ids = np.load(cohort / "source_row_ids.npy", mmap_mode="r", allow_pickle=False)
    assignment = np.load(cohort / "fold.npy", mmap_mode="r", allow_pickle=False)
    assert features.shape == (N_ROWS, 12)
    assert treatment.shape == visit.shape == source_ids.shape == assignment.shape == (N_ROWS,)
    assert np.all(source_ids[1:] > source_ids[:-1]) and np.unique(assignment).tolist() == list(range(N_FOLDS))
    assert np.bincount(assignment).tolist() == [N_ROWS // N_FOLDS] * N_FOLDS
    assert np.isin(treatment, (0, 1)).all() and np.isin(visit, (0, 1)).all()
    estimator_class = authors.cate_estimator()
    install(estimator_class)
    log_stratifier_progress(estimator_class)
    learners = authors.learners()
    run._set_seeds()

    completed = 0
    print(f"Starting {VARIANT} on {torch.cuda.get_device_name(0)} with {TOTAL} method-folds.", flush=True)
    try:
        for method in METHODS:
            for fold in range(N_FOLDS):
                indices = np.flatnonzero(assignment == fold)
                path = fold_path(output_dir, method, fold)
                if path.is_file():
                    restore_rng(
                        validate_predictions(path, indices, source_ids, treatment, visit, fingerprint)
                    )
                    completed += 1
                    print(f"Verified and resumed {method} fold {fold + 1}/{N_FOLDS}.", flush=True)
                    continue
                train = assignment != fold
                X_train, t_train, y_train = features[train], treatment[train], visit[train]
                X_test = features[indices]
                begin = time.perf_counter()
                progress(output_dir, "fitting", completed, method=method, fold=fold + 1)
                print(
                    f"{method} fold {fold + 1}/{N_FOLDS}: "
                    f"fitting on {len(X_train):,}, scoring {len(indices):,}.",
                    flush=True,
                )
                if method == "causalpfn":
                    estimator = estimator_class(
                        device=DEVICE, model_path=str(checkpoint), max_context_length=50_000
                    )
                    estimator.fit(X=X_train, t=t_train, y=y_train)
                    fit_seconds = time.perf_counter() - begin
                    progress(
                        output_dir,
                        "retrieval_and_gpu_inference",
                        completed,
                        method=method,
                        fold=fold + 1,
                        fit_seconds=fit_seconds,
                    )
                    print(
                        f"{method} fold {fold + 1}: CPU stratifier finished in {fit_seconds:.1f}s. "
                        "Starting retrieval and CUDA scoring.",
                        flush=True,
                    )
                    effect = estimator.estimate_cate(X_test)
                    del estimator
                else:
                    effect = learners[method].estimate_cate(
                        X_train=X_train, t_train=t_train, y_train=y_train, X_test=X_test
                    )
                effect = np.asarray(effect, dtype=np.float64).reshape(-1)
                if len(effect) != len(indices) or not np.isfinite(effect).all():
                    raise ValueError(f"Invalid {method} fold {fold + 1} predictions")
                save_predictions(
                    path, method, fold, indices, source_ids, treatment, visit, effect, fingerprint
                )
                validate_predictions(path, indices, source_ids, treatment, visit, fingerprint)
                completed += 1
                elapsed = time.perf_counter() - begin
                progress(
                    output_dir, "saved_fold", completed, method=method, fold=fold + 1, elapsed_seconds=elapsed
                )
                print(
                    f"Verified {completed}/{TOTAL}: {method} fold {fold + 1} in {elapsed:.1f}s.", flush=True
                )
        progress(output_dir, "scoring", completed)
        write_report(output_dir)
    except Exception as error:
        progress(output_dir, "failed", completed, error=str(error))
        raise
    progress(output_dir, "complete", completed)


if __name__ == "__main__":
    main()

"""Run CausalPFN and the four comparison learners on one dataset, fold by fold.

Usage (on the GPU server):
    python -m baseline.run --dataset hillstrom
    python -m baseline.run --dataset criteo                 # 50k subsample only
    python -m baseline.run --dataset criteo --include-full  # also the full table (Table 7; slow)
    python -m baseline.run --task hill_mens_vs_none --methods causalpfn

Each fold's predictions go to outputs/predictions/<method>/<task>/seed82718_fold<k>.parquet.
Finished folds are skipped, so an interrupted run can be restarted with the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.tasks import DATASET_TASKS, EXPECTED_ROWS, SEED, TASKS, build

METHODS = ["causalpfn", "x_learner", "s_learner", "da_learner", "t_learner"]
MAX_CONTEXT = 50_000  # notebooks/qini.ipynb uses 50,000 for all marketing tasks


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _set_seeds() -> None:
    import random

    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _predict(method: str, fold, device: str, learners: dict) -> np.ndarray:
    if method == "causalpfn":
        from baseline.authors import cate_estimator

        estimator = cate_estimator()(device=device, max_context_length=MAX_CONTEXT)
        estimator.fit(X=fold.X_train, y=fold.y_train, t=fold.t_train)
        return estimator.estimate_cate(X=fold.X_test)
    return learners[method].estimate_cate(
        X_train=fold.X_train, t_train=fold.t_train, y_train=fold.y_train, X_test=fold.X_test
    )


def run_task(task: str, methods: list[str], out_dir: Path, device: str) -> None:
    from baseline.authors import learners as make_learners

    print(f"\n=== {task} ===", flush=True)
    dataset = build(task)
    learners = make_learners()
    n_rows = len(dataset.all_X)
    print(f"rows={n_rows:,} treated={dataset.all_t.mean():.3f} outcome_rate={dataset.all_y.mean():.4f}")
    expected = EXPECTED_ROWS[task]
    if abs(n_rows - expected) / expected > 0.05:
        print(
            f"WARNING: {n_rows:,} rows, expected ~{expected:,}. "
            "Record this in TEA-6; it can explain a gap with the paper.",
            flush=True,
        )

    for method in methods:
        for k in range(len(dataset)):
            path = out_dir / "predictions" / method / task / f"seed{SEED}_fold{k}.parquet"
            if path.exists():
                print(f"skip {method} fold {k} (done)")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fold = dataset[k]
            start = time.time()
            pred = np.asarray(_predict(method, fold, device, learners), dtype=np.float64).reshape(-1)
            seconds = time.time() - start
            pd.DataFrame(
                {
                    "row_id": dataset.all_fold_test_idx[k].astype(np.int64),  # index into the task's arrays
                    "task": task,
                    "fold": np.int32(k),
                    "seed": np.int32(SEED),
                    "model": method,
                    "uplift_pred": pred,
                    "t": fold.t_test.astype(np.int8),
                    "y": fold.y_test.astype(np.float32),
                }
            ).to_parquet(path, index=False)
            print(f"{method:<10} fold {k}  n_test={len(pred):,}  {seconds:,.0f}s", flush=True)

    meta = {
        "task": task,
        "rows": int(n_rows),
        "methods": methods,
        "seed": SEED,
        "max_context_length": MAX_CONTEXT,
        "device": device,
        "git_commit": _git_commit(),
        "causalpfn_source_commit": "7da4afa",
        "python": platform.python_version(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "predictions" / f"{task}_run.json"
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=sorted(DATASET_TASKS))
    group.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--include-full", action="store_true", help="also run criteo_full (Table 7)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./outputs"))
    args = parser.parse_args()

    if args.task:
        tasks = [args.task]
    else:
        tasks = [t for t in DATASET_TASKS[args.dataset] if t != "criteo_full" or args.include_full]

    _set_seeds()
    device = _device(args.device)
    print(f"device={device} tasks={tasks} methods={args.methods}")
    for task in tasks:
        run_task(task, args.methods, Path(args.output_dir), device)


if __name__ == "__main__":
    main()

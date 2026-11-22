"""Freeze a new 2.5M-row Criteo Uplift v2.1 cohort and five held-out folds."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

SOURCE_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
SOURCE_ROWS = 13_979_592
SAMPLE_ROWS = 2_500_000
SAMPLE_SEED = 20_261_120
PAPER_SEED = 82_718
FOLD_SEED = int(np.random.default_rng(PAPER_SEED).integers(0, 2**32 - 1))
FOLD_COUNT = 5
FEATURES = [f"f{index}" for index in range(12)]
DEFAULT_OUTPUT_DIR = Path("outputs/criteo-2p5m-v21")
COHORT_FILES = {"X.npy", "t.npy", "y.npy", "source_row_ids.npy", "fold.npy"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source):
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError(f"Criteo source SHA-256 mismatch: {source_hash}")

    features = np.empty((SOURCE_ROWS, len(FEATURES)), dtype=np.float32)
    treatment = np.empty(SOURCE_ROWS, dtype=np.uint8)
    outcome = np.empty(SOURCE_ROWS, dtype=np.uint8)
    dtypes = {name: np.float32 for name in FEATURES}
    dtypes.update(treatment=np.uint8, visit=np.uint8)
    start = 0
    for chunk in pd.read_csv(source, usecols=list(dtypes), dtype=dtypes, chunksize=500_000):
        end = start + len(chunk)
        if end > SOURCE_ROWS:
            raise ValueError(f"Criteo source exceeds its declared {SOURCE_ROWS:,} rows")
        features[start:end] = chunk[FEATURES].to_numpy(copy=False)
        treatment[start:end] = chunk["treatment"].to_numpy(copy=False)
        outcome[start:end] = chunk["visit"].to_numpy(copy=False)
        start = end
    if start != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS:,} source rows, read {start:,}")
    if np.any(treatment > 1) or np.any(outcome > 1):
        raise ValueError("Treatment and visit must have binary values")
    return features, treatment, outcome


def sample_source_ids(strata):
    source_counts = np.bincount(strata, minlength=4)
    quotas, remainders = np.divmod(source_counts * SAMPLE_ROWS, len(strata))
    unallocated = SAMPLE_ROWS - quotas.sum()
    quotas[np.argsort(-remainders, kind="stable")[:unallocated]] += 1
    generator = np.random.default_rng(SAMPLE_SEED)
    chosen = [
        generator.choice(np.flatnonzero(strata == stratum), size=quota, replace=False)
        for stratum, quota in enumerate(quotas)
    ]
    row_ids = np.sort(np.concatenate(chosen)).astype(np.int64, copy=False)
    if len(row_ids) != SAMPLE_ROWS or np.any(np.diff(row_ids) <= 0):
        raise ValueError("Selected row IDs must contain exactly 2,500,000 unique source rows")
    sample_counts = np.bincount(strata[row_ids], minlength=4)
    if not np.array_equal(sample_counts, quotas) or np.any(sample_counts == 0):
        raise ValueError("The selected rows do not satisfy all four treatment-by-visit quotas")
    return row_ids, source_counts, sample_counts


def assign_folds(features, strata):
    fold_count = np.empty((FOLD_COUNT, 4), dtype=np.int64)
    assignments = np.full(SAMPLE_ROWS, FOLD_COUNT, dtype=np.uint8)
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=FOLD_SEED)
    for fold, (train, test) in enumerate(splitter.split(features, strata)):
        if len(train) != 2_000_000 or len(test) != 500_000 or np.any(assignments[test] != FOLD_COUNT):
            raise ValueError(f"Unexpected train or test membership for fold {fold}")
        assignments[test] = fold
        fold_count[fold] = np.bincount(strata[test], minlength=4)
    if np.any(assignments == FOLD_COUNT) or np.any(np.ptp(fold_count, axis=0) > 1):
        raise ValueError("Held-out folds must partition the cohort and preserve all four strata")
    return assignments, fold_count


def write_cohort(output, arrays, metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cohort-stage-", dir=output.parent) as temp:
        stage = Path(temp) / "cohort"
        stage.mkdir()
        files = {}
        for name, array in arrays.items():
            filename = f"{name}.npy"
            path = stage / filename
            np.save(path, array, allow_pickle=False)
            actual = np.load(path, mmap_mode="r", allow_pickle=False)
            if actual.dtype != array.dtype or not np.array_equal(actual, array):
                raise ValueError(f"Round-trip validation failed for {filename}")
            del actual
            files[filename] = {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": sha256(path)}
        metadata["files"] = files
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)


def verify_cohort(cohort):
    manifest_path = cohort / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["source"]["sha256"] != SOURCE_SHA256
        or manifest["n_rows"] != SAMPLE_ROWS
        or manifest["sampling"]["seed"] != SAMPLE_SEED
        or manifest["folds"]["random_state"] != FOLD_SEED
        or set(manifest["files"]) != COHORT_FILES
    ):
        raise ValueError("Cohort manifest does not match the Criteo v2.1 benchmark configuration")
    for filename, meta in manifest["files"].items():
        if sha256(cohort / filename) != meta["sha256"]:
            raise ValueError(f"Frozen cohort file SHA-256 mismatch: {filename}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Official Criteo Uplift v2.1 CSV gzip")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Benchmark output directory"
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() / "cohort"
    if output.exists():
        manifest_path = verify_cohort(output)
        print(f"Verified existing cohort at {manifest_path}", flush=True)
        return

    print("Verifying and reading Criteo Uplift v2.1", flush=True)
    source_x, source_t, source_y = read_source(source)
    strata = 2 * source_y + source_t
    row_ids, source_counts, sample_counts = sample_source_ids(strata)
    print(f"Selected {row_ids.size:,} source rows. Constructing five held-out folds.", flush=True)
    features = source_x[row_ids]
    if not np.isfinite(features).all():
        raise ValueError("Selected feature values must be finite")
    folds, fold_counts = assign_folds(features, strata[row_ids])
    arrays = {
        "X": features,
        "t": source_t[row_ids].astype(np.float32),
        "y": source_y[row_ids].astype(np.float32),
        "source_row_ids": row_ids,
        "fold": folds,
    }
    manifest = {
        "cohort": "criteo-2p5m-v21",
        "source": {"path": str(source), "sha256": SOURCE_SHA256, "rows": SOURCE_ROWS},
        "n_rows": SAMPLE_ROWS,
        "features": FEATURES,
        "treatment_column": "treatment",
        "outcome_column": "visit",
        "source_row_ids": "zero-based CSV row positions after the header, in ascending order",
        "strata": "2 * visit + treatment, counts in index order 0 through 3",
        "source_stratum_counts": source_counts.tolist(),
        "cohort_stratum_counts": sample_counts.tolist(),
        "sampling": {
            "seed": SAMPLE_SEED,
            "algorithm": (
                "NumPy PCG64 uniform choice without replacement per stratum, "
                "exact quotas by largest remainder"
            ),
        },
        "folds": {
            "algorithm": "shuffled sklearn.model_selection.StratifiedKFold",
            "paper_root_seed": PAPER_SEED,
            "random_state": FOLD_SEED,
            "n_splits": FOLD_COUNT,
            "test_rows_per_fold": 500_000,
            "train_rows_per_fold": 2_000_000,
            "test_stratum_counts": fold_counts.tolist(),
        },
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    write_cohort(output, arrays, manifest)
    print(f"Completed cohort and SHA-256 manifest at {output}", flush=True)


if __name__ == "__main__":
    main()

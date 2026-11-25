"""Recover the published Qwen C forecasts without loading the language model."""

from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import run
else:
    import run


def _adapter_pairs(folder, log, identity, allowed, recipients):
    """Read a single adapter's chunks, aligning their length-sorted rows to the study roster."""
    adapter = run.read(folder / "adapter.json")
    if adapter["identity"] != identity or log["permitted_id_hash"] != run.id_hash(allowed):
        raise ValueError(f"Adapter identity or training roster changed in {folder.name}")
    files = {name.replace("\\", "/"): digest for name, digest in adapter["files"].items()}
    if files["adapter/adapter_model.safetensors"] != log["adapter_sha256"]:
        raise ValueError(f"Adapter weights differ from the published manifest in {folder.name}")
    for name, digest in files.items():
        run.checked(folder / name, digest)
    with np.load(folder / "fit_ids.npz", allow_pickle=False) as saved:
        fit = saved["customer_id"]
    if (run.id_hash(fit) != log["fit_hash"] or len(np.unique(fit)) != log["fit_rows"]
            or not np.isin(fit, allowed).all() or np.intersect1d(fit, recipients).size):
        raise ValueError(f"Invalid adapter fitting customers in {folder.name}")
    if log["score_id_hash"] != run.id_hash(recipients) or log["score_rows"] != len(recipients):
        raise ValueError(f"Scoring roster differs from the published manifest in {folder.name}")

    stamp = dict(adapter=log["adapter_sha256"], **identity)
    index = pd.Index(recipients)
    pairs = np.full((len(index), 2), np.nan, dtype=np.float64)
    for marker in sorted(folder.glob("score_*.json")):
        chunk = run.read(marker)
        if chunk["identity"] != stamp:
            raise ValueError(f"Chunk belongs to a different adapter: {marker}")
        path = marker.with_suffix(".npz")
        run.checked(path, chunk["sha256"])
        with np.load(path, allow_pickle=False) as saved:
            pos = index.get_indexer(saved["customer_id"])
            values = saved["probabilities"]
            if (len(pos) != chunk["rows"] or values.shape != (len(pos), 2)
                    or np.any(pos < 0) or len(np.unique(pos)) != len(pos) or np.isfinite(pairs[pos]).any()):
                raise ValueError(f"Missing, unexpected or repeated customer in {path}")
            if not np.all((values >= 0) & (values <= 1)):
                raise ValueError(f"Invalid scenario probabilities in {path}")
            np.testing.assert_array_equal((values[:, 1] - values[:, 0]).astype(np.float32), saved["score"])
            pairs[pos] = values
    if not np.isfinite(pairs).all():
        raise ValueError(f"Incomplete scenario forecast coverage in {folder.name}")
    return pairs


def load_forecast_pairs(cache_dir, ids_by_part):
    """Return [no-SMS, SMS] columns in the requested training/validation customer order.

    Training forecasts come from the adapter that held each customer out. Validation forecasts
    come from the final training-only adapter. Its output chunks also contain evaluation customer
    IDs and predictions. No evaluation outcomes are used or returned.
    """
    if not ids_by_part or set(ids_by_part) - {"train", "validation"}:
        raise ValueError("Request training or validation customer IDs")
    root = Path(cache_dir)
    components = run.read(run.HERE.parents[1] / "reports/retailhero_abcd/compute.json")["components"]
    published_hash = next(entry["manifest_sha256"] for entry in components if entry["stage"] == "C")
    run.checked(root / "complete.json", published_hash)
    manifest = run.read(root / "complete.json")
    run.checked(root / "values.npz", manifest["sha256"])
    with np.load(root / "values.npz", allow_pickle=False) as bank:
        ids = {part: bank["id_" + part] for part in run.PARTS}
        folds = bank["fold_train"]
        deltas = {part: bank[part][:, 0] for part in ids_by_part}
    logs = {row["split"]: row for row in manifest["timings"]}
    pairs = {}
    if "train" in ids_by_part:
        pairs["train"] = np.full((len(folds), 2), np.nan, dtype=np.float64)
        for fold in np.unique(folds):
            tag = f"outer_{fold}"
            held = folds == fold
            pairs["train"][held] = _adapter_pairs(
                root / tag, logs[tag], manifest["identity"], ids["train"][~held], ids["train"][held]
            )
    if "validation" in ids_by_part:
        recipients = np.concatenate((ids["validation"], ids["test"]))
        full = _adapter_pairs(root / "full", logs["full"], manifest["identity"], ids["train"], recipients)
        pairs["validation"] = full[:len(ids["validation"])]

    result = {}
    for part, requested in ids_by_part.items():
        values = pairs[part]
        np.testing.assert_array_equal((values[:, 1] - values[:, 0]).astype(np.float32), deltas[part])
        pos = pd.Index(ids[part]).get_indexer(requested)
        if np.any(pos < 0) or len(np.unique(pos)) != len(pos):
            raise ValueError(f"Unknown or duplicate {part} customer requested")
        result[part] = values[pos]
    return result

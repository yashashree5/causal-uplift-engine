"""Fit PTONet with the cached Qwen uplift and no-SMS forecasts on X5 validation."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import run

# isort: split
import hosts
import numpy as np
import torch
from forecast_cache import load_forecast_pairs

PARTS = ("train", "validation")
CONFIG = "C_plus_no_sms"


def inputs(data_root):
    prepared = data_root / "outputs/retailhero_removal/prepared.npz"
    manifest = run.read(prepared.with_name("data_manifest.json"))
    prepared_digest = manifest["output_sha256"][prepared.name]
    run.checked(prepared, prepared_digest)
    with np.load(prepared, allow_pickle=False) as archive:
        ids, x, t, y = (archive[k] for k in ("client_id", "x", "t", "y"))
        data = {}
        for part in PARTS:
            indices = archive[part + "_index"]
            data[part] = dict(id=ids[indices], x=x[indices], t=t[indices], y=y[indices])

    cache = data_root / "outputs/retailhero_abcd/full/features/C"
    pairs = load_forecast_pairs(cache, {part: rows["id"] for part, rows in data.items()})
    bank = {}
    for part, pair in pairs.items():
        baseline = run.impute(data["train"]["x"], data[part]["x"])
        gap = (pair[:, 1] - pair[:, 0]).astype(np.float32)
        bank[part] = np.column_stack((baseline, gap, pair[:, 0].astype(np.float32)))
    sources = {name: run.sha(run.HERE / name) for name in (
        "forecast_ablation.py", "forecast_cache.py", "configurations.py", "run.py", "hosts.py", "design.json"
    )}
    sources.update(prepared_npz=prepared_digest, c_complete_json=run.sha(cache / "complete.json"),
                   ptonet_source=hosts.PTONET_SHA256)
    return data, bank, sources


def fit(seed, output, data, bank, sources):
    folder = output / f"ptonet_{seed}"
    if folder.exists():
        record = run.read(folder / "complete.json")
        if record["seed"] != seed or record["sources"] != sources or record["config"] != CONFIG:
            raise ValueError(f"Existing fit has different inputs: {folder}")
        for name, digest in record["files"].items():
            run.checked(folder / name, digest)
        print(f"Reused {folder.name}: {record['validation_value']:.4f}", flush=True)
        return

    print(f"Fitting {folder.name} on CPU", flush=True)
    start = perf_counter()
    scores, measured, state = run.ptonet(hosts, seed, bank, data)
    elapsed = perf_counter() - start
    val = data["validation"]
    value = run.policy_value(scores["validation"], val["id"], val["t"], val["y"])
    with TemporaryDirectory(prefix=f".{folder.name}-", dir=output) as temp_directory:
        temp = Path(temp_directory)
        run.save(temp / "predictions.npz", id_validation=val["id"], validation=scores["validation"])
        run.save(temp / "state.npz", **state)
        files = {name: run.sha(temp / name) for name in ("predictions.npz", "state.npz")}
        run.write(temp / "complete.json", dict(
            config=CONFIG, cohort="full", host="ptonet", seed=seed, validation_value=value,
            runtime_seconds=elapsed, **measured, sources=sources, files=files,
            roster={p: run.id_hash(data[p]["id"]) for p in PARTS}, torch_version=torch.__version__,
        ))
        temp.rename(folder)
    print(f"Completed {folder.name}: validation {value:.4f} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=run.ROOT,
                        help="Repository root containing the prepared X5 cohort and cached C forecasts")
    parser.add_argument("--seed", choices=run.DESIGN["host_seeds"]["ptonet"], type=int,
                        help="Fit one of the three study seeds; by default fit all three")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    hosts.ptonet_file()
    data, bank, sources = inputs(args.data_root)
    output = args.data_root / "outputs/retailhero_abcd/forecast_ablation"
    output.mkdir(parents=True, exist_ok=True)
    for seed in [args.seed] if args.seed is not None else run.DESIGN["host_seeds"]["ptonet"]:
        fit(seed, output, data, bank, sources)


if __name__ == "__main__":
    main()

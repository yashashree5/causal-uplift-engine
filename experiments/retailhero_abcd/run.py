"""Build and score registered X5 policies without opening evaluation labels."""

import argparse
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time

# Set the numerical libraries' thread counts before importing NumPy or scikit-learn.
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import numpy as np
from sklearn.model_selection import StratifiedKFold

if __package__:
    from . import configurations
else:
    import configurations

DESIGN = configurations.DESIGN
PARTS = configurations.PARTS
assemble_configuration = configurations.assemble_configuration
selected_components = configurations.selected_components
# These helpers are also imported by the component and evaluation scripts.
formula_rules = configurations.formula_rules
formulas = configurations.formulas
impute = configurations.impute

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/retailhero_abcd"
REPORT = ROOT / "reports/retailhero_abcd"
PREPARED = ROOT / "outputs/retailhero_removal/prepared.npz"
ROSTER = ROOT / "outputs/retailhero_followup_v2/roster.npz"
CHEAP = ("Original", "D", *DESIGN["formula_controls"])
SEED = 82718


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return path and json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def checked(path, meta):
    path = Path(path)
    if not path.exists() or sha(path) != meta:
        raise ValueError(f"Missing or changed artifact: {path}")


def id_hash(ids):
    return hashlib.sha256("\n".join(ids.tolist()).encode("ascii")).hexdigest()


def cohort(name):
    manifest = read(PREPARED.with_name("data_manifest.json"))
    checked(PREPARED, manifest["output_sha256"]["prepared.npz"])
    archive = read(ROSTER.with_name("prepared.json"))
    checked(ROSTER, next(d for p, d in archive["files"].items() if Path(p).name == ROSTER.name))
    with np.load(PREPARED, allow_pickle=False) as data, np.load(ROSTER, allow_pickle=False) as roster:
        ids, x, t, y = (data[k] for k in ("client_id", "x", "t", "y"))
        indices = {k: data[k + "_index"] for k in PARTS}
        if name == "development":
            indices["train"] = roster["index_train"]
        for part, index in indices.items():
            if part != "train" or name == "development":
                np.testing.assert_array_equal(ids[index], roster["id_" + part])
        assert (y[indices["test"]] == -1).all()
        result = {part: dict(id=ids[ix], x=x[ix], index=ix) for part, ix in indices.items()}
        for part in ("train", "validation"):
            result[part].update(t=t[indices[part]], y=y[indices[part]])
    train = result["train"]
    folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
    train["fold"] = np.full(len(train["id"]), -1, dtype=np.int8)
    for i, (_, holdout) in enumerate(folds.split(train["x"], train["t"])):
        train["fold"][holdout] = i
    assert (train["fold"] >= 0).all()
    return result


def component_bank(name, data, component):
    folder = OUT / name / "features" / component
    metadata = read(folder / "complete.json")
    for source, digest in metadata["sources"].items():
        checked(source, digest)
    path = folder / "values.npz"
    checked(path, metadata["sha256"])
    with np.load(path, allow_pickle=False) as saved:
        for part in PARTS:
            np.testing.assert_array_equal(saved["id_" + part], data[part]["id"])
        np.testing.assert_array_equal(saved["fold_train"], data["train"]["fold"])
        values = {part: saved[part] for part in PARTS}
    assert all(v.shape == (len(data[part]["id"]), DESIGN["component_columns"][component])
               and np.isfinite(v).all() for part, v in values.items())
    return values, sha(folder / "complete.json")


def bank_for(name, data, config):
    components, hashes = {}, {}
    for component in selected_components(config):
        if component != "D":
            components[component], hashes[component] = component_bank(name, data, component)
    baseline = {part: data[part]["x"] for part in PARTS}
    return assemble_configuration(config, baseline, components), hashes


def rank_policy(score, ids):
    if score.shape != ids.shape or not np.isfinite(score).all():
        raise ValueError("Invalid customer predictions")
    return np.lexsort((ids, -score))


def policy_value(score, ids, t, y, reach=.30):
    rows = rank_policy(score, ids)[:round(reach * len(ids))]
    arm = t[rows] == 1
    if arm.all() or not arm.any():
        raise ValueError("Policy selected customers from only one treatment arm")
    return float(1000 * (y[rows][arm].mean() - y[rows][~arm].mean()))


def cell_inputs(host_files, data, feature_hashes):
    model_files = [Path(__file__), HERE / "configurations.py", HERE / "hosts.py", HERE / "PROTOCOL.md",
                   HERE / "design.json", PREPARED, ROSTER, ROOT / DESIGN["formula_source"], *host_files]
    return dict(source={str(p): sha(p) for p in model_files}, features=feature_hashes,
                roster={part: id_hash(data[part]["id"]) for part in PARTS})


def ptonet(source, seed, bank, data):
    import torch

    train, validation, center, spread = source.scale(bank["train"], bank["validation"])
    for block in (train, validation):
        block.clamp_(-5, 5)
    t, y = [torch.as_tensor(data["train"][key], dtype=torch.float32) for key in ("t", "y")]
    val = data["validation"]
    trials = []
    selection = -np.inf
    for rate in (.1, .01):
        torch.manual_seed(seed)
        model = source.make_ptonet(train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        generator = torch.Generator().manual_seed(seed)
        best_value, best_epoch = -np.inf, 0
        trace = []
        for epoch in range(1, 51):
            model.train()
            for batch in torch.randperm(len(train), generator=generator).split(512):
                optimizer.zero_grad(set_to_none=True)
                loss = source.ptonet_loss(model, train[batch], t[batch], y[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError("PTONet nonfinite loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("PTONet nonfinite gradient")
                optimizer.step()
            score = np.diff(source.predict_ptonet(model, validation), axis=1).ravel()
            value = policy_value(score, val["id"], val["t"], val["y"])
            trace.append(value)
            if value > best_value:
                best_value, best_epoch = value, epoch
            if value > selection:
                selection = value
                winner = {k: p.detach().clone() for k, p in model.state_dict().items()}
                chosen = dict(rate=rate, epoch=epoch)
            if epoch - best_epoch >= 10:
                break
        trials.append(dict(rate=rate, best_epoch=best_epoch, best_value=best_value, validation=trace))
    model.load_state_dict(winner)
    scores = {"validation": np.diff(source.predict_ptonet(model, validation), axis=1).ravel()}
    inference_seconds = 0.
    if "test" in bank:
        test = torch.from_numpy(np.ascontiguousarray((bank["test"] - center) / spread, dtype=np.float32))
        test.clamp_(-5, 5)
        started = perf_counter()
        scores["test"] = np.diff(source.predict_ptonet(model, test), axis=1).ravel()
        inference_seconds = perf_counter() - started
    return scores, dict(**chosen, trials=trials, inference_seconds=inference_seconds,
                        gpu_seconds=0., peak_rss_bytes=source.peak_rss_bytes()), {
        "center": center, "spread": spread, **{k: v.numpy() for k, v in winner.items()}}


def causal_pfn(source, seed, bank, data):
    import faiss
    import torch

    faiss.omp_set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    calibration_n = len(data["validation"]["id"])
    # A joint query preserves the native chunk layout across all cells.
    query = np.vstack([bank["validation"], bank["test"]])
    score, measured = source.predict_causalpfn(bank["train"], query, data["train"]["t"], data["train"]["y"])
    return dict(validation=score[:calibration_n], test=score[calibration_n:]), dict(
        **measured, gpu_seconds=measured["train_s"]+measured["infer_s"],
        inference_seconds=measured["infer_s"], peak_rss_bytes=source.peak_rss_bytes()), {}


@contextmanager
def gpu_lane(label):
    import ctypes
    import uuid

    # Windows releases the named mutex automatically if a worker is interrupted.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, "Local\\RetailHeroABCDGpu")
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot create X5 GPU mutex")
    with_gpu = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
    try:
        if not with_gpu:
            raise RuntimeError("The X5 GPU lane is busy")
        ledger = OUT / "gpu_ledger"
        ledger.mkdir(parents=True, exist_ok=True)
        elapsed = sum(read(p)["end"] - read(p)["start"] for p in ledger.glob("*.json") if read(p)["end"])
        for pending in ledger.glob("*.json"):
            item = read(pending)
            if item["end"] is None:
                raise RuntimeError(f"Reconcile interrupted GPU interval before resuming: {pending}")
        if elapsed >= 12 * 3600:
            raise RuntimeError("12 GPU-hour budget reached")
        path = ledger / (uuid.uuid4().hex + ".json")
        start = time()
        write(path, dict(label=label, start=start, end=None))
        try:
            yield
        finally:
            write(path, dict(label=label, start=start, end=time()))
    finally:
        if with_gpu:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def fit(name, host, seed, configs):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if host == "ptonet":
        torch.use_deterministic_algorithms(True)
    if __package__:
        from . import hosts as source
    else:
        import hosts as source
    host_files = source.source_files(host)
    data = cohort(name)
    for config in configs:
        bank, hashes = bank_for(name, data, config)
        folder = OUT / name / "models" / f"{host}_{seed}_{config}"
        inputs = cell_inputs(host_files, data, hashes)
        complete = folder / "complete.json"
        if complete.exists():
            record = read(complete)
            assert record["identity"] == inputs
            for file, digest in record["files"].items():
                checked(folder / file, digest)
            print("Reused", folder.name, record["validation_value"], flush=True)
            continue
        started = perf_counter()
        if host == "causalpfn":
            with gpu_lane(folder.name):
                scores, measured, state = causal_pfn(source, seed, bank, data)
        else:
            scores, measured, state = ptonet(source, seed, bank, data)
        val = data["validation"]
        value = policy_value(scores["validation"], val["id"], val["t"], val["y"])
        files = {"predictions.npz": dict(id_validation=val["id"], id_test=data["test"]["id"], **scores)}
        if state:
            files["state.npz"] = state
        for file, values in files.items():
            save(folder / file, **values)
        measured.setdefault("input_width", bank["train"].shape[1])
        write(complete, dict(identity=inputs, cohort=name, host=host, seed=seed, config=config,
              validation_value=value, runtime_seconds=perf_counter()-started, **measured,
              files={file: sha(folder/file) for file in files}))
        print(f"{folder.name}: validation {value:+.3f}, {perf_counter()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("development", "full"))
    parser.add_argument("host", choices=DESIGN["hosts"])
    parser.add_argument("seed", type=int)
    options = (*DESIGN["core_configurations"], *DESIGN["formula_controls"], "cheap")
    parser.add_argument("config", nargs="+", choices=options)
    args = parser.parse_args()
    if args.seed not in DESIGN["host_seeds"][args.host]:
        parser.error("Unregistered host seed")
    configs = CHEAP if args.config == ["cheap"] else args.config
    fit(args.cohort, args.host, args.seed, configs)


if __name__ == "__main__":
    main()

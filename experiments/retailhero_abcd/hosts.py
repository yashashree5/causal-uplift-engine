"""Adapters for the pinned external host implementations used in the X5 study."""

import ast
import ctypes
import gc
import hashlib
import importlib
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PTONET_SHA256 = "bca648c54010f663d08b2f660d4c6932e8d9a929fd555c29e2519ff160f880e3"
CAUSALPFN_TREE_SHA256 = "560de57bc99055a145c551a10e8b1a136ced3d830e1293299010187baf6e73cc"
CHECKPOINT_SHA256 = "4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d"


def ptonet_file():
    path = Path(os.getenv("X5_PTONET_SOURCE", ROOT / "third_party/PUC/models/PTONet.py")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Set X5_PTONET_SOURCE to the authors' models/PTONet.py. Missing {path}")
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PTONET_SHA256:
        raise ValueError(f"PTONet source differs from the registered implementation: {path}")
    return path


def causalpfn_paths():
    repo = Path(os.getenv("X5_CAUSALPFN_REPO", ROOT / "third_party/CausalPFN")).resolve()
    checkpoint = Path(os.getenv("X5_CAUSALPFN_CHECKPOINT", ROOT / "checkpoints/causalpfn_v0.pt")).resolve()
    return repo / "src", checkpoint


def source_files(host):
    if host == "ptonet":
        return [ptonet_file()]
    if host != "causalpfn":
        raise ValueError(f"Unknown host: {host}")
    src, checkpoint = causalpfn_paths()
    if not (src / "causalpfn/__init__.py").is_file():
        raise FileNotFoundError(f"Initialize the CausalPFN submodule or set X5_CAUSALPFN_REPO. Missing {src}")
    # Hash the Python tree at 7da4afa, independent of local checkout line endings.
    files = sorted(src.rglob("*.py"), key=lambda file: file.relative_to(src).as_posix())
    tree = hashlib.sha256()
    for file in files:
        tree.update(file.relative_to(src).as_posix().encode() + b"\0")
        tree.update(hashlib.sha256(file.read_bytes().replace(b"\r\n", b"\n")).digest())
    if tree.hexdigest() != CAUSALPFN_TREE_SHA256:
        raise ValueError(f"CausalPFN sources differ from the pinned factorial-study snapshot: {src}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Set X5_CAUSALPFN_CHECKPOINT to causalpfn_v0.pt. Missing {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(f"CausalPFN weights differ from the registered checkpoint: {checkpoint}")
    return [checkpoint, *files]


def make_ptonet(input_width):
    path = ptonet_file()
    # The authors' module also imports their synthetic training harness. Load precisely
    # the four audited methods used in the real-data experiment from the verified blob.
    methods = {"__init__", "forward", "predict", "criterion"}
    definition = next(node for node in ast.parse(path.read_bytes()).body
                      if isinstance(node, ast.ClassDef) and node.name == "PTONet")
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    if {node.name for node in definition.body} != methods:
        raise ValueError("Incomplete PTONet definition")
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
    return scope["PTONet"](input_width, 16, 5)


def scale(train, query):
    values = train.astype(np.float64)
    center, spread = values.mean(0), values.std(0) + 1e-6
    return (torch.from_numpy(np.ascontiguousarray((values-center)/spread, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray((query-center)/spread, dtype=np.float32)), center, spread)


def ptonet_loss(model, x, treatment, outcome):
    n = len(x)
    arms = torch.cat((treatment[:, None], torch.zeros(n, 1), torch.ones(n, 1)), dim=0)
    yh, th, eh, eps, logits = model(torch.cat((x, x, x), dim=0), arms)
    uplift = yh[2*n:] - yh[n:2*n]
    return model.criterion(yh[:n], th[:n], eh[:n], eps[:n], logits[:n], uplift, outcome, treatment, .5, 10.)


def predict_ptonet(model, x):
    model.eval()
    with torch.no_grad():
        pairs = [model.predict(chunk) for chunk in x.split(512)]
    result = np.stack([torch.cat(arm).squeeze(-1).numpy() for arm in zip(*pairs)], axis=1)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise FloatingPointError("Invalid predicted probability")
    return result


def predict_causalpfn(train, query, treatment, outcome):
    src, checkpoint = causalpfn_paths()
    sys.path.insert(0, str(src))
    try:
        module = importlib.import_module("causalpfn")
    finally:
        sys.path.remove(str(src))
    if not Path(module.__file__).resolve().is_relative_to(src):
        raise ValueError(f"CausalPFN imported from unexpected source: {module.__file__}")
    if train.shape[1] >= 100 or not (np.isfinite(train).all() and np.isfinite(query).all()):
        raise ValueError("Native CausalPFN expects finite inputs with fewer than 100 columns")
    torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        start = perf_counter()
        model = module.CATEEstimator(device="cuda:0", model_path=str(checkpoint),
                                     max_context_length=50000, max_query_length=1023)
        model.fit(X=train, t=treatment.astype(np.float32), y=outcome.astype(np.float32))
        torch.cuda.synchronize()
        fit_seconds = perf_counter() - start
        start = perf_counter()
        score = np.asarray(model.estimate_cate(X=query), dtype=np.float64)
        torch.cuda.synchronize()
        infer_seconds = perf_counter() - start
        reserved = int(torch.cuda.max_memory_reserved())
        if reserved > 8 * 2**30:
            raise RuntimeError("CausalPFN exceeded the 8 GiB reserved GPU memory budget")
        return score, dict(tune_s=0., train_s=fit_seconds, infer_s=infer_seconds, gpu_reserved_bytes=reserved,
                           gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()), train_rows=len(train),
                           input_width=train.shape[1], max_context_length=50000, max_query_length=1023)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def peak_rss_bytes():
    if os.name == "nt":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
                (key, ctypes.c_size_t) for key in ("peak", "working", "pq", "q", "pn", "n", "page", "ppage")]
        data = MemoryCounters()
        data.cb = ctypes.sizeof(data)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle()), ctypes.byref(data), data.cb):
            raise OSError("GetProcessMemoryInfo failed")
        peak = int(data.peak)
    else:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    if peak > 8 * 2**30:
        raise RuntimeError("Host process exceeded the 8 GiB memory budget")
    return peak

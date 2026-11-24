"""Measure and cache cross-fitted Qwen3 8B SMS-minus-no-SMS scores."""

import argparse
import gc
import os
from pathlib import Path
from time import perf_counter

from run import HERE, OUT, PARTS, ROOT, SEED, checked, cohort, gpu_lane, id_hash, read, save, sha, write
from schedule import LIMIT, gpu_seconds

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import qwen_recipe as recipe

HISTORY = ROOT / "outputs/retailhero_llm/history_inputs.npz"
MODEL = Path(os.environ.get("X5_QWEN_MODEL_DIR", ROOT.parents[1] / "LLM-Models/huggingface" / recipe.CHECKPOINT["directory"]))
CHUNK = 1024


def sources():
    paths = [HERE / "qwen.py", HERE / "run.py", HERE / "design.json", Path(recipe.__file__),
             *sorted(p for p in MODEL.iterdir() if p.is_file() and p.suffix != ".safetensors"), HISTORY]
    return {str(p): sha(p) for p in paths}


def token_path(name, probe):
    return OUT / name / ("probes/C" if probe else "features/C") / "tokens.npz"


def tokenize(name, data, probe, identity):
    path = token_path(name, probe)
    groups = ("train",) if probe else PARTS
    indices = np.concatenate([data[part]["index"] for part in groups])
    if probe:
        indices = indices[:1536]
    ids = np.concatenate([data[part]["id"] for part in groups])[:len(indices)]
    entry = path.with_suffix(".json")
    if entry.exists():
        meta = read(entry)
        assert meta["sources"] == identity and meta["ids"] == id_hash(ids)
        checked(path, meta["sha256"])
        return
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side="left")
    with np.load(HISTORY, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["client_id"][indices], ids)
        s, seq, mask = [z[key][indices] for key in ("summary", "sequence", "sequence_mask")]
        aliases = z["summary_aliases"].tolist()
        assert len(aliases) == s.shape[1]
    arrays = np.full((2*len(indices), recipe.CAP), tokenizer.pad_token_id, dtype=np.uint32)
    lengths = np.zeros(len(arrays), dtype=np.uint16)
    reduced = 0

    for start in range(0, len(indices), CHUNK):
        stop = min(start+CHUNK, len(indices))
        prompts = [recipe.render(tokenizer, s[i], seq[i], mask[i], aliases, arm)
                   for i in range(start, stop) for arm in (0, 1)]
        batch = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        for j, tokens in enumerate(batch):
            row = 2*start+j
            if len(tokens) > recipe.CAP:
                i, arm = divmod(row, 2)
                valid = mask[i].copy()
                for slot in np.flatnonzero(valid):
                    valid[slot] = False
                    tokens = tokenizer.encode(recipe.render(tokenizer, s[i], seq[i], valid, aliases, arm),
                                              add_special_tokens=False)
                    if len(tokens) <= recipe.CAP:
                        break
                reduced += 1
            assert 0 < len(tokens) <= recipe.CAP
            lengths[row] = len(tokens)
            arrays[row, :len(tokens)] = tokens
        if start % (CHUNK*16) == 0:
            print("C tokenized", stop, "of", len(indices), "customers", flush=True)
    save(path, token_id=arrays, length=lengths, customer_id=ids, pad_id=np.array(tokenizer.pad_token_id))
    write(entry, dict(sources=identity, ids=id_hash(ids), sha256=sha(path), rows=len(ids),
                      prompts_with_oldest_baskets_removed=reduced, elapsed_seconds=perf_counter()-started,
                      token_min=int(lengths.min()), token_max=int(lengths.max()), token_mean=float(lengths.mean()),
                      system=recipe.SYSTEM, scenarios=recipe.ARM, question=recipe.QUESTION, batch=recipe.BATCH))


class TokenBank:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.tokens, self.length, self.ids = (z[key] for key in ("token_id", "length", "customer_id"))
            self.pad_id = int(z["pad_id"])

    def __getitem__(self, row):
        return self.tokens[row, :int(self.length[row])]


def train_adapter(model, body, head, tokens, t, y, fit, seed, folder, identity):
    marker = folder / "adapter.json"
    if marker.exists():
        meta = read(marker)
        assert meta["identity"] == identity and meta["fit_id_hash"] == id_hash(tokens.ids[fit])
        for file, digest in meta["files"].items():
            checked(folder / file, digest)
        return PeftModel.from_pretrained(model, folder / "adapter", is_trainable=False), meta
    torch.manual_seed(seed)
    wrapper, targets, params = recipe.adapt(model, body)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
    factual_keys = 2*np.arange(len(t)) + t
    started = perf_counter()
    loss, grad = recipe.epoch(body, head, opt, params, tokens, np.random.default_rng(seed).permutation(fit),
                              factual_keys, y, tokens.pad_id)
    torch.cuda.synchronize()
    seconds = perf_counter()-started
    adapter = folder / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    wrapper.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    meta = dict(identity=identity, seed=seed, targets=targets, parameters=sum(p.numel() for p in params),
                fit_id_hash=id_hash(tokens.ids[fit]), fit_rows=len(fit),
                train_arms=np.bincount(t[fit], minlength=2).tolist(), seconds=seconds,
                loss=loss, max_gradient=grad, **recipe.gpu_guard(),
                files={str(p.relative_to(folder)): sha(p) for p in adapter.iterdir() if p.is_file()})
    save(folder / "fit_ids.npz", customer_id=tokens.ids[fit])
    meta["files"]["fit_ids.npz"] = sha(folder / "fit_ids.npz")
    write(marker, meta)
    return wrapper, meta


def score(body, head, tokens, rows, folder, stamp, reserve):
    # Length-sort before making deterministic, independently resumable chunks.
    max_lengths = np.maximum(tokens.length[2*rows], tokens.length[2*rows+1])
    order = np.argsort(max_lengths, kind="stable")
    out = np.empty(len(rows), dtype=np.float32)
    timings = []
    for start in range(0, len(order), CHUNK):
        pos = order[start:start+CHUNK]
        selection = rows[pos]
        cache = folder / f"score_{start:06d}.npz"
        marker = cache.with_suffix(".json")
        if marker.exists():
            meta = read(marker)
            assert meta["identity"] == stamp
            checked(cache, meta["sha256"])
            with np.load(cache, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["customer_id"], tokens.ids[selection])
                effects = z["score"]
        else:
            if LIMIT - gpu_seconds() <= reserve:
                raise RuntimeError("Remaining GPU budget is reserved for completion of the host grid")
            args = (2*selection[:, None]+np.array([0, 1])).ravel()
            started = perf_counter()
            logits = recipe.infer(body, head, tokens, args, tokens.pad_id).reshape(-1, 2)
            seconds = perf_counter()-started
            probabilities = recipe.probabilities(logits)
            effects = np.diff(probabilities, axis=1).ravel().astype(np.float32)
            save(cache, customer_id=tokens.ids[selection], score=effects, probabilities=probabilities)
            meta = dict(identity=stamp, sha256=sha(cache), seconds=seconds, rows=len(selection), **recipe.gpu_guard())
            write(marker, meta)
        out[pos] = effects
        timings.append(meta["seconds"])
        if start == 0 or start % (CHUNK*8) == 0:
            print("C", folder.name, min(start+CHUNK, len(order)), "/", len(order),
                  "customers scored", flush=True)
    return out, sum(timings)


def run(name, action):
    data = cohort(name)
    probe = action == "probe"
    paths = sources()
    tokenize(name, data, probe, paths)
    if action == "tokenize":
        return
    path = token_path(name, probe)
    folder = path.parent
    token_hash = sha(path)
    identity = dict(tokens=token_hash, model=recipe.CHECKPOINT["sha256"])
    done = folder / "complete.json"
    if done.exists():
        meta = read(done)
        assert meta["identity"] == identity and meta["sources"] == paths
        if not probe:
            checked(folder / "values.npz", meta["sha256"])
        return
    tokens = TokenBank(path)
    n_train, n_val = [len(data[part]["id"]) for part in ("train", "validation")]
    t, y, folds = [data["train"][key] for key in ("t", "y", "fold")]
    if probe:
        work = [("probe", np.arange(512), np.arange(512, 1536))]
    else:
        work = [(f"outer_{fold}", np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(3)]
        work.append(("full", np.arange(n_train), np.arange(n_train, len(tokens.ids))))
    settings = dict(model=recipe.CHECKPOINT, batch=recipe.BATCH, token_limit=recipe.CAP,
                    epochs=1, optimizer="AdamW", lr=1e-4, rank=8, dropout=0, weight_decay=0)
    reserve = 300.
    forecast_path = ROOT / "reports/retailhero_abcd/budget.json"
    if not probe and forecast_path.exists():
        budget = read(forecast_path)
        forecast = budget["chosen"]
        reserve += 1.15 * 8 * forecast["projected_gpu_seconds"]["host_gpu"] / forecast["new_causalpfn_models"]
    output = np.full(len(tokens.ids), np.nan, dtype=np.float32)
    logs = []
    with gpu_lane(f"{name}/C/{action}"):
        free, total = torch.cuda.mem_get_info()
        if free < 11*2**30:
            raise RuntimeError(f"Qwen starts with {free/2**30:.2f} GiB free instead of the archived 11 GiB minimum")
        torch.cuda.set_per_process_memory_fraction(10*2**30/total)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        checked(MODEL / "model.safetensors", recipe.CHECKPOINT["sha256"])
        model, body, tokenizer, head = recipe.load(MODEL)
        torch.cuda.synchronize()
        load_seconds = perf_counter()-started
        assert tokens.pad_id == tokenizer.pad_token_id
        for split, (tag, permitted, query) in enumerate(work):
            if LIMIT - gpu_seconds() <= reserve + 240.:
                raise RuntimeError("Remaining GPU budget cannot start another Qwen adapter split")
            if len(permitted) > 2048:
                fit, _ = train_test_split(permitted, train_size=2048, stratify=t[permitted], random_state=SEED+split)
            else:
                fit = permitted
            assert not np.intersect1d(fit, query).size
            wrapper, record = train_adapter(model, body, head, tokens, t, y, fit, SEED+split, folder/tag, identity)
            stamp = dict(adapter=record["files"]["adapter/adapter_model.safetensors"]
                         if "adapter/adapter_model.safetensors" in record["files"]
                         else sha(folder/tag/"adapter/adapter_model.safetensors"), **identity)
            output[query], score_seconds = score(body, head, tokens, query, folder/tag, stamp, reserve)
            logs.append(dict(split=tag, fit_hash=record["fit_id_hash"], fit_rows=len(fit), score_rows=len(query),
                             permitted_id_hash=id_hash(tokens.ids[permitted]), score_id_hash=id_hash(tokens.ids[query]),
                             fit_seconds=record["seconds"], score_seconds=score_seconds,
                             adapter_sha256=stamp["adapter"], **recipe.gpu_guard()))
            model = wrapper.unload()
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()
        safety = recipe.gpu_guard()
    meta = dict(sources=paths, identity=identity, settings=settings, timings=logs, load_seconds=load_seconds, **safety)
    if not probe:
        assert np.isfinite(output).all()
        save(folder / "values.npz", train=output[:n_train, None],
             validation=output[n_train:n_train+n_val, None], test=output[n_train+n_val:, None],
             **{"id_"+part: data[part]["id"] for part in PARTS}, fold_train=folds)
        meta["sha256"] = sha(folder / "values.npz")
    write(done, meta)
    print("C", action, logs, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tokenize", "probe", "build"))
    parser.add_argument("--cohort", choices=("development", "full"), default="development")
    args = parser.parse_args()
    run(args.cohort, args.action)

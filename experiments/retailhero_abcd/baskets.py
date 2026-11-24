"""Prepare and pretrain the registered purchase-basket encoder."""

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from run import HERE, OUT, PARTS, PREPARED, SEED, checked, cohort, gpu_lane, id_hash, read, save, sha, write

RAW = PREPARED.parent / "data"
STORE = OUT / "baskets.npz"
MANIFEST = OUT / "baskets.json"
WIDTH = 32
CUTOFF = "2019-03-19 00:00:00"


def prepare():
    if MANIFEST.exists():
        checked(STORE, read(MANIFEST)["baskets_sha256"])
        return
    manifest = read(PREPARED.with_name("data_manifest.json"))
    sources = {}
    for name in ("purchases.csv.gz", "clients.csv.gz"):
        path = RAW / name
        digest = manifest["sources"][name]["sha256"]
        checked(path, digest)
        sources[str(path)] = digest
    started = perf_counter()
    clients = pd.Index(pd.read_csv(RAW / "clients.csv.gz", usecols=["client_id"], dtype=str).client_id)
    assert clients.is_unique
    data = np.zeros((len(clients), WIDTH, 4), dtype=np.float32)
    length = np.zeros(len(clients), dtype=np.int8)
    columns = ["client_id", "transaction_id", "transaction_datetime", "purchase_sum",
               "product_quantity", "product_id"]
    dtype = {key: str for key in ("client_id", "transaction_id", "transaction_datetime", "product_id")}
    dtype.update(purchase_sum="float64", product_quantity="float64")
    source_rows = 0

    def consume(frame):
        frame = frame.loc[frame.transaction_datetime < CUTOFF]
        if frame.empty:
            return
        grouped = frame.groupby(["client_id", "transaction_id"], sort=False)
        assert grouped.transaction_datetime.nunique(dropna=False).eq(1).all()
        baskets = grouped[["transaction_datetime", "purchase_sum"]].last()
        baskets["quantity"] = grouped.product_quantity.sum()
        baskets["products"] = grouped.product_id.nunique()
        baskets = baskets.reset_index().sort_values(["client_id", "transaction_datetime", "transaction_id"])
        retained = baskets.groupby("client_id", sort=False).tail(WIDTH).copy()
        retained["timestamp"] = pd.to_datetime(retained.transaction_datetime, format="%Y-%m-%d %H:%M:%S")
        grouped_retained = retained.groupby("client_id", sort=False)
        retained["gap"] = grouped_retained.timestamp.diff().dt.total_seconds().fillna(0) / 86400
        rows = clients.get_indexer(retained.client_id)
        slots = grouped_retained.cumcount().to_numpy()
        distinct = slots == 0
        assert (rows >= 0).all() and not length[rows[distinct]].any()
        np.maximum.at(length, rows, slots + 1)
        data[rows, slots] = retained[["purchase_sum", "quantity", "products", "gap"]].to_numpy()

    carry, last = None, None
    for block in pd.read_csv(RAW / "purchases.csv.gz", usecols=columns, dtype=dtype, chunksize=500000):
        assert block.client_id.is_monotonic_increasing and (last is None or block.client_id.iloc[0] >= last)
        source_rows += len(block)
        last = block.client_id.iloc[-1]
        if carry is not None:
            block = pd.concat([carry, block], ignore_index=True)
        final = block.client_id == last
        carry = block.loc[final].copy()
        consume(block.loc[~final])
        if source_rows % 5000000 == 0:
            print("A prepared", source_rows, "source rows", flush=True)
    consume(carry)
    assert source_rows == manifest["purchases"]["all_purchase_rows"] and np.isfinite(data).all()
    save(STORE, customer_id=clients.to_numpy(dtype="U10"), values=data, length=length)
    write(MANIFEST, dict(sources=sources, baskets_sha256=sha(STORE), cutoff=CUTOFF, source_rows=source_rows,
                         clients=len(clients), histories=int(np.count_nonzero(length)),
                         elapsed_seconds=perf_counter()-started))


def run(name, probe):
    import torch
    from torch import nn

    started = perf_counter()
    data = cohort(name)
    checked(STORE, read(MANIFEST)["baskets_sha256"])
    with np.load(STORE, allow_pickle=False) as stored:
        ids, raw, lengths = (stored[k] for k in ("customer_id", "values", "length"))
    lookup = pd.Index(ids)
    with np.load(PREPARED, allow_pickle=False) as prepared:
        unlabeled = np.flatnonzero(~lookup.isin(prepared["client_id"]))
    split = {part: lookup.get_indexer(data[part]["id"]) for part in PARTS}
    assert all((ix >= 0).all() for ix in split.values())
    fit = split["train"][:2000] if probe else np.r_[split["train"], unlabeled]
    assert not np.intersect1d(fit, np.r_[split["validation"], split["test"]]).size
    mask = np.arange(WIDTH)[None, :] < lengths[:, None]
    bins = np.quantile(raw[fit, :, 0][mask[fit]], np.arange(1, 10) / 10)
    targets = np.searchsorted(bins, raw[:, :, 0], side="right").astype(np.int8)
    values = np.sign(raw) * np.log1p(np.abs(raw))
    valid = values[fit][mask[fit]]
    center = valid.mean(0)
    spread = valid.std(0).clip(1e-6)
    values = (values - center) / spread
    raw = None

    class BasketEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(4, 128)
            self.position = nn.Parameter(torch.zeros(WIDTH, 128))
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.decoder = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
            self.bottleneck = nn.Linear(128, 16)
            self.spend = nn.Linear(16, 10)

        def forward(self, x, valid):
            causal = torch.ones(WIDTH, WIDTH, dtype=torch.bool, device=x.device).triu(1)
            hidden = self.decoder(self.input(x) + self.position, mask=causal, src_key_padding_mask=~valid,
                                  is_causal=True)
            return self.bottleneck(hidden)

    folder = OUT / name / ("probes/A" if probe else "features/A")
    complete = folder / "complete.json"
    sources = {str(p): sha(p) for p in (HERE / "baskets.py", HERE / "run.py", MANIFEST, HERE / "design.json")}
    if complete.exists():
        record = read(complete)
        assert record["sources"] == sources and record["fit_id_hash"] == id_hash(ids[fit])
        if not probe:
            checked(folder / "values.npz", record["sha256"])
            checked(folder / "state.npz", record["state_sha256"])
        return
    rng = np.random.default_rng(SEED)
    train = fit[lengths[fit] >= 2]
    score_rows = {"probe": fit} if probe else split
    with gpu_lane(f"{name}/A/{'probe' if probe else 'bank'}"):
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
        model = BasketEncoder().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        losses = []
        train_started = perf_counter()
        for _ in range(3):
            for batch in np.array_split(rng.permutation(train), max(1, (len(train) + 127) // 128)):
                x = torch.as_tensor(values[batch], device="cuda")
                m = torch.as_tensor(mask[batch], device="cuda")
                y = torch.as_tensor(targets[batch, 1:], device="cuda", dtype=torch.int64)
                z = model.spend(model(x, m)[:, :-1])
                loss = nn.functional.cross_entropy(z[m[:, 1:]], y[m[:, 1:]])
                if not torch.isfinite(loss):
                    raise FloatingPointError("A has nonfinite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        torch.cuda.synchronize()
        train_seconds = perf_counter()-train_started
        outputs = {}
        score_started = perf_counter()
        model.eval()
        model.requires_grad_(False)
        with torch.no_grad():
            for part, indices in score_rows.items():
                output = np.zeros((len(indices), 16), dtype=np.float32)
                nonempty = np.flatnonzero(lengths[indices])
                for positions in np.array_split(nonempty, max(1, (len(nonempty) + 255) // 256)):
                    if not len(positions):
                        continue
                    batch = indices[positions]
                    x = torch.as_tensor(values[batch], device="cuda")
                    m = torch.as_tensor(mask[batch], device="cuda")
                    z = model(x, m)[np.arange(len(batch)), lengths[batch].astype(int)-1]
                    output[positions] = z.cpu().numpy()
                assert np.isfinite(output).all()
                outputs[part] = output
        score_seconds = perf_counter()-score_started
        reserved = torch.cuda.max_memory_reserved()
    meta = dict(sources=sources, train_seconds=train_seconds, score_seconds=score_seconds,
                elapsed_seconds=perf_counter()-started, reserved_bytes=reserved,
                fit_rows=len(fit), optimized_rows=len(train), available_unlabeled=len(unlabeled),
                train_labeled=len(split["train"]), fit_id_hash=id_hash(ids[fit]),
                score_rows=sum(map(len, score_rows.values())), first_loss=losses[0], last_loss=losses[-1],
                bins=bins.tolist(), encoder="4 causal layers, 128 hidden units, 4 heads, 16 outputs",
                normalization="signed log1p, then training-basket mean and standard deviation")
    if not probe:
        save(folder / "values.npz", **outputs, **{"id_"+part: data[part]["id"] for part in PARTS},
             fold_train=data["train"]["fold"])
        save(folder / "state.npz", center=center, spread=spread, bins=bins,
             **{key: p.cpu().numpy() for key, p in model.state_dict().items()})
        meta.update(sha256=sha(folder / "values.npz"), state_sha256=sha(folder / "state.npz"))
    write(complete, meta)
    print("A", "probe" if probe else "bank", len(fit),
          round(train_seconds, 2), round(score_seconds, 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "probe", "build"))
    parser.add_argument("--cohort", choices=("development", "full"), default="development")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run(args.cohort, args.action == "probe")

"""Score X5 customers with a single frozen Qwen3 8B prompt estimating the SMS effect directly."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from filelock import FileLock
from run import checked, id_hash, read, save, sha, write

WEIGHTS = "2d7930212c1a43efade360fa2d66cc0188ebda793b08fb4f8ec1143299147381"
REVISION = "1deaf68f694c40dbce295da300851729d759b21a"
HISTORY = "e124483783f1e2f2dbc595e18f11d0190e54e0ffcecff745c9d63e0c35775c8c"
ROSTER = "7bf7ee6cebdf98f7e1745f4ead0f4079d629e0b9283fe28b30c7509dac0b8ed6"
LADDER = (-20, -10, -5, -2, 0, 2, 5, 10, 20)
PILOT = 2048
SALT = "x5-direct-qwen-20261125|"
CHUNK = 256
BATCH = 16
SYSTEM = (
    "You estimate the causal effect of a promotional SMS for X5 grocery customers. "
    "Given a customer's pre-campaign purchase history, estimate the change in the probability "
    "of making a purchase during the campaign if an SMS is sent instead of no SMS. "
    "Express this difference in percentage points. Choose the nearest option. "
    "Reply with just one digit: 1=-20, 2=-10, 3=-5, 4=-2, 5=0, 6=+2, 7=+5, 8=+10, 9=+20."
)
FIELDS = (
    "first redemption (Unix day; 17974 means none yet), "
    "mean express points spent, total express points spent, "
    "regular points earned last month, spending per day, post-redemption spending per day, transactions, "
    "days since last purchase, purchase span in days, total items, distinct products, distinct stores, "
    "mean transaction spend, SD of transaction spend, age in years, "
    "male flag, female flag, unknown-gender flag"
)
SCHEMA = (
    "redeem_day",
    "express_mean",
    "express_total",
    "points_month",
    "spent_day",
    "spent_after_redeem_day",
    "n_tx",
    "recency",
    "span",
    "qty_total",
    "n_products",
    "n_stores",
    "amount_mean",
    "amount_sd",
    "age",
    "male",
    "female",
    "unknown_gender",
)
QUESTION = "Which digit best represents the SMS-induced change in purchase probability?"
TEMPLATE = (
    "Before March 19, 2019. Customer summaries in order [{fields}]: {summary}\n"
    "Last baskets, oldest first [days before campaign, spend, quantity, distinct products]: "
    "{baskets}\n{question}"
)


def prompt(tokenizer, summary, baskets):
    def fmt(values):
        return ",".join(format(float(v) if v else 0.0, ".6g") if np.isfinite(v) else "NA" for v in values)

    text = TEMPLATE.format(
        fields=FIELDS, summary=fmt(summary), baskets=";".join(fmt(b) for b in baskets), question=QUESTION
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def test_history(root):
    path = root / "outputs/retailhero_llm/history_inputs.npz"
    checked(path, HISTORY)
    with np.load(path, allow_pickle=False) as data:
        indices = data["test_index"]
        ids = data["client_id"][indices]
        summary, seq, masks = (data[k][indices] for k in ("summary", "sequence", "sequence_mask"))
        if tuple(data["summary_aliases"]) != SCHEMA:
            raise ValueError("Unexpected pre-SMS feature order")
    if len(ids) != 40008 or id_hash(ids) != ROSTER or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected the locked X5 evaluation roster")
    return ids, summary, seq, masks


def cohort_order(ids):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256((SALT + ids[i]).encode("ascii")).digest())
    return np.asarray(order)


def register(directory, model_path, ids, order, tokenizer, digit_ids):
    checked(model_path / "model.safetensors", WEIGHTS)
    record = {
        "method": "frozen_qwen3_8b_direct_ordinal_cate",
        "checkpoint_sha256": WEIGHTS,
        "checkpoint_revision": REVISION,
        "history_archive_sha256": HISTORY,
        "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "checkpoint_files": {
            p.name: sha(p) for p in sorted(model_path.iterdir()) if p.suffix in (".json", ".txt", ".jinja")
        },
        "system": SYSTEM,
        "fields": FIELDS,
        "user_template": TEMPLATE,
        "question": QUESTION,
        "ladder_percentage_points": list(LADDER),
        "digit_token_ids": digit_ids,
        "readout": "expected percentage-point score under softmax restricted to digits 1 through 9",
        "greedy": "percentage points for the maximum-logit legal digit",
        "pilot_hash_prefix": SALT,
        "rosters": {"pilot": id_hash(ids[np.sort(order[:PILOT])]), "full": id_hash(ids)},
        "rows": {"pilot": PILOT, "full": len(ids)},
        "batch_size": BATCH,
        "chunk_size": CHUNK,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf8")).hexdigest(),
    }
    path = directory / "registration.json"
    if path.exists():
        if read(path) != record:
            raise ValueError("Frozen inference registration or inputs changed")
    else:
        write(path, record)
    return sha(path)


def evaluate_chunk(model, tokenizer, digit_head, prompts):
    import torch

    sequences = tokenizer(prompts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    lengths = np.fromiter(map(len, sequences), dtype=int)
    if lengths.max() > model.config.max_position_embeddings:
        raise ValueError("Prompt exceeds model context")
    probs = np.empty((len(sequences), len(LADDER)), dtype=np.float64)
    order = np.argsort(lengths, kind="stable")
    with torch.inference_mode():
        for start in range(0, len(order), BATCH):
            ix = order[start : start + BATCH]
            batch = np.full((len(ix), int(lengths[ix].max())), tokenizer.pad_token_id, dtype=np.int64)
            for j, row in enumerate(ix):
                batch[j, : lengths[row]] = sequences[row]
            tokens = torch.as_tensor(batch, device="cuda")
            size = torch.as_tensor(lengths[ix], device="cuda")
            mask = torch.arange(batch.shape[1], device="cuda")[None, :] < size[:, None]
            hidden = model.model(
                input_ids=tokens, attention_mask=mask.long(), use_cache=False
            ).last_hidden_state
            last = hidden[torch.arange(len(ix), device="cuda"), size - 1].float()
            logits = last @ digit_head.T
            probs[ix] = logits.double().softmax(dim=-1).cpu().numpy()
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Invalid constrained digit distribution")
    return probs, lengths


def publish(directory, scope, ids, prob, registration, seconds):
    folder = directory / scope
    ladder = np.array(LADDER, dtype=np.float64)
    score = prob @ ladder
    greedy = ladder[prob.argmax(axis=1)]
    path = folder / "predictions.npz"
    marker = folder / "complete.json"
    if marker.exists():
        existing = read(marker)
        if (
            existing["registration_sha256"] != registration
            or existing["roster_sha256"] != id_hash(ids)
            or existing["rows"] != len(ids)
        ):
            raise ValueError("Forecast publication changed")
        checked(path, existing["predictions_sha256"])
        return
    save(path, row_id=ids, score=score, greedy=greedy, digit_probabilities=prob)
    write(
        marker,
        dict(
            method="frozen_qwen3_8b_direct_ordinal_cate",
            scope=scope,
            rows=len(ids),
            roster_sha256=id_hash(ids),
            predictions_sha256=sha(path),
            registration_sha256=registration,
            unique_greedy_scores=len(np.unique(greedy)),
            unique_scores=len(np.unique(score)),
            forward_and_tokenization_seconds=seconds,
        ),
    )
    print("Locked", scope, len(ids), "predictions.", flush=True)


def run(root, model_path, directory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids, summaries, seq, masks = test_history(root)
    order = cohort_order(ids)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 10)]
    if any(len(tokens) != 1 for tokens in encoded) or len({tuple(t) for t in encoded}) != 9:
        raise ValueError("The nine ordinal options must be distinct single tokens")
    digits = [tokens[0] for tokens in encoded]
    registration = register(directory, model_path, ids, order, tokenizer, digits)
    print("Registered", registration, "for", len(ids), "customers.", flush=True)
    probs = np.empty((len(ids), len(LADDER)), np.float64)
    model, head, elapsed = None, None, 0.0
    for start in range(0, len(order), CHUNK):
        rows = order[start : start + CHUNK]
        path = directory / "chunks" / f"part_{start:06d}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            record = read(marker)
            if record["registration_sha256"] != registration:
                raise ValueError("Cannot mix different model or prompt registrations")
            checked(path, record["sha256"])
            with np.load(path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["row_id"], ids[rows])
                probs[rows] = stored["probabilities"]
        else:
            if model is None:
                torch.set_num_threads(2)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": 0},
                    dtype=torch.bfloat16,
                    local_files_only=True,
                    attn_implementation="sdpa",
                ).eval()
                model.requires_grad_(False)
                head = model.lm_head.weight[digits].detach().float()
            begun = perf_counter()
            prompts = [prompt(tokenizer, summaries[i], seq[i][masks[i]]) for i in rows]
            values, lengths = evaluate_chunk(model, tokenizer, head, prompts)
            seconds = perf_counter() - begun
            save(path, row_id=ids[rows], probabilities=values)
            probs[rows] = values
            record = dict(
                registration_sha256=registration,
                sha256=sha(path),
                rows=len(rows),
                seconds=seconds,
                min_tokens=int(lengths.min()),
                max_tokens=int(lengths.max()),
            )
            write(marker, record)
            print("Scored", start + len(rows), "/", len(ids), "in", round(seconds, 1), "chunk s.", flush=True)
        elapsed += record["seconds"]
        if start + len(rows) == PILOT:
            chosen = np.sort(order[:PILOT])
            publish(directory, "pilot", ids[chosen], probs[chosen], registration, elapsed)
    publish(directory, "full", ids, probs, registration, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.data_root / "outputs/retailhero_direct_llm"
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "inference.lock", timeout=0):
        run(args.data_root, args.model_dir, output)

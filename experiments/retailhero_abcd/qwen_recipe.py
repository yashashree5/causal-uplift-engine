"""The Qwen3 8B prompt, last-four-layer LoRA, and binary purchase readout for C."""

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = dict(
    directory="qwen3-8b-bnb4-1deaf68",
    revision="1deaf68f694c40dbce295da300851729d759b21a",
    sha256="2d7930212c1a43efade360fa2d66cc0188ebda793b08fb4f8ec1143299147381",
    layers=36,
    width=4096,
)
BATCH = 8
CAP = 512
ARM = ("The customer receives no promotional SMS.", "The customer receives a promotional SMS.")
SYSTEM = "Assess a retail purchase using only the supplied history. Answer Yes or No."
QUESTION = (
    "Under this randomly assigned campaign scenario, will the customer make a purchase after the campaign?"
)


def render(tokenizer, summary, sequence, mask, aliases, arm):
    def number(value):
        return format(float(value) if value else 0., ".6g") if np.isfinite(value) else "NA"

    summary_text = ",".join(map(number, summary))
    purchases = ";".join(",".join(map(number, row)) for row in sequence[mask])
    content = (f"Before SMS cutoff 2019-03-19.\nSummary[{','.join(aliases)}]={summary_text}"
               f"\nLast purchases [days_before_sms,amount,quantity,products]={purchases}"
               f"\n{ARM[arm]}\n{QUESTION}")
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=content)]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load(path):
    torch.set_num_threads(4)
    torch.manual_seed(820368)
    torch.cuda.manual_seed_all(820368)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    body = model.model
    assert len(body.layers) == CHECKPOINT["layers"] and body.config.hidden_size == CHECKPOINT["width"]
    answer_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    assert all(len(ids) == 1 for ids in answer_ids)
    head = model.lm_head.weight[[ids[0] for ids in answer_ids]].detach().float()
    return model, body, tokenizer, head


def adapt(model, body):
    paths = []
    for i in range(len(body.layers) - 4, len(body.layers)):
        layer = body.layers[i]
        for section, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                               ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                module = getattr(getattr(layer, section), name)
                assert hasattr(module, "in_features") and hasattr(module, "out_features")
                paths.append(f"model.layers.{i}.{section}.{name}")
    config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.,
                        target_modules=paths, bias="none", revision=CHECKPOINT["revision"])
    wrapper = get_peft_model(model, config)
    params = [p for p in wrapper.parameters() if p.requires_grad]
    assert len(paths) == 28 and params and all(torch.isfinite(p).all() for p in params)
    return wrapper, paths, params


def gpu_guard():
    reserved = torch.cuda.max_memory_reserved()
    if reserved > 10 * 2**30:
        raise RuntimeError(f"Torch reserved {reserved} bytes, above 10 GiB")
    return dict(gpu_reserved_bytes=reserved, gpu_allocated_bytes=torch.cuda.max_memory_allocated())


def probabilities(logits):
    return 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -70, 70)))


def batch_logits(body, head, sequences, pad_id):
    lengths = torch.tensor(list(map(len, sequences)), dtype=torch.long, device="cuda")
    width = max(map(len, sequences))
    ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    for i, tokens in enumerate(sequences):
        ids[i, :len(tokens)] = tokens
    ids = torch.as_tensor(ids, device="cuda")
    mask = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
    hidden = body(input_ids=ids, attention_mask=mask.long(), use_cache=False).last_hidden_state
    last = hidden[torch.arange(len(ids), device="cuda"), lengths-1]
    logits = last.float() @ head.T
    cap = getattr(body.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits[:, 1] - logits[:, 0]


def infer(body, head, tokens, rows, pad_id):
    order = sorted(range(len(rows)), key=lambda i: len(tokens[rows[i]]))
    result = np.empty(len(rows), dtype=np.float32)
    body.eval()
    with torch.no_grad():
        for start in range(0, len(order), BATCH):
            chunk = order[start:start+BATCH]
            margin = batch_logits(body, head, [tokens[rows[i]] for i in chunk], pad_id)
            result[chunk] = margin.cpu().numpy()
            gpu_guard()
    assert np.isfinite(result).all()
    return result


def epoch(body, head, optimizer, params, tokens, rows, factual_keys, y, pad_id):
    sum_loss, max_grad = 0., 0.
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start+BATCH]
        optimizer.zero_grad(set_to_none=True)
        logits = batch_logits(body, head, [tokens[i] for i in factual_keys[batch]], pad_id)
        loss = binary_cross_entropy_with_logits(
            logits, torch.as_tensor(y[batch], device="cuda", dtype=torch.float32)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, .5, error_if_nonfinite=True).item()
        optimizer.step()
        sum_loss += loss.item() * len(batch)
        max_grad = max(max_grad, grad)
        gpu_guard()
    return sum_loss / len(rows), max_grad

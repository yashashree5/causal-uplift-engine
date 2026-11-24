"""Assemble the X5 feature configurations for CausalPFN and PTONet."""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
PARTS = ("train", "validation", "test")
CONTROL_SEED = 91427


def selected_components(config):
    if config in DESIGN["formula_controls"]:
        return "D"
    if config not in DESIGN["core_configurations"]:
        raise ValueError(f"Unknown X5 configuration: {config!r}")
    return "" if config == "Original" else config


def impute(fit, query):
    """Replace nonfinite entries with the training mean, or zero for an empty column."""
    fit = fit.astype(np.float64)
    finite = np.isfinite(fit)
    count = finite.sum(0)
    mean = np.divide(np.where(finite, fit, 0).sum(0), count, out=np.zeros(fit.shape[1]), where=count > 0)
    return np.where(np.isfinite(query), query, mean).astype(np.float32)


def formula_rules():
    """Load the four archived rules and reproduce their seeded random control."""
    path = ROOT / DESIGN["formula_source"]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DESIGN["formula_sha256"]:
        raise ValueError(f"Archived X5 formula file has changed: {path}")
    archived = [tuple(rule) for rule in json.loads(payload)["programs"]]
    rng = np.random.default_rng(CONTROL_SEED)
    controls = []
    for op, a, b in archived:
        interchangeable = op in ("add", "mul")
        options = [(i, j) for i in range(4) for j in range(4) if i != j
                   and (not interchangeable or i < j) and (i, j) != (a, b)]
        i, j = options[int(rng.integers(len(options)))]
        controls.append((op, i, j))
    return archived, controls


def formulas(x, rules):
    columns = x.astype(np.float64).T
    operations = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": lambda a, b: a / (1 + np.abs(b))}
    return np.column_stack([operations[op](columns[a], columns[b]) for op, a, b in rules]).astype(np.float32)


def rounded_operands(x):
    """Reproduce the six-significant-digit numeric inputs of the original Qwen prompt."""
    operands = x[:, :4]
    numbers = (float(format(float(v) if v else 0., ".6g")) if np.isfinite(v) else np.nan
               for v in operands.flat)
    return np.fromiter(numbers, dtype=np.float64, count=operands.size).reshape(operands.shape)


def formula_block(baseline, config):
    rounded = {part: rounded_operands(x) for part, x in baseline.items()}
    fixed, random = formula_rules()
    rules = random if config == "D_random" else fixed
    result = {}
    for index, part in enumerate(PARTS):
        block = formulas(impute(rounded["train"], rounded[part]), rules)
        if config == "D_shuffled":
            rng = np.random.default_rng(np.random.SeedSequence([CONTROL_SEED, index]))
            block = block[rng.permutation(len(block))]
        result[part] = block
    return result


def assemble_configuration(config, baseline, components=None):
    """Build train, validation and test matrices in the experiment's column order.

    baseline maps each partition to its six raw numeric columns, in the order
    listed in programs.json's first six fields. components maps A, B or C to
    partition arrays aligned to the same customer order, with 16, 1 or 1
    columns respectively. Training B and C scores should be out of fold.
    D and its controls are calculated directly from the baseline.
    """
    selected = selected_components(config)
    x = {part: np.asarray(baseline[part]) for part in PARTS}
    for part, matrix in x.items():
        if matrix.ndim != 2 or matrix.shape[1] != DESIGN["baseline_columns"]:
            raise ValueError(f"{part}: expected six baseline columns, got {matrix.shape}")
    supplied = {} if components is None else components
    blocks = {part: [impute(x["train"], x[part])] for part in PARTS}
    for component in selected:
        if component == "D":
            values = formula_block(x, config)
        elif component in supplied:
            values = supplied[component]
        else:
            raise ValueError(f"Missing component {component} for configuration {config}")
        width = DESIGN["component_columns"][component]
        for part in PARTS:
            value = np.asarray(values[part])
            if value.shape != (len(x[part]), width) or not np.isfinite(value).all():
                raise ValueError(
                    f"{part}: {component} must have {len(x[part])} rows and {width} finite columns"
                )
            blocks[part].append(value)
    return {part: np.column_stack(parts) for part, parts in blocks.items()}


if __name__ == "__main__":
    for name in DESIGN["core_configurations"] + DESIGN["formula_controls"]:
        width = DESIGN["baseline_columns"] + sum(
            DESIGN["component_columns"][c] for c in selected_components(name)
        )
        print(f"{name:12s} {width:2d} columns")

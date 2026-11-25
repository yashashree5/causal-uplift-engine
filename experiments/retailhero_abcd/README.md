# X5 configurations for PTONet and CausalPFN

The [X5 results and review guide](../../reports/retailhero_abcd/README.md) includes both models' scores, paired uncertainty intervals and charts at 10%, 20%, 30% and 50% SMS reach.

Use [configurations.py](configurations.py) to assemble the input matrix for any of the 16 configurations or either formula control. This part requires Python and NumPy. Run these commands from the repository root to list the configurations and verify the implementation on synthetic inputs.

```console
python experiments/retailhero_abcd/configurations.py
python experiments/retailhero_abcd/verify_plan.py
```

The Python API takes the six baseline columns for each split and any selected A, B and C feature arrays. Rows must be in the same customer order. It computes D locally from the fixed [Qwen formulas](programs.json). The example uses synthetic arrays to show the interface.

```python
import numpy as np

from experiments.retailhero_abcd.configurations import assemble_configuration

sizes = {"train": 12, "validation": 4, "test": 6}
baseline = {part: np.zeros((n, 6)) for part, n in sizes.items()}
history = {part: np.zeros((n, 16)) for part, n in sizes.items()}
matrices = assemble_configuration("AD", baseline, {"A": history})
assert matrices["test"].shape == (6, 26)
```

## Component builders

| Source | Purpose |
|---|---|
| [baskets.py](baskets.py) | Build A from pre-SMS purchases. |
| [tabular.py](tabular.py) | Build B using TabPFN-3.5 and a doubly robust effect regressor. |
| [qwen.py](qwen.py) and [qwen_recipe.py](qwen_recipe.py) | Build C using Qwen3 8B, with per-customer SMS and no-SMS prompts. |
| [configurations.py](configurations.py) | Build D and its controls, and assemble the selected features for either host. |
| [run.py](run.py) and [hosts.py](hosts.py) | Fit a given host, seed and configuration on the chosen cohort. |
| [schedule.py](schedule.py) | Generate selected feature banks and run their configurations on both hosts. |

## Running with the prepared experiment inputs

The model-fitting path uses the existing X5 study inputs and a Windows CUDA machine. Customer data and downloaded model weights are provisioned separately. The following paths are relative to the repository root.

| Input | Expected location |
|---|---|
| Six-field cohort, treatment, masked outcomes, split indices and integrity manifest | `outputs/retailhero_removal/prepared.npz` and adjacent `data_manifest.json` |
| Customer IDs for the development and evaluation partitions | `outputs/retailhero_followup_v2/roster.npz` and adjacent `prepared.json` |
| Raw basket data for A | `outputs/retailhero_removal/data/clients.csv.gz` and `purchases.csv.gz` in the same directory |
| Eighteen summaries, eight recent transactions, masks and field names for C | `outputs/retailhero_llm/history_inputs.npz` |
| TabPFN-3.5 classifier for B | `outputs/retailhero_abcd/dependencies/tabpfn-v3.5-20260909.safetensors` |
| Offline Qwen3 8B NF4 model and tokenizer | Set `X5_QWEN_MODEL_DIR` to the local checkpoint directory. |
| Native CausalPFN sources and weights | Initialize the repository's CausalPFN submodule or set `X5_CAUSALPFN_REPO`. Set `X5_CAUSALPFN_CHECKPOINT` to the study weights. |
| Author implementation of PTONet | Set `X5_PTONET_SOURCE` to a local copy of [the authors' pinned PTONet.py](https://github.com/euzmin/PUC/blob/6c9f60839fc8707a8d5646de8944f148bcf46c65/models/PTONet.py). |

The host environment needs NumPy, pandas, scikit-learn, PyTorch, FAISS, psutil and the selected host's dependencies. The feature environment also needs TabPFN 10.0.0 for B and Transformers, PEFT, bitsandbytes and Accelerate for Qwen3 8B. Set `X5_FEATURE_PYTHON` to its Python executable if the environments are separate. With the prepared inputs available, the following commands fit one configuration or schedule both hosts.

```console
python experiments/retailhero_abcd/run.py development ptonet 20261123 D
python experiments/retailhero_abcd/schedule.py full --components A B C D
```

The scheduler writes local feature banks, model files, predictions and timings under `outputs/retailhero_abcd/`. See the [experiment protocol](PROTOCOL.md) for the fold rules, feature dimensions and distinction between C and D.

## Reproduce the PTONet forecast follow-up

This validation-only experiment compares the archived PTONet Original and C fits with a new fit that adds Qwen's no-SMS purchase forecast to C. The new model has eight input columns. The three CPU fits reuse the original out-of-fold Qwen scores for training and the training-only adapter for validation. [Follow-up results and frozen-pilot context](../../reports/retailhero_abcd/FORECAST.md)

Use the host Python environment and the pinned `X5_PTONET_SOURCE` described above. Set `--data-root` to the repository containing the prepared X5 cohort, the original C cache (including its chunk and adapter files), and the saved Original and C PTONet fits under `outputs/retailhero_abcd/full/models/`. If these files are in the current repository, use the default paths below.

```console
python experiments/retailhero_abcd/forecast_ablation.py
python experiments/retailhero_abcd/forecast_comparison.py
```

The first command fits three PTONet seeds and saves their validation predictions under `outputs/retailhero_abcd/forecast_ablation/`. If all three completed fits are already available, run the second command to regenerate the per-seed scores and paired comparison in the `summary/` directory.

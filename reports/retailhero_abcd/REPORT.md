# X5 RetailHero factorial evaluation

Both hosts used 140,027 training customers, 20,004 validation customers, and 40,008 evaluation customers. The reused evaluation cohort is exploratory. The primary value is incremental buyers per 1,000 SMS at 30% reach. The prespecified secondary reaches are 10% and 20%. The 50% reach was requested after review of the original results and is exploratory.

None of the seven registered comparisons supported a gain at 30% reach.

The run used **8.991 GPU-hours of the 12-hour budget**, including probes and retries. The measured forecast with headroom was 9.74 GPU-hours. Validation selected **PTONet A**. Evaluation outcomes were opened after prediction and selection hashes were locked.

## Registered decisions

| Challenger | Reference | Challenger value | Reference value | Gain | Paired 95% interval | Holm p | Decision |
|---|---|---:|---:|---:|---|---:|---|
| causalpfn_ABCD | causalpfn_Original | 51.02 | 50.61 | +0.41 | [-19.08, +18.07] | 1.0000 | Estimated gain of 0.41, with no adjusted statistical support. |
| causalpfn_D | causalpfn_Original | 56.39 | 50.61 | +5.79 | [-10.89, +21.05] | 1.0000 | Estimated gain of 5.79, with no adjusted statistical support. |
| causalpfn_ABCD | causalpfn_ABC | 51.02 | 56.17 | -5.15 | [-21.57, +11.01] | 1.0000 | Estimated decline of 5.15, with no adjusted statistical support. |
| ptonet_ABCD | ptonet_Original | 55.16 | 46.96 | +8.19 | [-2.06, +18.62] | 0.8594 | Estimated gain of 8.19, with no adjusted statistical support. |
| ptonet_D | ptonet_Original | 47.91 | 46.96 | +0.95 | [-8.12, +9.71] | 1.0000 | Estimated gain of 0.95, with no adjusted statistical support. |
| ptonet_ABCD | ptonet_ABC | 55.16 | 55.82 | -0.66 | [-10.85, +9.76] | 1.0000 | Estimated decline of 0.66, with no adjusted statistical support. |
| ptonet_A | causalpfn_Original | 52.56 | 50.61 | +1.95 | [-14.18, +16.90] | 1.0000 | Estimated gain of 1.95, with no adjusted statistical support. |

## Removing one feature block from ABCD

| Host | Removed component | ABCD value | Reduced value | ABCD gain | Paired 95% interval |
|---|---|---:|---:|---:|---|
| causalpfn | A | 51.02 | 53.27 | -2.25 | [-21.30, +15.42] |
| causalpfn | B | 51.02 | 49.37 | +1.66 | [-14.73, +16.57] |
| causalpfn | C | 51.02 | 52.55 | -1.52 | [-16.41, +13.64] |
| causalpfn | D | 51.02 | 56.17 | -5.15 | [-21.57, +11.01] |
| ptonet | A | 55.16 | 58.65 | -3.49 | [-14.18, +7.40] |
| ptonet | B | 55.16 | 53.04 | +2.12 | [-8.15, +12.02] |
| ptonet | C | 55.16 | 52.38 | +2.78 | [-5.81, +11.83] |
| ptonet | D | 55.16 | 55.82 | -0.66 | [-10.85, +9.76] |

The A, B and C removal contrasts are exploratory. Each contrast concerns the complete feature block and its inputs.

## Average effect of adding each component

| Host | Component | Matched configuration pairs | Mean gain at 30% | Paired 95% interval |
|---|---|---:|---:|---|
| causalpfn | A | 8 | +0.98 | [-10.01, +11.25] |
| causalpfn | B | 8 | -2.64 | [-8.84, +2.98] |
| causalpfn | C | 8 | +0.60 | [-5.42, +6.83] |
| causalpfn | D | 8 | -0.18 | [-6.25, +6.08] |
| ptonet | A | 8 | +2.45 | [-5.75, +10.34] |
| ptonet | B | 8 | +2.43 | [-1.17, +5.83] |
| ptonet | C | 8 | +2.30 | [-1.01, +6.19] |
| ptonet | D | 8 | +0.70 | [-2.34, +3.91] |

These exploratory estimates average matched configuration gains across backgrounds with the other components held fixed.

## Full grid and formula controls

| Host | Configuration | 10% value | 20% value | 30% value | 50% value | 30% gain vs Original | Paired 95% interval | 30% seed SD | Host GPU-hours |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| causalpfn | Original | 47.01 | 45.74 | 50.61 | 37.28 | +0.00 | [+0.00, +0.00] | — | 0.0540 |
| causalpfn | A | 111.72 | 69.02 | 55.41 | 45.48 | +4.80 | [-14.22, +21.78] | — | 0.0864 |
| causalpfn | B | 68.71 | 59.03 | 45.25 | 39.23 | -5.36 | [-22.90, +9.30] | — | 0.0504 |
| causalpfn | C | 66.38 | 54.03 | 60.10 | 54.44 | +9.49 | [-7.55, +25.42] | — | 0.0551 |
| causalpfn | D | 92.83 | 69.61 | 56.39 | 46.33 | +5.79 | [-10.89, +21.05] | — | 0.0512 |
| causalpfn | AB | 82.47 | 63.60 | 54.83 | 46.76 | +4.22 | [-14.57, +22.21] | — | 0.0838 |
| causalpfn | AC | 80.80 | 61.04 | 57.59 | 48.40 | +6.99 | [-11.40, +24.36] | — | 0.0870 |
| causalpfn | AD | 83.72 | 67.79 | 58.06 | 43.89 | +7.46 | [-11.54, +25.11] | — | 0.0854 |
| causalpfn | BC | 79.36 | 58.02 | 51.85 | 48.85 | +1.24 | [-15.68, +17.51] | — | 0.0529 |
| causalpfn | BD | 92.16 | 69.37 | 55.58 | 46.53 | +4.97 | [-11.60, +21.39] | — | 0.0520 |
| causalpfn | CD | 67.71 | 49.99 | 54.12 | 47.83 | +3.52 | [-13.25, +19.95] | — | 0.0543 |
| causalpfn | ABC | 72.01 | 56.75 | 56.17 | 41.72 | +5.56 | [-13.21, +22.86] | — | 0.0858 |
| causalpfn | ABD | 88.74 | 63.20 | 52.55 | 41.20 | +1.94 | [-17.71, +19.35] | — | 0.0864 |
| causalpfn | ACD | 91.35 | 55.89 | 49.37 | 39.91 | -1.24 | [-20.76, +16.59] | — | 0.0884 |
| causalpfn | BCD | 71.21 | 57.76 | 53.27 | 47.41 | +2.66 | [-14.30, +19.69] | — | 0.0551 |
| causalpfn | ABCD | 85.41 | 55.05 | 51.02 | 44.96 | +0.41 | [-19.08, +18.07] | — | 0.0888 |
| causalpfn | D_random | 83.43 | 68.55 | 55.34 | 48.24 | +4.73 | [-11.09, +19.59] | — | 0.0512 |
| causalpfn | D_shuffled | 54.90 | 48.18 | 46.04 | 46.64 | -4.57 | [-21.16, +10.76] | — | 0.0527 |
| ptonet | Original | 69.99 | 50.00 | 46.96 | 45.47 | +0.00 | [+0.00, +0.00] | 7.92 | 0.0000 |
| ptonet | A | 96.48 | 70.84 | 52.56 | 43.12 | +5.60 | [-5.78, +16.55] | 10.98 | 0.0000 |
| ptonet | B | 56.03 | 48.59 | 48.52 | 41.99 | +1.55 | [-7.69, +10.90] | 11.62 | 0.0000 |
| ptonet | C | 77.50 | 64.07 | 55.82 | 42.35 | +8.85 | [-1.09, +19.53] | 1.84 | 0.0000 |
| ptonet | D | 81.01 | 59.65 | 47.91 | 39.60 | +0.95 | [-8.12, +9.71] | 6.54 | 0.0000 |
| ptonet | AB | 82.47 | 63.93 | 55.36 | 41.98 | +8.39 | [-3.40, +19.71] | 9.21 | 0.0000 |
| ptonet | AC | 84.33 | 60.98 | 47.17 | 39.04 | +0.20 | [-10.18, +11.21] | 7.53 | 0.0000 |
| ptonet | AD | 86.44 | 69.63 | 53.54 | 46.59 | +6.58 | [-4.06, +16.81] | 3.49 | 0.0000 |
| ptonet | BC | 71.55 | 58.36 | 50.24 | 40.49 | +3.28 | [-5.42, +12.50] | 5.28 | 0.0000 |
| ptonet | BD | 75.12 | 62.48 | 48.82 | 41.70 | +1.85 | [-7.47, +11.27] | 5.06 | 0.0000 |
| ptonet | CD | 75.27 | 56.79 | 48.52 | 36.89 | +1.56 | [-7.21, +11.30] | 5.17 | 0.0000 |
| ptonet | ABC | 97.41 | 66.02 | 55.82 | 44.95 | +8.85 | [-1.57, +20.22] | 6.84 | 0.0000 |
| ptonet | ABD | 85.89 | 69.94 | 52.38 | 41.92 | +5.41 | [-5.82, +16.20] | 10.69 | 0.0000 |
| ptonet | ACD | 89.80 | 69.11 | 53.04 | 39.19 | +6.07 | [-6.05, +18.79] | 10.63 | 0.0000 |
| ptonet | BCD | 64.99 | 62.14 | 58.65 | 49.34 | +11.69 | [+2.07, +21.31] | 2.33 | 0.0000 |
| ptonet | ABCD | 81.92 | 69.13 | 55.16 | 45.97 | +8.19 | [-2.06, +18.62] | 10.77 | 0.0000 |
| ptonet | D_random | 85.62 | 64.14 | 52.67 | 43.42 | +5.70 | [-4.72, +16.80] | 3.61 | 0.0000 |
| ptonet | D_shuffled | 94.38 | 72.75 | 57.06 | 43.34 | +10.10 | [-0.04, +20.55] | 6.43 | 0.0000 |

PTONet ranks and scores each seed policy individually and averages the resulting policy values. The seed-level file includes SMS and control customer counts, buyers, normalized Qini and normalized AUUC. Bootstrap replicates draw customers with replacement within treatment arm and rank each resample with customer-ID tie breaking. All policies use the same 5,000 draws.

The random control treats subtraction as ordered. Its subtraction reverses the archived subtraction, retaining that component’s information up to sign. The shuffled block permutes rows jointly within each partition. These are exploratory controls.

| Host | Reference control | D gain at 30% | Paired 95% interval |
|---|---|---:|---|
| causalpfn | D_random | +1.05 | [-13.61, +16.41] |
| causalpfn | D_shuffled | +10.35 | [-6.10, +26.24] |
| ptonet | D_random | -4.76 | [-14.33, +4.32] |
| ptonet | D_shuffled | -9.15 | [-19.21, -0.15] |

## Post hoc comparison at 50% reach

Each policy ranks the same evaluation customers and selects 20,004 for SMS. Scores are incremental buyers per 1,000 SMS sent. The archived rankings and validation selection were fixed before this budget was requested.

| Model | Configuration | Score | Gain vs Original | Paired 95% interval for gain |
|---|---|---:|---:|---|
| PTONet | Original | 45.47 | +0.00 | Reference |
| PTONet | ABCD | 45.97 | +0.50 | [-6.92, +7.60] |
| PTONet | BCD (highest observed) | 49.34 | +3.87 | [-1.80, +9.50] |
| CausalPFN | Original | 37.28 | +0.00 | Reference |
| CausalPFN | ABCD | 44.96 | +7.68 | [-4.09, +20.48] |
| CausalPFN | C (highest observed) | 54.44 | +17.15 | [+5.74, +28.10] |

Highest observed refers to the largest evaluation score among the available core configurations. The paired intervals are pointwise and do not account for choosing that maximum. [50% estimates and paired intervals](decision-table.csv)

## Compute and incremental scoring cost

| Stage | Shared build GPU-hours | Seconds per 1,000 customers | Device | Scope |
|---|---:|---:|---|---|
| D | 0.0000 | 0.004 | CPU | formulas including six-significant-digit input rounding |
| A | 0.0104 | 0.006 | GPU | embedding forward pass with prepared basket tensors |
| B | 0.0189 | 0.003 | CPU | DR regressor with prepared six-field inputs |
| C | 7.5119 | 132.295 | GPU | two-scenario forward passes with tokenized prompts |
| causalpfn Original | — | 2.812 | GPU | original host prediction per seed |
| ptonet Original | — | 0.002 | CPU | original host prediction per seed |

B used one frozen TabPFN-3.5 estimator per outcome arm with a deterministic 2,048-customer context and empirical SMS probabilities within the fitting folds. The public competition documentation describes the treatment flag. [X5 data documentation](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)

C used the archived Qwen3 8B NF4 checkpoint, 2,048 factual prompts per LoRA adapter, and inference batches of eight. All prompts fit the 512-token cap with the archived 18 summaries and up to eight baskets.

## Standalone feature diagnostics

- B scored 96.63 at 10%, 76.26 at 20%, 63.67 at 30%, 52.13 at 50% reach.
- C scored 50.52 at 10%, 37.86 at 20%, 34.14 at 30%, 36.74 at 50% reach.

## Artifacts

- [Validation milestone](VALIDATION.md)
- [Decision table by configuration and reach](decision-table.csv)
- [Registered and exploratory paired contrasts](comparisons.csv)
- [Marginal factorial effects](factorial-effects.csv)
- [Per-seed policies and arm counts](seed-policies.csv)
- [Measured compute](compute.json)
- [Roster choice and projections](budget.json)
- [Evaluation and lock hashes](results.json)
- [Follow-up on adding the no-SMS forecast to PTONet C at 30% validation reach](FORECAST.md)

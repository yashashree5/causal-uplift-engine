# Does the no-SMS purchase forecast help PTONet target customers?

Adding the no-SMS forecast lowered PTONet's average validation score from **71.75 to 63.48** incremental buyers per 1,000 SMS at 30% reach. The paired difference was **−8.27** with a 95% interval of **[−21.59, +4.50]**. The added forecast scored lower in all three seeds. We would keep the existing C representation for PTONet on this evidence. [Validation scores](forecast-validation.csv) · [Per-seed results and timings](forecast-validation-seeds.csv)

| Validation targeting rule | Original | Current C | C plus no-SMS forecast | Gain over current C | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Average of three separately ranked seed scores | 64.40 | 71.75 | 63.48 | −8.27 | [−21.59, +4.50] |
| One list ranked by the mean of three predictions | 64.97 | 69.92 | 65.00 | −4.92 | [−24.67, +15.30] |

Each targeting rule selects 6,001 of 20,004 validation customers. The first row follows the original study's seed-averaging rule. The second shows what happens if a campaign sends messages using a single ensemble list. Both use the difference in treated and untreated purchase rates among the selected customers, multiplied by 1,000. A paired bootstrap resampled customers within each treatment arm 5,000 times and reapplied every targeting rule to the same draws.

The same validation customers were used to select learning rates and stopping epochs. The intervals describe customer sampling variability for the selected fits. Independent confirmation would require new data. Original and current C use saved validation predictions, and a separate current-C refit reproduced every archived prediction for seed 20261123. The three new fits used 66.3 summed seconds of CPU-run wall time and zero additional GPU time.

## What changed

Qwen3 8B with LoRA scores the same pre-campaign purchase history twice. Write `p0` for its estimated purchase probability without an SMS and `p1` for the estimate with an SMS. Current C supplies the difference, `p1 - p0`. The follow-up adds `p0` so PTONet can also learn from the underlying purchase likelihood. A 10% → 15% customer and a 60% → 65% customer both have an estimated five-point gain, but their no-SMS forecasts differ.

| PTONet input | Columns | Customer features |
|---|---:|---|
| Original | 6 | Six baseline fields |
| Current C | 7 | Baseline fields and `p1 - p0` |
| C plus no-SMS forecast | 8 | Baseline fields, `p1 - p0` and `p0` |

The comparison of interest is the eight-column model against current C. All three use the same 140,027 training customers and 20,004 validation customers. Each training customer's forecasts come from an adapter fitted outside that customer's fold. One adapter fitted on training data supplied the validation forecasts. The archived forecasts provide both columns without another Qwen pass. [Component C implementation](../../experiments/retailhero_abcd/qwen.py)

## Earlier frozen-forecast pilot

The earlier X5 pilot supplied two separate yes/no scores from a **frozen** LLM to each host, alongside the same numeric purchase history available to the comparator. It used 1,024 training, 256 validation and 1,024 evaluation customers. Its metric was additional purchases per 1,000 **eligible customers** above random targeting, averaged across budgets. The present follow-up uses incremental buyers per 1,000 **SMS sent** at a fixed 30% reach.

| Pilot host | Frozen LLM | Gain over matched numeric history | Pointwise 95% interval | Holm-adjusted p |
|---|---|---:|---:|---:|
| PTONet | Qwen3 8B | −2.53 | [−16.32, +11.25] | 1.000 |
| PTONet | Gemma4 12B | +13.19 | [−6.62, +33.00] | 1.000 |
| CausalPFN | Qwen3 8B | −2.47 | [−6.01, +1.07] | 1.000 |
| CausalPFN | Gemma4 12B | −0.74 | [−3.71, +2.24] | 1.000 |

These four forecast comparisons belonged to the pilot’s 18-comparison family. The present comparison keeps the LoRA adapters fixed and tests how much of their forecast information to give PTONet. [Archived pilot contrasts](forecast-frozen-pilot.csv)

# X5 RetailHero SMS uplift results

Explore the configuration scores, budget profiles and paired comparisons in the [ABCD results notebook](../../notebooks/retailhero_abcd_analysis.ipynb).

A frozen Qwen3 8B estimates the SMS effect directly from each customer's purchase history. The follow-up evaluates its ranking alongside Original and ABCD for both PTONet and CausalPFN. [Standalone LLM comparison](DIRECT_LLM.md)

At 30% reach, PTONet with all four components gained an estimated **8.19 incremental buyers per 1,000 SMS** over original PTONet. CausalPFN with the same components gained **0.41** over original CausalPFN. Both paired 95% intervals include zero. Across all seven prespecified comparisons, none established a gain after the Holm correction. [Registered comparisons and full report](REPORT.md)

| Model at 30% reach | Original score | ABCD score | ABCD gain | Paired 95% interval for gain | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| PTONet | 46.96 | 55.16 | +8.19 | [−2.06, +18.62] | 0.8594 |
| CausalPFN | 50.61 | 51.02 | +0.41 | [−19.08, +18.07] | 1.0000 |

## Reading the scores

**Reach** is the fraction of evaluation customers a model would target. At 30% reach it ranks all 40,008 customers and selects the top 30%. Within this selected group, we subtract the no-SMS purchase rate from the SMS purchase rate and multiply by 1,000. A score of 55.16 means an estimated 55 additional buyers per 1,000 SMS sent. The score measures the *rate* of additional purchases, so a narrower budget can have a higher score while targeting fewer people. Gains are calculated before rounding the displayed scores. [Decision table](decision-table.csv)

The study used 140,027 training customers and 20,004 validation customers. Evaluation reused a cohort from earlier work, so the findings are exploratory. The primary budget was 30%. The 10% and 20% budgets were prespecified secondary analyses. The 50% budget was added after reviewing the original results, using the same locked rankings. PTONet scores average the policy values from three seeds. Gains use 5,000 paired bootstrap samples stratified by treatment arm. [Evaluation metadata](results.json)

## Configurations and controls

Both models start with the same six pre-SMS customer fields. Each can use any subset of four components, giving 16 configurations including Original and ABCD.

| Component | What it supplies |
|---|---|
| A | A 16-number embedding from an encoder trained on purchase sequences. |
| B | One uplift score from a doubly robust learner using tabular purchase-probability models. |
| C | One uplift score from Qwen3 8B with LoRA, computed from its predictions under SMS and no SMS. |
| D | Four fixed numerical formulas from an earlier Qwen run. Each customer's formula values are calculated directly from the input fields. |

The earlier Qwen run repeated the four worked examples in its prompt. D therefore tests the usefulness of those fixed features. Independent formula discovery remains an open question.

The two additional controls test D. **Random formulas** replace the four formulas with alternatives using the same operators. One alternative reverses the original subtraction and preserves that signal up to its sign. **Shuffled formulas** permute all four formula values together across customers within each split. [Formula validation](VALIDATION.md)

The grid chart shows all 144 model/configuration/budget scores. The gains chart compares Original, ABCD and the highest-scoring core configuration at each budget. Choosing the highest observed score on the evaluation cohort makes that result optimistic. Its interval is pointwise and does not adjust for selecting the winner. Validation selected PTONet A before evaluation. [Configuration scores](figures/configuration_scores.png) · [Gains and intervals](figures/full_model_gains.png)

## Files for review

| File | What to check |
|---|---|
| [Full report](REPORT.md) | Registered comparisons, component removals, formula controls, standalone B and C, costs and interpretation. |
| [Decision table](decision-table.csv) | 144 rows covering 18 configurations and controls per model across all four budgets. Each row includes the absolute score and the gain versus that model's Original. |
| [Paired comparisons](comparisons.csv) | 220 comparisons with estimates, paired uncertainty and comparison family. |
| [Factorial effects](factorial-effects.csv) | 32 average effects, each computed across eight matched configuration pairs. |
| [Per-seed policies](seed-policies.csv) | 296 rows with treatment-arm counts, purchase counts, Qini and AUUC. Includes the two standalone policies. |
| [Evaluation metadata](results.json) | Selection, cohort counts, label and prediction-lock digests, bootstrap settings and all seven registered results. |
| [Measured compute](compute.json) and [budget plan](budget.json) | Measured total of 8.991 GPU-hours and the prospective budget. Shared feature costs are charged once. |
| [Development validation](VALIDATION.md) and [validation metadata](validation.json) | Early formula-control results, distinct from the full evaluation. |

These files support review of the aggregate outcomes. Reproducing scores from individual customers requires the separately provisioned X5 cohorts, locked rankings and model artifacts. The configuration builders are available at the pinned review commit. [Configuration source and usage](https://github.com/yashashree5/causal-uplift-engine/tree/37227369dfc7d1bbd68c79eeb46b92e9b81e2fea/experiments/retailhero_abcd)

## How the results were calculated

The original evaluator ranks customers, calculates each seed’s targeting score, and uses the same 5,000 customer resamples for every policy. Its source matches the SHA-256 recorded when the predictions were locked. The 50% extension applies that evaluator to the same rankings and checks that every earlier score and bootstrap draw is preserved. [Historical evaluator](source/evaluate.py.txt) · [50% extension](source/add_top50.py.txt)

The report script summarizes the measured compute. The plotting script draws both charts from the decision table and configuration design. These source snapshots preserve the exact Python used in the study. [Report and compute source](source/publish.py.txt) · [Chart source](source/plot_results.py.txt)

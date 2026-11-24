# X5 purchase uplift configurations

PTONet and CausalPFN receive the same customer inputs for each configuration. Original contains the six pre-SMS X5 summary fields. The 15 other configurations append every nonempty subset of A, B, C and D in that order. The full ABCD matrix has 28 columns. Two additional formula controls use the same six baseline columns and four control columns. The [design](design.json) lists the names, dimensions and training seeds.

| Component | New columns | Construction |
|---|---:|---|
| A | 16 | Pretrain a four-layer causal decoder on up to 32 chronological baskets to predict the next spend decile. Freeze the encoder and use the last observed basket's embedding. Exclude validation and evaluation customers from pretraining. |
| B | 1 | Use two TabPFN-3.5 outcome models to estimate purchase probabilities under SMS and no SMS. Fit a gradient boosting regressor to the resulting cross-fitted doubly robust training targets. |
| C | 1 | Fit a rank-8 LoRA adapter on Qwen3 8B with 2,048 factual examples per fit. Score the same history twice and subtract the no-SMS purchase probability from the SMS purchase probability. |
| D | 4 | Compute four arithmetic features from the baseline loyalty fields. Qwen3 8B returned these expressions from the worked examples in an earlier prompt. The pinned [formula file](programs.json) is sufficient to calculate D. |

## Column and split conventions

The raw baseline field order is `redeem_day`, `express_mean`, `express_total`, `points_month`, `spent_day`, `spent_after_redeem_day`. Use pre-SMS information with a March 19, 2019 cutoff. Numeric imputation uses only the training rows, with zero as the fallback for an entirely missing column. For D, first round its operands to six significant digits to match the original prompt. Calculate in float64 and store the four outputs as float32.

The four fixed expressions are `redeem_day + express_mean`, `express_total - points_month`, `redeem_day * express_total`, and `express_mean / (1 + abs(points_month))`. The random control changes the operand pairs once using seed 91427. The shuffled control jointly permutes the four fixed formula values across customers within each partition.

Keep customer order aligned across the baseline and every feature bank. Use three treatment-stratified folds to create out-of-fold B and C scores for training customers. Validation and test predictions use separate fits on the training population. Use the same partitions for every host and configuration. The experiment used 140,027 training, 20,004 validation and 40,008 evaluation customers. Validation ranked policies by additional buyers per 1,000 SMS at 30% reach. The 10% and 20% budgets were secondary. The 50% budget was added during follow-up evaluation. [X5 competition](https://ods.ai/competitions/x5-retailhero-uplift-modeling)

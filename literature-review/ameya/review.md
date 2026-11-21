# Literature Review: Ameya Khond

**Focus:** Causal estimation, uplift objectives, imbalance, uncertainty, and treatment allocation.

**Status:** first-pass draft. Exact numerical results and table references require full-text verification before submission. U1–U8 are identifiers local to this review.

## Papers I took

| ID | Paper | Type |
|---|---|---|
| U1 | Do-PFN: In-Context Learning for Causal Effect Estimation (Robertson et al., 2025) | Causal foundation model |
| U2 | Causal Machine Learning for Predicting Treatment Outcomes (Feuerriegel et al., 2024) | Methods review |
| U3 | Uplift Modeling with Continuous Treatments: A Predict-Then-Optimize Approach (De Vos et al., 2024) | Treatment allocation |
| U4 | Class Flipping for Uplift Modeling and Heterogeneous Treatment Effect Estimation on Imbalanced RCT Data (Rudaś & Jaroszewicz, 2024) | Imbalance correction |
| U5 | The Impact of Heteroskedasticity on Uplift Modeling (Bokelmann & Lessmann, 2023) | Ranking diagnostics |
| U6 | Propensity Score Oversampling and Matching for Uplift Modeling (Vairetti, Gennaro & Maldonado, 2024) | Observational uplift |
| U7 | Partial Counterfactual Identification and Uplift Modeling: Theoretical Results and Real-World Assessment (Verhelst et al., 2024 journal volume; online 2023) | Counterfactual bounds |
| U8 | Uplift Modeling with Quasi-Loss-Functions (Hu, de Haan & Skiera, 2024) | Uplift objectives |

## 1. Paper notes

### U1 Do-PFN (Robertson et al., 2025)

- **Link:** https://arxiv.org/abs/2506.06039
- **Problem:** estimate causal effects using a pretrained model that learns from examples supplied in context.
- **Method:** applies the prior-data fitted network approach to causal effect estimation. A transformer learns from simulated causal tasks so that it can make predictions for a new dataset through in-context examples.
- **Results:** quantitative comparisons and benchmark details still require verification from the full paper.
- **Limitations:** the fit between simulated training tasks and real data needs examination. Large-table scalability must be measured rather than assumed.
- **Relevance to us:** a candidate causal foundation-model comparator. Evaluate its effects and resource usage using the same data splits and uplift metrics as Model 1.
- **Open questions:** how does it select context rows, and does its inference procedure already overlap with separate treatment-arm retrieval?

### U2 Causal Machine Learning for Predicting Treatment Outcomes (Feuerriegel et al., 2024)

- **Link:** https://doi.org/10.1038/s41591-024-02902-1
- **Problem:** use machine learning to predict treatment benefit while preserving the assumptions needed for causal interpretation.
- **Method:** reviews causal machine-learning approaches for treatment outcomes, discussing heterogeneous effects, identification, validation, and practical application. It connects statistical causal inference with flexible predictive models.
- **Data:** a methodological review with a healthcare focus rather than a marketing benchmark.
- **Limitations:** recommendations require adaptation to campaign outcomes and customer targeting. The review does not establish that a particular estimator will work best on our data.
- **Relevance to us:** use it to explain why predicting customer response differs from estimating incremental response, and to document randomization, overlap, and treatment definitions.
- **Open questions:** which validation practices transfer to Hillstrom and Criteo when individual counterfactual outcomes remain unobserved?

### U3 Uplift Modeling with Continuous Treatments (De Vos et al., 2024)

- **Link:** https://arxiv.org/abs/2412.09232
- **Problem:** choose a treatment dose when actions are continuous rather than simply treatment versus control.
- **Method:** first estimates conditional average dose responses, then formulates dose allocation as an integer linear optimization problem. The allocation can incorporate resource constraints, fairness considerations, and individual costs or benefits.
- **Data/results:** demonstrates the framework through applications in healthcare, lending, and human resource management; exact numerical comparisons require full-text verification.
- **Limitations:** binary campaign datasets do not by themselves identify the effects of arbitrary discount amounts or contact intensities. A continuous-treatment extension needs suitable dose data and identification assumptions.
- **Relevance to us:** a design reference for extending uplift scores into budget-constrained decisions. Continuous campaign intensity is a future extension when suitable data are available.
- **Open questions:** which costs and business constraints can be specified for our current binary targeting tasks?

### U4 Class Flipping for Uplift Modeling and HTE Estimation (Rudaś & Jaroszewicz, 2024)

- **Link:** https://arxiv.org/abs/2412.10009
- **Problem:** correct treatment and outcome imbalance in randomized experiments without distorting predicted treatment effects.
- **Method:** flips selected outcome labels according to a designed transformation rather than relying on ordinary over- or undersampling. It treats transformed-outcome models separately and analyzes when the treatment-effect target is preserved.
- **Data/results:** the paper combines theoretical analysis with empirical experiments. Exact datasets and effect sizes need verification in the full text.
- **Limitations:** the transformation must follow the paper's assumptions and estimator-specific procedure; arbitrary label changes are not justified.
- **Relevance to us:** compare the method with class weights and unmodified training on imbalanced campaign data. Apply any transformation only within training folds.
- **Open questions:** does the method suit M1's separate outcome heads, or is it more appropriate for a transformed-outcome baseline?

### U5 The Impact of Heteroskedasticity on Uplift Modeling (Bokelmann & Lessmann, 2023)

- **Link:** https://arxiv.org/abs/2312.05234
- **Problem:** determine whether unequal outcome variance can distort the ranking used for treatment prioritization.
- **Method:** analyzes ranking bias theoretically and illustrates it using simulation and real data. It shows that high-effect units can accumulate near the bottom of an estimated ranking under heteroskedasticity.
- **Results:** the central finding concerns a failure of ranking-based allocation; numerical magnitudes require full-text verification.
- **Limitations:** the practical importance depends on the outcome distribution and estimator. Findings should be tested on our tasks, especially when transferring from continuous to binary outcomes.
- **Relevance to us:** add subgroup and ranking diagnostics rather than treating one aggregate Qini score as a complete assessment.
- **Open questions:** how stable are the highest-ranked customer groups across folds, seeds, and outcome definitions?

### U6 Propensity Score Oversampling and Matching for Uplift Modeling (Vairetti et al., 2024)

- **Link:** https://doi.org/10.1016/j.ejor.2024.03.024
- **Problem:** estimate uplift in small or imbalanced datasets where treatment assignment may be confounded.
- **Method:** ProSOM combines propensity-score matching with class-aware oversampling to improve comparability between treatment groups. It extends matching to settings with limited data and treatment imbalance.
- **Data/results:** evaluates the approach on benchmark datasets; exact values and table references require checking.
- **Limitations:** matching on observed variables cannot eliminate unmeasured confounding. Synthetic examples also require a clear definition of valid covariates and outcomes.
- **Relevance to us:** an optional observational-data experiment. For randomized campaign data, preserve the original experiment as the main benchmark and test whether preprocessing changes the estimand or evaluation distribution.
- **Open questions:** does matching discard useful overlap, and what effective sample size remains after preprocessing?

### U7 Partial Counterfactual Identification and Uplift Modeling (Verhelst et al., 2024)

- **Link:** https://doi.org/10.1007/s10994-023-06317-w
- **Problem:** determine what uplift estimates reveal about joint factual and counterfactual outcomes.
- **Method:** derives bounds on counterfactual probabilities expressed through uplift-related quantities. It also studies a point estimator under conditional independence of potential outcomes and evaluates the approach with simulations and customer data.
- **Data:** synthetic experiments and a telecom customer campaign dataset.
- **Limitations:** the point estimate requires an additional assumption about potential outcomes. Identification bounds are conceptually different from sampling confidence intervals.
- **Relevance to us:** support careful interpretation of customer-level scores. A predicted CATE does not reveal whether a particular customer would respond only under treatment.
- **Open questions:** can the dashboard report segment-level bounds without presenting them as individual ground truth?

### U8 Uplift Modeling with Quasi-Loss-Functions (Hu et al., 2024)

- **Link:** https://doi.org/10.1007/s10618-024-01042-x
- **Problem:** train toward uplift when individual treatment effects cannot be observed as ordinary supervised labels.
- **Method:** proposes four quasi-loss functions for uplift estimation and evaluates UpliftQL models on simulated and empirical data with binary and continuous outcomes.
- **Results:** the authors report competitive performance among the compared models. Exact table-level numerical comparisons remain to be checked.
- **Limitations:** a different loss does not solve confounding or reveal individual counterfactuals. Applying it to a transformer with separate heads requires checking the mathematical objective.
- **Relevance to us:** compare an uplift-oriented loss against factual outcome prediction while holding the encoder, folds, and tuning budget fixed.
- **Open questions:** which loss is appropriate for the visit outcome, and does it improve held-out uplift ranking consistently?

## 2. New papers found

| Paper | Link | Year | Why it matters |
|---|---|---|---|
| Rudaś & Jaroszewicz, **Class Flipping for Uplift Modeling and Heterogeneous Treatment Effect Estimation on Imbalanced RCT Data** [U4; shared ID F2] | https://arxiv.org/abs/2412.10009 | 2024 | An imbalance experiment that checks whether correction preserves the treatment-effect target |
| De Vos et al., **Uplift Modeling with Continuous Treatments: A Predict-Then-Optimize Approach** [U3; shared ID F3] | https://arxiv.org/abs/2412.09232 | 2024 | Connects estimated treatment benefit with constrained allocation |
| Hu et al., **Uplift Modeling with Quasi-Loss-Functions** [U8; shared ID F4] | https://doi.org/10.1007/s10618-024-01042-x | 2024 | A candidate alternative to factual prediction losses for Model 1 |

These three papers are included in the shared `../READING_LIST.md` as F2–F4.

## 3. Synthesis

**What is already solved?** Causal foundation models offer in-context effect estimation [U1], while causal-ML methodology explains the assumptions behind treatment-benefit prediction [U2]. The other papers address specific pieces of the pipeline: imbalance [U4, U6], ranking reliability [U5], counterfactual interpretation [U7], uplift objectives [U8], and constrained allocation [U3].

**What is still open?** These papers motivate experiments rather than establish that one combined system will outperform existing approaches. For our project, the open questions are whether separate treated/control retrieval improves uplift on large tables, which training objective works best, and whether gains survive subgroup checks and repeated runs.

**Does anything already do what our Model 1 does?** This first-pass review does not establish novelty. Do-PFN [U1] is a direct causal-model comparison whose context construction must be checked in detail. The other papers focus on losses, data correction, interpretation, or allocation rather than demonstrating the entire proposed architecture. A full-text comparison and citation search are necessary before claiming that the combination of a tabular transformer, separate treatment-arm retrieval, and causal outcome heads is new.

**Risks this review uncovered:**

- Label transformations or oversampling can change the target unless implemented under the relevant assumptions.
- Neighbor retrieval, preprocessing, and synthetic-data generation must use training folds only.
- Overall uplift scores can hide unstable rankings or weak subgroup performance.
- CATE estimates, counterfactual identification bounds, and sampling confidence intervals answer different questions.
- Continuous-dose allocation requires data beyond binary treatment assignments.

## 4. Related-work paragraph for the report

Recent causal machine-learning work distinguishes treatment-benefit estimation from ordinary response prediction [U2], and Do-PFN studies causal estimation through in-context learning [U1]. Reliable uplift targeting also depends on the training objective and data distribution. Quasi-loss functions provide objectives designed for uplift [U8], class flipping addresses imbalance in randomized data [U4], and propensity-score oversampling and matching address comparability in observational settings [U6]. Unequal outcome variance can distort uplift rankings [U5], while partial counterfactual identification clarifies what those scores can and cannot reveal about individual response [U7]. Predict-then-optimize methods connect effect estimates with constrained treatment allocation [U3]. These findings motivate our comparison of Model 1's retrieval and loss choices, together with leakage checks, subgroup evaluation, and explicit assumptions for downstream targeting.

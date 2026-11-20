# Reading List

Every arXiv ID below was checked against arXiv's records. Add new papers with a PR; include the link and which theme they belong to.

**Core** papers are read by everyone. The other papers are split by role (see [README.md](README.md#who-reads-what)).

## A. The baseline and in-context (PFN) models

| ID | Paper | Venue | Why we read it |
|---|---|---|---|
| A1 | Balazadeh et al., **CausalPFN: Amortized Causal Effect Estimation via In-Context Learning** — [arXiv 2506.07918](https://arxiv.org/abs/2506.07918) | NeurIPS 2025 | **Core.** Our baseline. Read Sections 3, 5, 7 and Appendix D.2, D.6 (retrieval, large-table drop) |
| A2 | Hollmann et al., **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** — [arXiv 2207.01848](https://arxiv.org/abs/2207.01848) | ICLR 2023 | The model family CausalPFN builds on |
| A3 | Robertson et al., **Do-PFN: In-Context Learning for Causal Effect Estimation** — [arXiv 2506.06039](https://arxiv.org/abs/2506.06039) | 2025 | Closest competitor to CausalPFN; check whether it handles large tables |
| A4 | Ye et al., **A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities** — [arXiv 2502.17361](https://arxiv.org/abs/2502.17361) | 2025 | How PFN-style models behave on large tables |
| A5 | Thomas et al., **Retrieval & Fine-Tuning for In-Context Tabular Models** (LoCalPFN) — [arXiv 2406.05207](https://arxiv.org/abs/2406.05207) | NeurIPS 2024 | **Core.** Retrieval + fine-tuning for TabPFN. Closest idea to our Model 1; we must explain how we differ |

## B. Uplift and treatment-effect estimation

| ID | Paper | Venue | Why we read it |
|---|---|---|---|
| B1 | Künzel et al., **Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning** — [arXiv 1706.03461](https://arxiv.org/abs/1706.03461) | PNAS 2019 | **Core.** S-, T-, X-Learner: the comparison methods in the paper's tables |
| B2 | Wager & Athey, **Estimation and Inference of Heterogeneous Treatment Effects using Random Forests** — [arXiv 1510.04342](https://arxiv.org/abs/1510.04342) | JASA 2018 | Causal forests; standard strong baseline |
| B3 | Nie & Wager, **Quasi-Oracle Estimation of Heterogeneous Treatment Effects** (R-learner) — [arXiv 1712.04912](https://arxiv.org/abs/1712.04912) | Biometrika 2021 | Theory behind modern CATE losses |
| B4 | Shalit et al., **Estimating Individual Treatment Effect: Generalization Bounds and Algorithms** (TARNet/CFR) — [arXiv 1606.03976](https://arxiv.org/abs/1606.03976) | ICML 2017 | **Core.** Shared encoder + two outcome heads: the design of our M1a |
| B5 | Shi et al., **Adapting Neural Networks for the Estimation of Treatment Effects** (DragonNet) — [arXiv 1906.02120](https://arxiv.org/abs/1906.02120) | NeurIPS 2019 | Neural uplift model with a propensity head |
| B6 | Zhang et al., **Cross-Head Attention Uplift Network with Inverse Propensity Score under Unobserved Confounding** (CHAUN) — [arXiv 2606.27114](https://arxiv.org/abs/2606.27114) | 2026 preprint | Attention between a customer’s two potential-outcome representations |

## C. Deep learning and retrieval for tables

| ID | Paper | Venue | Why we read it |
|---|---|---|---|
| C1 | Gorishniy et al., **Revisiting Deep Learning Models for Tabular Data** (FT-Transformer) — [arXiv 2106.11959](https://arxiv.org/abs/2106.11959) | NeurIPS 2021 | **Core.** Feature tokenizer + transformer: our M1 encoder |
| C2 | Gorishniy et al., **TabR: Tabular Deep Learning Meets Nearest Neighbors** — [arXiv 2307.14338](https://arxiv.org/abs/2307.14338) | ICLR 2024 | Retrieval-augmented tabular model: the idea behind M1b, for ordinary prediction |
| C3 | Liu et al., **Explicit Feature Interaction-aware Uplift Network for Online Marketing** (EFIN) — [arXiv 2306.00315](https://arxiv.org/abs/2306.00315) | KDD 2023 | Tests how feature self-attention and treatment-aware attention affect the ranking of Criteo customers by visit uplift |
| C4 | Wei et al., **Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth** (MTMT) — [arXiv 2408.12803](https://arxiv.org/abs/2408.12803) | 2024 preprint | Uses attention between treatment and customer features and evaluates the single-treatment model on Criteo visits |

## D. Uplift evaluation and datasets

| ID | Paper | Venue | Why we read it |
|---|---|---|---|
| D1 | Gutierrez & Gérardy, **Causal Inference and Uplift Modelling: A Review of the Literature** — [PMLR v67](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf) | PAPIs 2016 (PMLR 2017) | **Core.** Qini curve definition used in the authors' code |
| D2 | Diemert et al., **A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling** — [arXiv 2111.10106](https://arxiv.org/abs/2111.10106) | 2021 | The Criteo dataset paper: how it was collected, what `exposure` means |
| D3 | Singh, **A Large-Scale Empirical Comparison of Meta-Learners and Causal Forests for Heterogeneous Treatment Effect Estimation in Marketing Uplift Modeling** — [arXiv 2604.06123](https://arxiv.org/abs/2604.06123) | 2026 | Recent results on full Criteo; numbers to compare against |
| D4 | Hillstrom, **MineThatData E-Mail Analytics and Data Mining Challenge** (2008) | Blog / dataset | Source and description of the Hillstrom data |
| D5 | Zhu et al., **Rethinking Causal Ranking: A Balanced Perspective on Uplift Model Evaluation** (PTONet / PUC) — [PMLR v267](https://proceedings.mlr.press/v267/zhu25s.html) | ICML 2025 | Principled Uplift Curve for evaluation and a related loss for binary uplift models |
| D6 | Bokelmann & Lessmann, **Heteroscedasticity-aware stratified sampling to improve uplift modeling** (HS sampling) — [Journal article](https://doi.org/10.1016/j.ejor.2025.02.030) | EJOR 2025 | Oversampling high-variance strata at a fixed experimental sample size |
| D7 | Imai & Li, **Experimental Evaluation of Individualized Treatment Rules** — [Paper](https://imai.fas.harvard.edu/research/files/indtreat.pdf) | JASA 2023 (online 2021) | Defines the gain over random targeting and the difference between two targeting rules at the same budget |

## E. LLMs and experiments (Semester 2 context)

| ID | Paper | Venue | Why we read it |
|---|---|---|---|
| E1 | Ruan et al., **Can Language Models Boost the Power of Randomized Experiments Without Statistical Bias?** (CALM) — [arXiv 2510.05545](https://arxiv.org/abs/2510.05545) | 2025 | LLM predictions to reduce variance in experiments (Model 4) |
| E2 | Midavaine et al., **LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation** — [arXiv 2610.37570](https://arxiv.org/abs/2610.37570) | 2026 | Fine-tuned LLM for uplift in industry (Model 4) |
| E3 | Xu et al., **Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation** (CURL) — [arXiv 2607.26599](https://arxiv.org/abs/2607.26599) | 2026 preprint | Frozen LLM embeddings for selected uncertain examples, with a Hillstrom benchmark |
| E4 | Huynh, Piskorz et al., **Improving Treatment Effect Estimation with LLM-Based Data Augmentation** (GATE) — [Paper](https://openreview.net/pdf?id=2Q3gFNbpAr) | ICML 2025 workshop on Foundation Models for Structured Data | Tests whether synthetic counterfactual labels improve treatment-effect estimates on a biased Hillstrom sample |
| E5 | Arbour et al., **AI-Assisted Variance Reduction in Randomized Experiments** — [arXiv 2606.08853](https://arxiv.org/abs/2606.08853) | KDD 2026 | Compares LLM visit forecasts with ridge regression and XGBoost for estimating the Hillstrom email effect more precisely |

## Search queries for new papers

Run these on Google Scholar, arXiv, and Semantic Scholar. Check papers from 2024 onwards first.

- `CausalPFN` (who cites it: use "Cited by")
- `in-context learning treatment effect estimation`
- `retrieval augmented uplift modeling`
- `tabular foundation model large tables context length`
- `uplift modeling transformer`
- `Criteo uplift benchmark` and `Hillstrom uplift`
- `heterogeneous treatment effect neural network nearest neighbors`

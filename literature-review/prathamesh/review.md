# Literature Review: Causal Uplift Engine

**Status:** first pass. Each paper's notes are drawn from its abstract and the sections cited below. A more thorough second pass will follow as the model implementation requires it.

## Papers

| ID | Paper | Type |
|---|---|---|
| A6 | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (Qu et al., 2025) | Core |
| A7 | TabDPT: Scaling Tabular Foundation Models on Real Data (Ma et al., NeurIPS 2025) | Core |
| A8 | TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (Gorishniy et al., ICLR 2025) | Baseline |
| A9 | TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (Grinsztajn et al., 2025/2026) | Baseline |
| B5 | Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation (Chauhan et al., AISTATS 2024) | Core |
| B6 | Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer (Shirakawa et al., ICML 2024) | Core |
| B7 | Statistical Learning for Heterogeneous Treatment Effects: Pretraining, Prognosis, and Prediction (Schuessler et al., 2025) | Core |
| C5 | Deep Disentangled Representation Network for Treatment Effect Estimation (Meng et al., 2025) | Core |

## 1. Paper notes

### A6 TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (Qu et al., 2025)

- **Link:** https://arxiv.org/abs/2502.05564
- **Problem:** in-context learning with transformers grows costly once every training row has to take part in attention.
- **Method:** TabICL applies column- and row-wise attention to build fixed-dimensional row representations, then runs a transformer over those representations. It is pretrained on synthetic classification tasks with datasets of up to 60,000 rows and is built to handle larger tables at inference time.
- **Results:** TabICL matches TabPFNv2 while running faster on the TALENT benchmark, and it does especially well on datasets exceeding 10,000 rows.
- **Limitations:** it is a predictive classification model rather than a causal or uplift estimator.
- **Relevance to us:** the two-stage row representation offers a way to ease CausalPFN's context-size limitation. M1 could adopt a similar compact encoder while retrieving treated and control neighbors separately.
- **Open questions:** whether row representations trained for prediction stay useful for capturing treatment heterogeneity and Qini performance.

### A7 TabDPT: Scaling Tabular Foundation Models on Real Data (Ma et al., NeurIPS 2025)

- **Link:** https://arxiv.org/abs/2410.18164 · conference version: https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html
- **Problem:** tabular foundation models have to generalize to unseen datasets, yet synthetic-only pretraining and full-table attention cap their scalability.
- **Method:** TabDPT pairs retrieval-based in-context learning with self-supervised column masking and pretraining on real tabular data. It handles both classification and regression without task-specific fine-tuning.
- **Results:** the paper shows strong results on OpenML-CC18 classification and OpenML-CTR23 regression, and reports gains as model size and the volume of real pretraining data grow.
- **Limitations:** it is not trained to estimate potential outcomes or uplift, and its retrieval context keeps treated and untreated units together.
- **Relevance to us:** TabDPT offers a helpful retrieval-plus-pretraining design for Model 1's large-table problem and the foundation-model direction in Model 4.
- **Open questions:** whether retrieval and self-supervised pretraining on campaign data can sharpen uplift estimation without creating treatment leakage.

### A8 TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (Gorishniy et al., ICLR 2025)

- **Link:** https://arxiv.org/abs/2410.24210 · conference version: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html
- **Problem:** elaborate tabular transformers can be costly, whereas plain neural networks tend to overfit heterogeneous tabular data.
- **Method:** TabM builds on an MLP backbone with parameter-efficient BatchEnsemble-style components, so a single model yields several related predictions that are combined into an ensemble.
- **Results:** TabM reports strong performance and a favorable accuracy-efficiency balance across tabular deep-learning benchmarks.
- **Limitations:** it is not a causal model and does not estimate separate potential outcomes on its own.
- **Relevance to us:** TabM is a practical neural baseline for M1. It can be recast as an S-Learner or T-Learner and compared against the retrieval transformer using matching folds and Qini evaluation.
- **Open questions:** whether parameter-efficient ensembling can sharpen uplift ranking when treatment groups are imbalanced, particularly on Criteo.

### A9 TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (Grinsztajn et al., 2025/2026)

- **Link:** https://arxiv.org/abs/2511.08667
- **Problem:** tabular foundation models have to accommodate larger tables and more features while keeping inference practical.
- **Method:** TabPFN-2.5 scales the TabPFN family to larger tables and wider feature spaces, and adds distillation into compact MLP and tree-ensemble models.
- **Results:** the report evaluates on TabArena and shows strong results against tuned tree models and AutoGluon, handling tables of roughly 50,000 rows and 2,000 features.
- **Limitations:** it is a general predictive model rather than an uplift estimator, and its reported results still need testing on the project's marketing datasets.
- **Relevance to us:** it serves as a current foundation-model baseline for smaller Hillstrom and Criteo subsets, and its scaling and distillation ideas also bear on deploying the final model.
- **Open questions:** whether a tabular foundation model can supply useful representations for treated/control retrieval without causal fine-tuning.

### B5 Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation (Chauhan et al., AISTATS 2024)

- **Link:** https://proceedings.mlr.press/v238/chauhan24a.html
- **Problem:** treatment-effect learners often divide scarce data into separate treatment groups, which shrinks the information each model can use.
- **Method:** the paper introduces learned soft weight sharing, letting treatment-specific learners exchange information dynamically while keeping their treatment-specific behavior.
- **Results:** the method improves ITE estimation relative to fully separate or fixed-sharing treatment learners, especially when treatment groups are small.
- **Limitations:** it is not a retrieval transformer and does not directly tackle very large marketing tables.
- **Relevance to us:** it suggests a useful way to share information between M1's treated and control branches while keeping separate outcome heads.
- **Open questions:** whether sharing the retrieval-attention block lifts Qini when Criteo's treatment assignment is heavily imbalanced.

### B6 Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer (Shirakawa et al., ICML 2024)

- **Link:** https://proceedings.mlr.press/v235/shirakawa24a.html
- **Problem:** deep models for counterfactual prediction can be biased and may not deliver statistical inference for repeated treatments.
- **Method:** Deep LTMLE combines a transformer with heterogeneous type embeddings and temporal-difference learning, then applies targeted minimum loss-based estimation for bias correction and confidence intervals.
- **Results:** the paper reports better counterfactual estimation and inference for longitudinal treatment policies.
- **Limitations:** Hillstrom and the main Criteo task are single-period randomized experiments, so the full longitudinal method falls outside the current implementation scope.
- **Relevance to us:** it points to a future architecture for repeated campaigns, exposure histories, and sequential targeting in Models 2 and 3.
- **Open questions:** how treatment embeddings and targeted correction might carry over from sequential campaigns to multiple campaign arms.

### B7 Statistical Learning for Heterogeneous Treatment Effects: Pretraining, Prognosis, and Prediction (Schuessler, Sverdrup & Tibshirani, 2025)

- **Link:** https://arxiv.org/abs/2505.00310
- **Problem:** CATE estimation is hard when covariates are plentiful but the treatment-effect signal is weak.
- **Method:** the authors pretrain prognostic outcome structure and fold it into an R-learner-style residualized objective, using baseline outcome risk to strengthen treatment-effect estimation.
- **Results:** the paper reports lower estimation error and better detection of treatment-effect heterogeneity when prognostic and effect-modifying signals are related.
- **Limitations:** the gain hinges on how closely baseline prognosis tracks treatment-effect heterogeneity.
- **Relevance to us:** M1's shared encoder could be pretrained first for factual outcome prediction and then fine-tuned for uplift, giving a direct ablation against random initialization.
- **Open questions:** whether outcome pretraining lifts Qini on Hillstrom and Criteo or only improves factual BCE.

### C5 Deep Disentangled Representation Network for Treatment Effect Estimation (Meng et al., 2025)

- **Link:** https://arxiv.org/abs/2507.06650
- **Problem:** pre-treatment features can play distinct roles in treatment assignment, baseline outcome prediction, and treatment-effect prediction.
- **Method:** the model combines mixture-of-experts, multi-head attention, an orthogonality regularizer, and importance reweighting to learn separate representation components.
- **Results:** the paper reports gains over treatment-effect baselines on semi-synthetic and production datasets.
- **Limitations:** the main experiments rely on observational data, and the learned representations are not guaranteed to map exactly onto true causal roles.
- **Relevance to us:** its representation design can inform an M1 ablation with separate baseline-response and uplift representations, particularly if the project later adds exposure or multiple campaign treatments.
- **Open questions:** whether disentangled representations improve uplift ranking on randomized marketing data once unnecessary observational weighting is removed.

## 2. New papers found

| Paper | Link | Year | Why it matters |
|---|---|---|---|
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | https://arxiv.org/abs/2502.05564 | 2025 | Scalable transformer architecture for large tabular contexts |
| TabDPT: Scaling Tabular Foundation Models on Real Data | https://arxiv.org/abs/2410.18164 | 2025 | Retrieval and self-supervised pretraining for tabular foundation models |
| TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling | https://arxiv.org/abs/2410.24210 | 2025 | Efficient neural baseline for tabular prediction |

## 3. Synthesis

Recent tabular foundation models tackle transformer scaling through efficient row representations, retrieval, real-data pretraining, and parameter-efficient ensembles. TabICL and TabDPT speak directly to the large-table limitation in CausalPFN, while TabM offers a strong practical neural baseline. TabPFN-2.5 serves as a current foundation-model reference for smaller tables and model distillation.

Recent causal methods contribute complementary ideas for the uplift model: dynamic information sharing can strengthen treatment-specific branches, targeted transformer estimation supports sequential treatments, prognosis-aware pretraining can improve CATE learning, and disentangled representations can separate baseline from treatment-effect signals.

The opportunity for the project is to pair scalable retrieval with causal treatment-effect modeling: a trained transformer retrieves leakage-safe treated and control neighbors from large randomized marketing experiments and estimates customer-level uplift. The core evaluation should rely on raw and normalized Qini, using the same folds, seeds, and baseline protocol as CausalPFN.

## 4. Related-work paragraph for the report

Recent tabular foundation models improve the scalability of transformer-based learning through efficient row representations, retrieval, real-data pretraining, and parameter-efficient ensembles. TabICL and TabDPT target large tabular contexts, TabM supplies an efficient neural baseline, and TabPFN-2.5 extends tabular foundation modeling to larger tables. Within causal machine learning, dynamic information sharing strengthens treatment-specific learners, targeted heterogeneous transformers address sequential treatment policies, prognosis-aware pretraining improves CATE estimation, and disentangled representations separate the differing roles of pre-treatment features. Our project builds on these directions by training a retrieval-based transformer that attends separately to similar treated and control customers in large randomized marketing experiments and predicts individual uplift.

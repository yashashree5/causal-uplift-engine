# Literature Review: Causal Uplift Engine

**Status:** first pass. Notes come from each paper's abstract and the sections listed. A deeper second pass follows as the model implementation needs it.

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
- **Problem:** transformer-based in-context learning becomes expensive when every training row must participate in attention.
- **Method:** TabICL uses column- and row-wise attention to create fixed-dimensional row representations and then applies a transformer to those representations. It is pretrained on synthetic classification tasks with datasets up to 60,000 rows and is designed to handle larger tables at inference time.
- **Results:** TabICL is competitive with TabPFNv2 and faster on the TALENT benchmark. It performs particularly well on datasets with more than 10,000 rows.
- **Limitations:** it is a predictive classification model, not a causal or uplift estimator.
- **Relevance to us:** the two-stage row representation can help address the context-size limitation of CausalPFN. M1 can use a similar compact encoder while retrieving treated and control neighbors separately.
- **Open questions:** whether row representations learned for prediction remain useful for estimating treatment heterogeneity and Qini performance.

### A7 TabDPT: Scaling Tabular Foundation Models on Real Data (Ma et al., NeurIPS 2025)

- **Link:** https://arxiv.org/abs/2410.18164 · conference version: https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html
- **Problem:** tabular foundation models need to generalize to unseen datasets, but synthetic-only pretraining and full-table attention limit their scalability.
- **Method:** TabDPT combines retrieval-based in-context learning with self-supervised column masking and pretraining on real tabular data. It supports classification and regression without task-specific fine-tuning.
- **Results:** the paper reports strong performance on OpenML-CC18 classification and OpenML-CTR23 regression. It also reports improvements as model size and the amount of real pretraining data increase.
- **Limitations:** it is not trained to estimate potential outcomes or uplift, and its retrieval context does not separate treated and untreated units.
- **Relevance to us:** TabDPT provides a useful retrieval-plus-pretraining design for the large-table problem in Model 1 and the foundation-model direction in Model 4.
- **Open questions:** whether retrieval and self-supervised pretraining on campaign data can improve uplift estimation without introducing treatment leakage.

### A8 TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (Gorishniy et al., ICLR 2025)

- **Link:** https://arxiv.org/abs/2410.24210 · conference version: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html
- **Problem:** complex tabular transformers can be expensive, while standard neural networks can overfit heterogeneous tabular data.
- **Method:** TabM uses an MLP backbone with parameter-efficient BatchEnsemble-style components. One model produces multiple related predictions that are combined as an ensemble.
- **Results:** TabM reports strong performance and a favorable accuracy-efficiency trade-off across tabular deep-learning benchmarks.
- **Limitations:** it is not a causal model and does not directly estimate separate potential outcomes.
- **Relevance to us:** TabM is a practical neural baseline for M1. It can be adapted into an S-Learner or T-Learner and compared with the retrieval transformer using the same folds and Qini evaluation.
- **Open questions:** whether parameter-efficient ensembling can improve uplift ranking when treatment groups are imbalanced, especially on Criteo.

### A9 TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (Grinsztajn et al., 2025/2026)

- **Link:** https://arxiv.org/abs/2511.08667
- **Problem:** tabular foundation models need to support larger tables and more features while keeping inference practical.
- **Method:** TabPFN-2.5 extends the TabPFN family to larger tables and wider feature spaces. It also introduces distillation into compact MLP and tree-ensemble models.
- **Results:** the report evaluates TabArena and reports strong results against tuned tree models and AutoGluon, with support for tables of approximately 50,000 rows and 2,000 features.
- **Limitations:** it is a general predictive model rather than an uplift estimator, and its reported results need to be tested on the project's marketing datasets.
- **Relevance to us:** it is a current foundation-model baseline for smaller Hillstrom and Criteo subsets. Its scaling and distillation ideas are also relevant to deployment of the final model.
- **Open questions:** whether a tabular foundation model can provide useful representations for treated/control retrieval without causal fine-tuning.

### B5 Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation (Chauhan et al., AISTATS 2024)

- **Link:** https://proceedings.mlr.press/v238/chauhan24a.html
- **Problem:** treatment-effect learners often split limited data into separate treatment groups, reducing the amount of information available to each model.
- **Method:** the paper proposes learned soft weight sharing so treatment-specific learners can exchange information dynamically while retaining treatment-specific behavior.
- **Results:** the method improves ITE estimation over fully separate or fixed-sharing treatment learners, particularly when treatment groups are small.
- **Limitations:** it is not a retrieval transformer and does not directly address very large marketing tables.
- **Relevance to us:** it provides a useful design for sharing information between the treated and control branches of M1 while preserving separate outcome heads.
- **Open questions:** whether sharing the retrieval-attention block improves Qini when the Criteo treatment assignment is highly imbalanced.

### B6 Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer (Shirakawa et al., ICML 2024)

- **Link:** https://proceedings.mlr.press/v235/shirakawa24a.html
- **Problem:** deep models for counterfactual prediction can be biased and may not provide statistical inference for repeated treatments.
- **Method:** Deep LTMLE uses a transformer with heterogeneous type embeddings and temporal-difference learning, followed by targeted minimum loss-based estimation for bias correction and confidence intervals.
- **Results:** the paper reports improved counterfactual estimation and inference for longitudinal treatment policies.
- **Limitations:** Hillstrom and the main Criteo task are single-period randomized experiments, so the complete longitudinal method is outside the current implementation scope.
- **Relevance to us:** it provides a future architecture for repeated campaigns, exposure histories, and sequential targeting in Models 2 and 3.
- **Open questions:** how treatment embeddings and targeted correction could be adapted from sequential campaigns to multiple campaign arms.

### B7 Statistical Learning for Heterogeneous Treatment Effects: Pretraining, Prognosis, and Prediction (Schuessler, Sverdrup & Tibshirani, 2025)

- **Link:** https://arxiv.org/abs/2505.00310
- **Problem:** CATE estimation is difficult when there are many covariates but limited treatment-effect signal.
- **Method:** the authors pretrain prognostic outcome structure and incorporate it into an R-learner-style residualized objective. The approach uses information about baseline outcome risk to improve treatment-effect estimation.
- **Results:** the paper reports lower estimation error and improved detection of treatment-effect heterogeneity when prognostic and effect-modifying signals are related.
- **Limitations:** the benefit depends on the relationship between baseline prognosis and treatment-effect heterogeneity.
- **Relevance to us:** M1's shared encoder can first be pretrained for factual outcome prediction and then fine-tuned for uplift, providing a direct ablation against random initialization.
- **Open questions:** whether outcome pretraining improves Qini on Hillstrom and Criteo or only improves factual BCE.

### C5 Deep Disentangled Representation Network for Treatment Effect Estimation (Meng et al., 2025)

- **Link:** https://arxiv.org/abs/2507.06650
- **Problem:** pre-treatment features can have different roles in treatment assignment, baseline outcome prediction, and treatment-effect prediction.
- **Method:** the model uses mixture-of-experts, multi-head attention, an orthogonality regularizer, and importance reweighting to learn separate representation components.
- **Results:** the paper reports improvements over treatment-effect baselines on semi-synthetic and production datasets.
- **Limitations:** the main experiments use observational data, and the learned representations are not guaranteed to correspond exactly to true causal roles.
- **Relevance to us:** its representation design can guide an M1 ablation with separate baseline-response and uplift representations, especially if the project later adds exposure or multiple campaign treatments.
- **Open questions:** whether disentangled representations improve uplift ranking on randomized marketing data after removing unnecessary observational weighting.

## 2. New papers found

| Paper | Link | Year | Why it matters |
|---|---|---|---|
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | https://arxiv.org/abs/2502.05564 | 2025 | Scalable transformer architecture for large tabular contexts |
| TabDPT: Scaling Tabular Foundation Models on Real Data | https://arxiv.org/abs/2410.18164 | 2025 | Retrieval and self-supervised pretraining for tabular foundation models |
| TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling | https://arxiv.org/abs/2410.24210 | 2025 | Efficient neural baseline for tabular prediction |

## 3. Synthesis

Recent tabular foundation models address transformer scaling through efficient row representations, retrieval, real-data pretraining, and parameter-efficient ensembles. TabICL and TabDPT are especially relevant to the large-table limitation in CausalPFN, while TabM provides a strong practical neural baseline. TabPFN-2.5 is a current foundation-model reference for smaller tables and model distillation.

Recent causal methods provide complementary ideas for the uplift model: dynamic information sharing can improve treatment-specific branches, targeted transformer estimation supports sequential treatments, prognosis-aware pretraining can improve CATE learning, and disentangled representations can separate baseline and treatment-effect signals.

The project opportunity is to combine scalable retrieval with causal treatment-effect modeling: a trained transformer retrieves leakage-safe treated and control neighbors from large randomized marketing experiments and estimates customer-level uplift. The main evaluation should use raw and normalized Qini, with the same folds, seeds, and baseline protocol as CausalPFN.

## 4. Related-work paragraph for the report

Recent tabular foundation models improve the scalability of transformer-based learning through efficient row representations, retrieval, real-data pretraining, and parameter-efficient ensembles. TabICL and TabDPT are designed for large tabular contexts, while TabM provides an efficient neural baseline and TabPFN-2.5 extends tabular foundation modeling to larger tables. In causal machine learning, dynamic information sharing improves treatment-specific learners, targeted heterogeneous transformers address sequential treatment policies, prognosis-aware pretraining improves CATE estimation, and disentangled representations separate different roles of pre-treatment features. Our project builds on these directions by training a retrieval-based transformer that attends separately to similar treated and control customers in large randomized marketing experiments and predicts individual uplift.


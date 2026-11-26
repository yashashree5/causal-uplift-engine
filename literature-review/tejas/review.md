# Literature Review: Tejas

**Role:** Model engineer (assumed; `WORK_DIVISION.md` still lists roles as TBD, so confirm) · **Linear issue:** TEA-2

**Scope:** all six core papers (A1, A5, B1, B4, C1, D1), the model engineer's role papers (B3, B5, C2, A3), and new papers from 2024 onward.

**Caveat on the new-paper search:** I ran three web searches and did not do a systematic search. I did not check the "Cited by" list for CausalPFN on Google Scholar, so the list in section 2 is not exhaustive.

---

## 1. Paper notes

### A1 CausalPFN (Balazadeh et al., 2025)

- **Link:** [arXiv 2506.07918](https://arxiv.org/abs/2506.07918), NeurIPS 2025
- **Problem:** Can one pretrained transformer estimate causal effects (CATE, CEPOs) from observational data with no per-task training?
- **Method:** A PFN trained on a large library of simulated data-generating processes that satisfy ignorability. At inference it reads the context table and each query row, and outputs a distribution over the effect. The authors' retrieval step for large tables is in App. D.2: fit a gradient-boosted regressor for weak CATE estimates, sort context and queries by that estimate, and for each query batch take a contiguous window of context rows with a similar weak-CATE range. Context length at inference is up to 50,000 rows (App. D.3).
- **Data:** Synthetic DGPs for training. Evaluated on IHDP, Lalonde, ACIC, and five marketing RCTs (Hillstrom, Lenta, X5, Megafon, Criteo) via scikit-uplift.
- **Results:** Table 2 (50k stratified subsamples, normalized Qini): Criteo **0.859**, average **0.952**. Table 7 (full tables, retrieval): Criteo **0.939**, X5 **0.746** (down from 0.922), Lenta **0.947** (down from 1.000), Mega **0.954** (down from 0.970), average **0.924**. Each table normalizes to its own best model, so the two tables are not directly comparable.
- **Limitations:** Assumes strong ignorability, which cannot be tested. The authors say the full-table drop is probably caused by the retrieval heuristic ("We conjecture…", Table 7 caption), and they leave long contexts as future work.
- **Relevance to us:** This is our baseline and the source of the "large-table drop" that Model 1 is meant to fix. Model 1b replaces the weak-CATE window with learned, arm-specific neighbours.
- **Open questions:**
  - Our plan (M3) expects a "Criteo drop" in Table 7, but the paper's own Criteo numbers go **up** from Table 2 to Table 7 (0.859 → 0.939). The drops are on X5, Lenta, and Mega. See the risk section.
  - Does a similarity-based retrieval beat the weak-CATE window the authors use?

### A5 LoCalPFN (Thomas et al., 2024)

- **Link:** [arXiv 2406.05207](https://arxiv.org/abs/2406.05207), NeurIPS 2024
- **Problem:** TabPFN's in-context learning fails to scale with dataset size and complexity because attention memory grows quadratically with context length.
- **Method:** For each query, take its k nearest neighbours as the context (k = min(10·√n, a fixed maximum)). Then fine-tune TabPFN end-to-end on these local contexts. To keep fine-tuning cheap, many queries share one context (Sec. 2.4). Raw standardized features work about as well as learned embeddings for retrieval (Fig. 5, Table 9).
- **Data:** 95 classification datasets from TabZilla/OpenML.
- **Results:** Table 1, AUC, all 95 datasets (IQM): TabPFN **0.917** → LoCalPFN **0.958**. Medium/large datasets: TabPFN 0.927 → LoCalPFN **0.968**, versus XGBoost 0.965.
- **Limitations:** Only TabPFN was tested as the base model. Only classification was evaluated. Retrieval uses a single outcome, with no treatment arms.
- **Relevance to us:** Closest published idea to Model 1b: local retrieval plus fine-tuning. The main difference is that LoCalPFN retrieves neighbours from one pool, while M1b retrieves treated and control neighbours separately and reads their outcomes with attention. The paper also shows that fine-tuning with retrieval beats either one alone (Fig. 5, left), so M1b needs the same ablation.
- **Open questions:** Does raw-feature kNN retrieval, which LoCalPFN found competitive, beat the weak-CATE window in CausalPFN on Criteo? Does the gain carry over from classification to a binary outcome with treatment effects?

### B1 Meta-learners (Künzel et al., 2019)

- **Link:** [arXiv 1706.03461](https://arxiv.org/abs/1706.03461), PNAS 2019
- **Problem:** How do we estimate CATE with any supervised learner, especially when one arm is much larger than the other?
- **Method:** The S-learner fits one model with the treatment as a feature. The T-learner fits one model per arm. The X-learner first fits T-learner outcome models, then imputes individual effects (treated: Y1 − μ0(X); controls: μ1(X) − Y0), fits a model to each imputed set, and combines the two with a weight g(x) that is usually the propensity score.
- **Data:** Simulation studies with random forests and BART as base learners. Details are in the supporting information.
- **Results:** The X-learner is designed to do well with unbalanced designs and smooth CATEs. The S-learner is strong when the CATE is near zero. Exact numbers are in the paper's simulation tables.
- **Limitations:** The X-learner's advantage comes from assumptions about the CATE's structure. The paper gives no uplift-specific evaluation.
- **Relevance to us:** These are our comparison methods. In CausalPFN's Table 2, the X-learner is the best model on Criteo 50k (1.000), so it is the bar Model 1 must clear. Criteo's 85% treated share is the unbalanced case the X-learner was built for.
- **Open questions:** Does the X-learner's edge on the 50k Criteo subsample persist with our own folds and our own Qini code?

### B4 TARNet (Shalit et al., 2017)

- **Link:** [arXiv 1606.03976](https://arxiv.org/abs/1606.03976), ICML 2017
- **Problem:** How can we bound the error of individual treatment effect (ITE) estimates when we observe only one potential outcome per unit?
- **Method:** TARNet learns one shared representation Φ(x) and two outcome heads h(Φ, 0) and h(Φ, 1). Each training example updates only the head for its observed arm. CFR (the paper's main method) adds an IPM penalty (Wasserstein or MMD) that makes the treated and control representations similar, with weights w_i = t/(2u) + (1−t)/(2(1−u)) where u = P(T=1).
- **Data:** IHDP (semi-synthetic, 747 units) and Jobs.
- **Results:** Table 1, out-of-sample √PEHE on IHDP: TARNet **0.95**, CFR MMD **0.78**. Lower is better. The table lists the standard error as ±.0, which is less precise than the ±0.02 in Jim's review; check the table before quoting an SE.
- **Limitations:** PEHE needs the true effect, which exists only in semi-synthetic data. The paper does not evaluate on randomized marketing data.
- **Relevance to us:** M1a is TARNet with a transformer encoder. The arm-weighting in our plan is the same idea as CFR's w_i, and it addresses the 85/15 imbalance in Criteo.
- **Open questions:** Does an IPM balancing penalty help M1a on Criteo's imbalanced arms, and does it help ranking as well as PEHE?

### C1 FT-Transformer (Gorishniy et al., 2021)

- **Link:** [arXiv 2106.11959](https://arxiv.org/abs/2106.11959), NeurIPS 2021
- **Problem:** Are deep models for tabular data beaten by a simple, well-tuned baseline?
- **Method:** The Feature Tokenizer maps each feature to its own embedding (numerical: scaled vector plus bias; categorical: lookup). A [CLS] token and a standard transformer encoder process the tokens, and the [CLS] output feeds the prediction. The paper also proposes a ResNet-like baseline.
- **Data:** A set of public tabular benchmarks for supervised classification and regression.
- **Results:** FT-Transformer is among the strongest deep models on the benchmark, and the ResNet-like baseline is a strong default. Table numbers are in the paper.
- **Limitations:** Training costs more than simple models, and the cost grows with the number of features. The benchmarks are supervised and in-distribution, with no uplift task.
- **Relevance to us:** This is the Model 1 encoder (`models/m1/encoder.py`). Criteo has only 12 features, so per-feature tokens are cheap. Hillstrom has 8 fields.
- **Open questions:** Does the feature tokenizer help over an MLP encoder for two-head uplift? Test it in the M1a ablation.

### D1 Uplift review (Gutierrez & Gérardy, 2017)

- **Link:** [PMLR v67](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf), PAPIs 2016 (PMLR 2017)
- **Problem:** What is the right way to evaluate uplift models, and which methods are used in practice?
- **Method:** A review of the uplift literature. It defines the uplift curve (Eq. 16, bins of predicted uplift) and the Qini curve from Radcliffe (2007). The Qini coefficient is the area under the Qini curve. The review notes that these curves are valid only under randomization or a balanced design.
- **Data:** No new experiment. It is a literature review.
- **Results:** Qualitative comparison of methods.
- **Limitations:** It predates most of the newer uplift methods and does not define normalized Qini.
- **Relevance to us:** The Qini curve is our main metric. We have to match the authors' exact definition and normalization (the plan's M3 check), and the CausalPFN tables normalize per table, so raw scores need to be reported too.
- **Open questions:** How often do Qini and PUC (D5) rank the same models in different orders on the same predictions?

### B3 R-learner (Nie & Wager, 2021)

- **Link:** [arXiv 1712.04912](https://arxiv.org/abs/1712.04912), Biometrika 2021
- **Problem:** Can we estimate heterogeneous effects with any ML learner, while controlling for confounding through an orthogonal loss?
- **Method:** Residualize the outcome Y − m̂(X) and the treatment W − ê(X) with cross-fitting. Then minimize the R-loss Σ[(Y − m̂) − τ(X)(W − ê)]² over τ. The loss is the same for any learner, so you can use regression trees, boosting, or neural nets.
- **Data:** Simulation studies, with penalized regression, kernel ridge, and boosting as base learners.
- **Results:** Shows that the R-learner has promising performance and error bounds for kernel regression.
- **Limitations:** Requires cross-fitting. The guarantees rely on nuisance models converging at a reasonable rate.
- **Relevance to us:** The R-loss is a reasonable alternative to the arm-weighted BCE in our M1a loss, because it targets the effect directly.
- **Open questions:** Does an R-loss head beat two separate outcome heads on Criteo?

### B5 DragonNet (Shi et al., 2019)

- **Link:** [arXiv 1906.02120](https://arxiv.org/abs/1906.02120), NeurIPS 2019
- **Problem:** Can a neural network use the propensity score during training to estimate treatment effects?
- **Method:** One shared representation trunk feeds three heads: a propensity head ĝ(x) and two outcome heads Q̂(0, x) and Q̂(1, x). The paper argues the propensity score is sufficient for causal estimation (Theorem 2.1). Targeted regularization adds a term that perturbs the outcome heads, inspired by TMLE.
- **Data:** Semi-synthetic benchmarks, including IHDP.
- **Results:** Table 1 reports mean absolute error on IHDP. DragonNet with targeted regularization is the best neural method there.
- **Limitations:** Evaluation is semi-synthetic. The paper does not test ranking metrics on randomized campaigns.
- **Relevance to us:** Our M1a has the same two-head structure as DragonNet, but without the propensity head. The propensity head is a cheap auxiliary task, and with Criteo at 85% treated it could help.
- **Open questions:** Does adding a propensity head to M1a improve Qini on Criteo? The result would tell us whether the auxiliary task helps.

### C2 TabR (Gorishniy et al., 2024)

- **Link:** [arXiv 2307.14338](https://arxiv.org/abs/2307.14338), ICLR 2024
- **Problem:** Can a neural network for tabular data use nearest neighbours from the training set and still beat tree ensembles?
- **Method:** For each target object, retrieve the m nearest candidates from the training set using a learned similarity. Encode the candidates with the same encoder, then use attention over the retrieved (key, label) pairs to adjust the target's representation. The retrieved contexts are refreshed as the encoder changes (Sec. 3, Fig. 2 and 4). The target object can optionally be added to its own context set.
- **Data:** Public tabular benchmarks, up to several million objects.
- **Results:** The paper reports that TabR has the best average performance among deep models on its benchmark, and it sets new state-of-the-art results on several datasets.
- **Limitations:** Predicts an outcome, not a treatment effect. Training recomputes retrieval over all candidates, which the authors note is slow on large data.
- **Relevance to us:** M1b's learned retrieval (rebuild neighbour lists from the encoder's own vectors between epochs) follows TabR. The leakage rule in our plan should match TabR's option of excluding the target from its own context.
- **Open questions:** Does attention over retrieved neighbours help when the retrieved set is split by arm? That is the M1b-specific question TabR does not address.

### A3 Do-PFN (Robertson et al., 2025)

- **Link:** [arXiv 2506.06039](https://arxiv.org/abs/2506.06039), 2025
- **Problem:** Can a PFN estimate interventional distributions from observational data alone?
- **Method:** Pretrain a PFN on synthetic structural causal models. For each SCM, sample an observational dataset and interventional queries. At inference, the model gets the observational table and a query (t, x) and predicts the conditional interventional distribution.
- **Data:** Synthetic SCMs for pretraining. Evaluated on synthetic and semi-synthetic benchmarks (RealCause, Amazon Sales, Law School). Its CATE sample-size experiment varies the context size up to about 2,000 rows.
- **Results:** Competitive with DoWhy and other baselines on the paper's benchmarks. Robust to noise and non-linearity not seen in pretraining (Fig. 4).
- **Limitations:** Section 5 of the paper lists four limitations. The sample-size experiment stops at about 2,000 rows, so the paper gives no evidence on 50k-row tables.
- **Relevance to us:** Competitor to CausalPFN. The key question from the plan ("does it handle large tables?") is not answered by this paper.
- **Open questions:** Does Do-PFN's accuracy hold when the context grows to 50k rows, as in our CausalPFN comparison?

---

## 2. New papers found

Each entry was checked against the search result or the paper's abstract. These are added to `READING_LIST.md` under section F.

| Paper | Link | Year | Why it matters |
|---|---|---|---|
| CausalFM: Foundation Models for Causal Inference via Prior-Data Fitted Networks | [arXiv 2506.10914](https://arxiv.org/abs/2506.10914) | 2025 (ICLR 2026) | Trains PFNs for back-door (CATE), IV, and front-door settings from SCM priors. A second PFN competitor to CausalPFN that we need in the comparison. |
| TabDPT: Scaling Tabular Foundation Models on Real Data | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf) | 2025 | Retrieves the top-K feature-space neighbours for each query during both training and inference (Sec. 3.3–3.4). The closest recent precedent for M1b's retrieval, but for prediction, not causal effects. |
| Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data | [arXiv 2509.00326](https://arxiv.org/abs/2509.00326) | 2025 | Exact attention over long contexts with tiling, with no KNN compression. Its existence weakens the case that large tables need retrieval, so we should compare against it. |
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | [arXiv 2502.05564](https://arxiv.org/abs/2502.05564) | 2025 | Handles up to 500K training rows with a column-then-row attention design. Another alternative to context retrieval for large tables. |
| Scalable In-Context Learning on Tabular Data via Retrieval-Augmented LLMs (TabRAG) | [arXiv 2502.03147](https://arxiv.org/abs/2502.03147) | 2025 | Non-parametric retrieval module for tabular data with power-law error scaling. Relevant to the design of M1b's retrieval function. |
| CauGramer: Causal Graph Transformer for Treatment Effect Estimation under Unknown Interference | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/f7397355f68fa42e5d4739c2eb9b4cf6-Paper-Conference.pdf) | 2025 | Uses cross-attention over neighbours' treatments and features for treatment-effect estimation. The neighbours are graph peers, not feature-space neighbours, so it is related but not a direct precedent. |
| Data-Driven Uplift Modeling (bipartite matching) | [IEEE Access, doi:10.1109/access.2025.3557468](https://doi.org/10.1109/access.2025.3557468) | 2025 | Matches treated and control units on a weighted bipartite graph before fitting an ordinary model. A matching baseline that is cheap to add to the comparison. |

---

## 3. Synthesis

**What is already solved?**

- Two-head models for treatment effects are established: TARNet (B4), DragonNet (B5), and CFR. Meta-learners (B1) and the R-loss (B3) are strong reference methods.
- Retrieval helps tabular models: LoCalPFN (A5) on classification, TabR (C2) for prediction, and TabDPT and TabRAG on large tables.
- Long-context tabular ICL is being addressed by exact attention (Chunked TabPFN) and by linear-cost architectures (TabICL, TabFlex).

**What is still open?**

- CausalPFN's large-table drop is only a conjecture in the paper. The paper attributes it to the retrieval heuristic, and its own numbers show the drop mainly on X5, Lenta, and Mega, not on Criteo (see risks).
- No paper I found evaluates neighbour retrieval for uplift on randomized marketing data with arm-specific neighbours. The paper's own sample-size experiment (Do-PFN) stops at 2,000 rows.
- CHAUN (Zhang et al., 2026) is a preprint whose PDF has template artefacts (e.g. a "Received 5 June 2009" line and the Woodstock conference template). Its reported gains over DragonNet are small, so treat it as weak evidence until it is published.

**Does anything already do what our Model 1 does?**

**Answer: no paper I found does all of M1 at once.** M1 combines these parts:
1. a transformer encoder over features,
2. two outcome heads, one per arm (M1a),
3. attention over the **retrieved treated neighbours and retrieved control neighbours**, each with their observed outcomes (M1b),
4. training on the observed arm with a weighted loss, evaluated on Qini.

The closest papers each cover part of this:

| Component | Closest existing work | What it does not do |
|---|---|---|
| Two outcome heads | TARNet (B4), DragonNet (B5), CHAUN | No retrieval of neighbours |
| Cross-head attention | CHAUN | Attends across heads within one customer, not across retrieved customers |
| Retrieval + attention for tabular prediction | TabR (C2), TabDPT, LoCalPFN (A5) | Predicts a single outcome; no treatment arms |
| Retrieval for causal effects | CausalPFN (A1), via weak-CATE windows | Not neighbour-based; no learned retrieval |
| Matched treated and control pairs | Data-driven uplift (2025) | Matches first, then fits a standard model |

**Searches I ran (to answer this question):**

1. "retrieval augmented uplift modeling transformer nearest neighbors treated control 2025 arXiv"
2. "in-context learning causal effect estimation prior-fitted network foundation model 2025 2026 arXiv CausalFM"
3. "tabular in-context learning large datasets context retrieval scaling TabPFN 2025 arXiv"
4. Looked at the "Cited by" of the core papers in the text only. I did not check Google Scholar or Semantic Scholar.

Searches 1 and 2 are the ones that matter most, and I would repeat them on Google Scholar before the report.

**Risks this review uncovered for our project:**

1. **The "Criteo drop" in M3 is not in the paper's own numbers.** Table 2 (50k) gives Criteo 0.859, and Table 7 (full table) gives 0.939. The large-table drop is on X5 (0.922 → 0.746), Lenta (1.000 → 0.947), and Mega (0.970 → 0.954). Before we write "reproduce the Criteo drop" as a milestone, we should check our own numbers. If we reproduce 0.859 and 0.939, the drop story does not hold on Criteo, and the report needs a different claim. This should go to the team before M3.
2. **Table 2 and Table 7 are normalized differently.** Scores normalize to the best model in each table, so cross-table comparisons are indicative only. Report raw Qini alongside the normalized score.
3. **Long-context baselines are strong.** Chunked TabPFN and TabICL handle large tables without retrieval. A fair M1 comparison needs at least one of them as a non-causal reference, not only CausalPFN.
4. **PFN-based causal models are moving fast.** CausalFM (ICLR 2026) and Do-PFN are direct competitors. Both should be in the final comparison or in the related-work section with a reason for leaving them out.
5. **Leakage in retrieval.** TabR's design shows that the retrieval set and the target must be kept separate. Our leakage test should check the case where the target is in its own context.
6. **The retrieval ablation must be separate from the fine-tuning ablation.** LoCalPFN shows that joint retrieval and fine-tuning beats either alone (Fig. 5, left). M1b's ablations should report both.

---

## 4. Related-work paragraph for the report

Causal effect estimation with flexible models has moved from separate learners to shared-representation networks. TARNet (B4) learns one representation with two outcome heads and adds a balancing penalty, and DragonNet (B5) adds a propensity head. Meta-learners (B1) and the R-learner (B3) provide strong non-neural baselines. More recent work borrows from tabular in-context learning. CausalPFN (A1) trains a PFN on simulated causal data and handles large tables by ordering rows on a gradient-boosted CATE estimate, which the authors conjecture is behind reduced accuracy on their full-table runs. LoCalPFN (A5) and TabR (C2) show that retrieving nearest neighbours and attending over them improves tabular prediction, while Do-PFN (A3) and CausalFM (2506.10914) extend the PFN approach to interventional queries. What is missing from this literature is a model that combines a two-head treatment architecture with attention over neighbours retrieved separately from the treated and control groups. We address this gap with Model 1, which we compare against CausalPFN and against classical learners on Hillstrom and Criteo.


# Literature Review: Jim

**Role:** To confirm · **Linear issue:** TEA-4

## 1. Paper notes

### B4 TarNet (Shalit et al., 2017)

- **Source** — Shalit, Johansson, and Sontag, *Estimating Individual Treatment Effect: Generalization Bounds and Algorithms*, ICML 2017. The authors spell the model **TARNet**, short for Treatment-Agnostic Representation Network. [Paper](https://proceedings.mlr.press/v70/shalit17a.html)
- **Problem** — How can a model estimate the effect of treatment for someone when we observe their outcome under only one assignment?
- **Method** — TarNet learns one shared representation of pre-treatment features. Two heads estimate outcomes under treatment and control. Each training example supervises its assigned head, and the predicted difference between the heads gives the estimated treatment effect. [Figure 1](https://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf#page=2)
- **Data** — The authors evaluated TarNet on the semi-synthetic IHDP benchmark and the Jobs study.
- **Results** — On IHDP, TarNet’s held-out root PEHE (treatment-effect estimation error) was **0.95 ± 0.02**. [Table 1](https://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf#page=8)
- **Limitations** — The IHDP result measures error against simulated effects. Whether TarNet selects better recipients at our 20% campaign budget requires evaluation on the randomized visit data.
- **Relevance to us** — TarNet supplies an established two-head baseline for estimating visit uplift on randomized campaigns.
- **Open question** — How well does TarNet’s uplift ranking identify the customers who benefit most at a fixed contact budget?

### E3 CURL (Xu et al., 2026)

- **Source** — *Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation*. [arXiv preprint](https://arxiv.org/html/2607.26599v1)
- **Problem** — Can a frozen LLM help estimate heterogeneous treatment effects when the structure in pre-treatment covariates is hard for a learner to extract from limited data?
- **Method** — CURL uses repeated MC-dropout predictions to prioritize examples where the host learner is uncertain about treatment assignment or treatment effects. A frozen LLM reads the pre-treatment profile in two separate prompts and supplies hidden-state embeddings for assignment and for effect heterogeneity. The adapter routes the embeddings through different paths and continues training the host over successive rounds. [§4 and Appendix C](https://arxiv.org/html/2607.26599v1)
- **Data** — Semi-synthetic IHDP and Adult, plus Jobs and the randomized Hillstrom men’s email versus no-email comparison. Hillstrom has 64,000 customers across the three original arms. CURL uses an 80% training partition with validation drawn from it. [Appendix A](https://arxiv.org/html/2607.26599v1#A1)
- **Results** — On Hillstrom, adding CURL to the T-Learner moved AUUC from **0.106 ± 0.035** to **0.136 ± 0.038** and AUQC from **0.020 ± 0.002** to **0.023 ± 0.003** (mean ± SD over five runs). [Table 1](https://arxiv.org/html/2607.26599v1#S5.T1)
- **Limitations** — The method depends on interpretable pre-treatment feature descriptions and requires LLM inference. Criteo’s anonymized features make transfer uncertain. Hillstrom provides ranking metrics for the randomized campaign, with individual effects unobserved.
- **Relevance to us** — Informs the Hillstrom semantic-augmentation pilot and the Semester 2 LLM work (M4).
- **Open question** — With model capacity and training schedule held fixed, do customer-specific embeddings improve held-out ranking over zeroed or shuffled embeddings?

### D5 PTONet and PUC (Zhu et al., 2025)

- **Source** — *Rethinking Causal Ranking: A Balanced Perspective on Uplift Model Evaluation*. [ICML proceedings](https://proceedings.mlr.press/v267/zhu25s.html)
- **Problem** — Can uplift evaluation and training better account for information in both positive and negative binary outcomes?
- **Method** — The Principled Uplift Curve (PUC) ranks customers by predicted effect and accumulates contributions from all four observed treatment–outcome combinations. Treated responders and control nonresponders contribute positively, while treated nonresponders and control responders contribute negatively. The authors derive the Principled Uplift Loss and combine it with a DragonNet-derived architecture in PTONet. [Equations 9–13 and Figure 4](https://proceedings.mlr.press/v267/zhu25s.html)
- **Data** — A 10,000-row simulation, Criteo visits, and Lazada. The appendix lists Criteo splits of about 12.6M train and 1.40M test and Lazada splits of about 4.17M train and 0.91M test. [Table 10](https://proceedings.mlr.press/v267/zhu25s.html)
- **Results** — On Criteo visits, normalized PUC area was **0.191 ± 0.01** for PTONet and **0.186 ± 0.01** for EUEN (mean ± SD). [Table 5](https://proceedings.mlr.press/v267/zhu25s.html)
- **Limitations** — The analysis assumes binary treatment and outcome and relies on unconfoundedness. The released PUC implementation can depend on row order when uplift scores tie. [Reference implementation](https://github.com/euzmin/PUC/blob/6c9f60839fc8707a8d5646de8944f148bcf46c65/utils.py)
- **Relevance to us** — Jim’s Hillstrom reproduction already calculates PUC alongside CausalPFN’s official Qini on the same pooled out-of-fold predictions. PTONet also supplies a candidate comparator for the binary outcome task.
- **Open question** — How often do PUC and official Qini disagree on the ordering of models on the same Hillstrom predictions?

### B6 CHAUN (Zhang et al., 2026)

- **Source** — *Cross-Head Attention Uplift Network with Inverse Propensity Score under Unobserved Confounding*. [arXiv preprint](https://arxiv.org/html/2606.27114v1)
- **Problem** — Can sharing information across potential-outcome heads improve uplift prediction? How sensitive is estimation when measured covariates omit confounders?
- **Method** — CHAUN builds two representations of each customer from shared features. Each outcome head attends to both representations before making its prediction, and the difference between the predictions is the estimated uplift. A propensity head supplies inverse weights, and the proposed RA-IPS training objective optimizes against adverse weights within a specified sensitivity bound. [§4](https://arxiv.org/html/2606.27114v1)
- **Data** — Criteo (11,183,673 train and 112,788 test), Lazada (926,669 train and 181,669 test), and proprietary production data (21,350,813 train and 11,963,551 test). [Table 1](https://arxiv.org/html/2606.27114v1)
- **Results** — Criteo mean test QINI was **0.1955** for CHAUN and **0.1898** for DragonNet. The ablation without cross-head attention scored **0.1915**. [Tables 3 and 5](https://arxiv.org/html/2606.27114v1)
- **Limitations** — Criteo uses actual ad exposure, visit outcomes, and propensity-matched test data, which limits comparison with evaluation by randomized assignment. RA-IPS results depend on the assumed bound on hidden confounding.
- **Relevance to us** — CHAUN tests attention across one customer’s two outcome representations. M1b tests attention over other customers’ outcomes in separately retrieved treated and control neighborhoods. The two mechanisms motivate different ablations.
- **Open question** — Does cross-head attention improve ranking over a matched two-head baseline under Hillstrom’s randomized assignment?

### D6 HS sampling (Bokelmann and Lessmann, 2025)

- **Source** — *Heteroscedasticity-aware stratified sampling to improve uplift modeling*. [European Journal of Operational Research](https://doi.org/10.1016/j.ejor.2025.02.030)
- **Problem** — Can oversampling customers with noisier outcomes improve uplift modeling and precision for a fixed experimental sample size?
- **Method** — The design estimates baseline outcome risk using historical untreated records. For rare binary outcomes, higher baseline risk can indicate higher outcome variance. The design oversamples the high-variance stratum before treatment randomization. The authors use weights to recover population proportions when estimating the average treatment effect or evaluating a ranking. [§4](https://doi.org/10.1016/j.ejor.2025.02.030)
- **Data** — Three simulated settings plus Criteo conversion (13,979,592 source rows) and Lenta (687,029 source rows). [Published article](https://doi.org/10.1016/j.ejor.2025.02.030)
- **Results** — On Criteo conversion, the causal forest achieved an AUQ of **0.00094**, a **24.46% relative increase** over simple random sampling at **20,000 training rows** per condition. [Published version, Table 2](https://doi.org/10.1016/j.ejor.2025.02.030)
- **Limitations** — The design requires historical untreated outcomes and useful separation of the strata by outcome variance. Population evaluation depends on the sampling weights. The Criteo conversion result uses a different endpoint from our visit task.
- **Relevance to us** — Provides a data-design comparison for training at fixed sample sizes. M1b’s retrieval occurs after the training set has been sampled.
- **Open question** — Does HS sampling improve ranking at the same training budget on the Criteo visit endpoint when both models use an identical held-out test set?

### E4 GATE (Huynh, Piskorz et al., 2025)

- **Source** — *Improving Treatment Effect Estimation with LLM-Based Data Augmentation*. ICML workshop on Foundation Models for Structured Data, July 18, 2025. [Paper](https://openreview.net/pdf?id=2Q3gFNbpAr) · [Workshop](https://icml-structured-fm-workshop.github.io/2025/)
- **Problem** — Can synthetic outcomes help an uplift learner when few comparable customers appear in both treatment arms?
- **Method** — GPT-3.5 Turbo reads a customer profile, the alternative treatment, and examples from the training data. GATE averages ten generated outcomes per profile and uses their variance to select synthetic labels. It adds the selected opposite-arm examples to the training set, then fits a conventional uplift learner. [Sections 2–3 and Appendix D.4](https://openreview.net/pdf?id=2Q3gFNbpAr)
- **Data** — The authors evaluate Lalonde, STAR, and Hillstrom. For Hillstrom they construct a biased subset of 1,070 customers with eight features from the randomized men’s-email and no-email groups. The outcome is a website visit within two weeks. [Appendix D.3](https://openreview.net/pdf?id=2Q3gFNbpAr#page=17)
- **Results** — On Hillstrom, the T-learner’s root PEHE (treatment-effect estimation error) fell from **0.40 ± 0.01** to **0.24 ± 0.01** with GATE (mean ± SD across three seeds). The reference effects came from two random forests fitted to separate randomized data. [Table 1](https://openreview.net/pdf?id=2Q3gFNbpAr#page=4)
- **Limitations** — The induced imbalance differs from the original randomized campaign. Low dispersion in the generated labels can occur even when the model is wrong. Imputing each arm’s mean outcome was competitive on Hillstrom. [Section 4.2](https://openreview.net/pdf?id=2Q3gFNbpAr#page=4)
- **Relevance to us** — Offers a second use of an LLM for the Hillstrom pilot. CURL (E3) supplies customer features, while GATE supplies training labels. Either model can be evaluated on observed visits under the same email budget (D7).
- **Open question** — At a 20% email budget, does a T-learner trained with Qwen3 8B labels yield more additional visitors than the same learner trained with observed data alone or with group-average labels for the same profiles?

### D7 Fixed-budget policy evaluation (Imai and Li, 2023)

- **Source** — *Experimental Evaluation of Individualized Treatment Rules*. Published online in 2021 and in the 2023 issue of the *Journal of the American Statistical Association*. [Journal article](https://doi.org/10.1080/01621459.2021.1923511)
- **Problem** — When only a fixed proportion of a population can receive treatment, how much does choosing recipients with a model improve the average outcome?
- **Method** — Equation 3 defines the population average prescriptive effect by comparing a targeting rule with random selection at the same budget. Equation 5 compares two targeting rules at a common budget. Both contrasts average over the eligible population. [Section 2.3](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=4)
- **Data** — The Tennessee Project STAR experiment randomized more than 7,000 pupils across three class settings in 79 schools. The paper compares small and regular classes and evaluates rules for allocating the limited small-class places. [Section 6.1](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=11)
- **Results** — At a 20% budget, the cross-validated causal-forest rule improved the population-average writing score by **2.37 points** relative to the LASSO rule (95% CI **0.76 to 3.98**). [Table 4](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=12)
- **Limitations** — The empirical result concerns education. A Hillstrom visit gain and its uncertainty must be estimated on campaign data, accounting for how the targeting rule was fitted and evaluated.
- **Relevance to us** — With a binary visit outcome, 1,000 times the equal-budget policy-value difference estimates extra visitors per 1,000 eligible customers. At 20%, both policies email 200 of every 1,000. Equation 3 supports the comparison with random targeting. Equation 5 supports the comparison between an LLM model and its matched control. [Equations 3 and 5](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=4)
- **Open question** — On held-out Hillstrom customers, how large is the paired difference in extra visitors between the LLM model and its matched control, and how precise is that estimate?

## 2. New papers found

Papers not on the reading list. Add them to `READING_LIST.md` in the same PR.

| Paper | Link | Year | Why it matters |
|---|---|---|---|
| E3 CURL | [Xu et al.](https://arxiv.org/abs/2607.26599) | 2026 | Tests uncertainty-guided LLM feature augmentation with an existing uplift learner on Hillstrom |
| D5 PTONet and PUC | [Zhu et al.](https://proceedings.mlr.press/v267/zhu25s.html) | 2025 | Provides a supplementary ranking metric and a corresponding training loss |
| B6 CHAUN | [Zhang et al.](https://arxiv.org/abs/2606.27114) | 2026 | Provides a neural comparison for attention between treatment and control outcome heads |
| D6 HS sampling | [Bokelmann and Lessmann](https://doi.org/10.1016/j.ejor.2025.02.030) | 2025 | Studies whether a fixed training budget is better spent on higher-variance customers |
| E4 GATE | [Huynh, Piskorz et al.](https://openreview.net/pdf?id=2Q3gFNbpAr) | 2025 | Tests LLM-generated counterfactual training labels on a biased Hillstrom sample |
| D7 Fixed-budget policy evaluation | [Imai and Li](https://imai.fas.harvard.edu/research/files/indtreat.pdf) | 2023 | Defines the equal-budget policy comparison underlying extra visitors per 1,000 eligible customers |

## 3. Synthesis

**What is already solved?**

TarNet [B4] estimates how much treatment changes an outcome by predicting it twice for each person, once under treatment and once under control. On Hillstrom, CURL [E3] added frozen LLM features to a T-learner and raised its mean area under the uplift curve from **0.106 to 0.136** across five runs. GATE [E4] also used an LLM, this time to generate training labels. PTONet/PUC [D5] evaluates the full customer ranking, while Imai and Li [D7] show how to compare targeting decisions at a fixed budget. [TarNet](https://proceedings.mlr.press/v70/shalit17a.html) · [CURL, Table 1](https://arxiv.org/html/2607.26599v1#S5.T1) · [GATE](https://openreview.net/pdf?id=2Q3gFNbpAr) · [PUC](https://proceedings.mlr.press/v267/zhu25s.html) · [Imai and Li, §2.3](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=4)

**What is still open?**

If we can target only 200 of every 1,000 customers, would an LLM help TarNet choose people whose decision to visit would change? I’d try **Gemma 4 12B** and **Qwen3 8B** separately. When TarNet’s uplift estimate varies across repeated predictions, the frozen LLM would encode the customer’s pre-campaign profile and TarNet would score the customer again. Hillstrom has recognizable fields such as purchase history. Criteo Uplift lets us test whether the idea still works when the fields are anonymous.

**Does anything already do what our Model 1 does?**

**M1a is TarNet. M1b adds retrieval.** CHAUN [B6] shares information between two representations of the same customer. In M1b, each TarNet head would use visits from similar training customers in its own treatment arm. Comparing M1a and M1b tells us what those neighbors add. [TarNet](https://proceedings.mlr.press/v70/shalit17a.html) · [CHAUN, §4.1.3](https://arxiv.org/html/2606.27114v1#S4.SS1.SSS3)

I’d run **all four models on both datasets** and score them on the same held-out customers within each campaign.

| Project model | Hillstrom | Criteo Uplift |
|---|---|---|
| M1 · TarNet and TarNet + retrieval | Compare two-week visits for each email arm versus no email. | Compare visits under randomized ad eligibility versus control. |
| M2 · Multi-campaign transformer | Train on all three arms and evaluate each email against no email. | Use the binary version with ad eligibility and control. |
| M3 · Exposure-aware transformer | Use email assignment. Hillstrom doesn’t record opens. | Use actual exposure as an extra training target and evaluate the effect of ad eligibility on visits. |
| M4 · Campaign LLM | Fine-tune on email targeting using named customer fields. | Fine-tune on ad targeting using the supplied anonymous fields. |

**Risks this review uncovered for our project:**

A different recipient list can still produce the same number of visits. Alongside Qini, I’d report **extra visitors per 1,000 eligible customers** over random targeting at the same 20% budget. To measure the LLM’s contribution, I’d compare its embeddings with an independently fitted zero-vector control and report a paired 95% interval. Shuffling embeddings across customers would check whether the correct profile matters. HS sampling [D6] shows why the training sample matters, so I’d match training rows and budgets across fits. Retrieval would use only other customers from the training fold. [HS sampling](https://doi.org/10.1016/j.ejor.2025.02.030) · [Imai and Li, equations 3 and 5](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=4)

## 4. Related-work paragraph for the report

I propose testing whether **Gemma 4 12B or Qwen3 8B can help TarNet choose whom to target on Hillstrom and Criteo Uplift**. TarNet [B4] estimates how much a campaign changes a customer’s chance of visiting. CURL [E3] improved Hillstrom rankings using LLM features, and GATE [E4] used an LLM to generate training outcomes. My experiment builds on CURL. When TarNet’s estimate is uncertain, a frozen LLM would encode the customer’s profile using information available before the campaign. The two datasets let me test that idea with both named purchase-history fields and anonymized numbers. [TarNet](https://proceedings.mlr.press/v70/shalit17a.html) · [CURL, Table 1](https://arxiv.org/html/2607.26599v1#S5.T1) · [GATE](https://openreview.net/pdf?id=2Q3gFNbpAr)

I’d evaluate **all four models on both datasets**. Within M1, I’d compare TarNet with TarNet + retrieval to see whether similar customers’ visits improve targeting. Then I’d hold retrieval fixed and compare each LLM’s embeddings with zero vectors. Following Imai and Li [D7], each rule would select 200 customers per 1,000. The main result would be how many more people we estimate would visit when TarNet uses the customer’s own embedding to choose recipients. [Imai and Li, §2.3](https://imai.fas.harvard.edu/research/files/indtreat.pdf#page=4)

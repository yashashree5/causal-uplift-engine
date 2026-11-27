---
inclusion: always
---

# Product context: causal uplift targeting on X5 RetailHero

DATA 298A/298B capstone. X5, a grocery retailer, pays for every promotional SMS. The system estimates, per customer, the **uplift** (P(buy with SMS) minus P(buy without SMS)) and messages the customers with the highest uplift up to a fixed budget.

**Research claim.** Causal foundation models such as CausalPFN (NeurIPS 2025) read only flat feature tables. We test whether raw purchase histories, read by transformer and LLM models, improve uplift targeting. Semester 2 adds four proposed models (M1 to M4) and a RAG assistant. All of them read the tables this semester's pipeline produces.

**Dataset (the only one).** X5 RetailHero, one randomized SMS experiment from winter 2019. Source: Hugging Face `dllllb/retailhero-uplift` (resolves to `pytorch-lifestream/retailhero-uplift`). Tables: `purchases` (45.8M rows, one per product per transaction), `clients` (400k), `uplift_train` (200k labeled: `treatment_flg`, `target`), `uplift_test` (200k, no labels), `products` (43k). The average effect is about 3.3 percentage points.

**Pipeline in one line.** Hugging Face -> raw (CSV) -> bronze (Parquet) -> silver (clean transactions) -> gold (features, sequences, summaries, labels, folds) on Cloud Storage, run by Airflow. In parallel a replay producer streams purchases through Pub/Sub to a consumer that updates the same features live.

**"Real-time" is simulated.** X5 is a finished 2019 experiment. Say this plainly in every report, README and demo.

## Constraints from the professor

- One dataset only.
- No classical ML models as project models (only recent transformer or LLM approaches). Classical learners may appear only as reference rows with his approval.
- Statistical procedures (sequential tests, doubly robust formulas, CUPED) are allowed.
- Gains are claimed only with bootstrap confidence intervals.

## Who owns what (semester 1)

- **Amay:** data collection (download, raw files and manifest in the bucket, 1% sample), full EDA and findings, replay producer.
- **Prathamesh:** cleaning and all feature extraction (silver, feature library, gold, labels and folds, quality checks, data dictionary).
- **Tejas:** Google Cloud setup and storage layers, bronze Parquet, Airflow DAGs, Pub/Sub, stream sink and consumer, monitoring, CI.
- **Yashashree:** baseline reproduction, evaluation harness, M0 on gold.

Do not implement another owner's work. If a task needs it, write down the interface and the open question instead.

# Project Plan: Causal Uplift Engine

**Team:** 5 members · **Tracking:** Linear (team key `CUE`) + GitHub · **Status:** Semester 1 kickoff

This is the document every team member should read first. It explains what we are building, why, who owns what, and how we avoid getting stuck.

---

## 1. The project in one paragraph

Marketing teams send ads and emails to millions of customers, but many of those customers would have bought anyway. **Uplift models** estimate, for each customer, how much a campaign actually changes their behavior, so the company can target only the people it persuades. We reproduce a recent state-of-the-art uplift model, **CausalPFN (NeurIPS 2025)**, on two real randomized marketing experiments (**Hillstrom** and **Criteo**). We then build our own transformer that fixes CausalPFN's main weakness: it gets *worse* on large campaigns, which are exactly the ones industry cares about.

## 2. Problem statement

CausalPFN is a pretrained transformer. It reads rows from a past experiment ("context") and predicts each new customer's uplift without any training. It can only read about 50,000 rows at once. On larger tables it has to pick a subset of rows to read, and its authors report that performance drops on the full Criteo table (paper Table 7, Appendix D.6). They suspect this row-selection step is the cause and list scaling as open future work.

**Our question:** can a transformer that is *trained* on the full campaign, and that reads the most similar treated and untreated customers for each prediction, remove this drop?

## 3. Scope

| Semester | Deliverables |
|---|---|
| **Semester 1 (now)** | Data collection, Snowflake warehouse, automated pipeline, CausalPFN reproduction, **Model 1** |
| Semester 2 | Models 2, 3, 4 (Section 6) and the final report |

## 4. Datasets

| Dataset | Size | Treatment | Outcome | Our use |
|---|---|---|---|---|
| **Hillstrom** (MineThatData, 2008) | 64,000 customers, 8 fields, 3 arms (men's email, women's email, no email) | Email | Site visit in 2 weeks | Small-data test; reproduction |
| **Criteo Uplift** (Criteo AI Lab) | 10% sample used by the paper (~2.5M rows per paper; record the exact count), 12 anonymized features | Eligible for ads (85% treated) | Visit (also conversion, exposure) | Large-data test; Model 1's main target |

Both are loaded with `scikit-uplift` (`fetch_hillstrom`, `fetch_criteo(percent10=True)`), exactly as the authors do.

**License note:** Criteo is CC BY-NC-SA 4.0 (non-commercial). Do not commit data to GitHub. Data lives in Snowflake and on the college storage only.

## 5. Baseline: CausalPFN reproduction

- Paper: Balazadeh et al., *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*, NeurIPS 2025. https://arxiv.org/abs/2506.07918
- Code: https://github.com/vdblm/CausalPFN (pinned as submodule `third_party/CausalPFN` at commit `7da4afa`, the version closest to the paper; see `baseline/README.md`)
- Protocol: 5-fold honest splitting, seed `82718`, normalized Qini score, comparison methods X/S/T/DA-Learner (EconML).

### Numbers we must reproduce

Normalized Qini: higher is better, and the best method in each column = 1.000.

| Paper table | Setting | CausalPFN | Best method |
|---|---|---|---|
| Table 2 | Hillstrom arm 1, full 64k | 0.992 | S-Learner 1.000 |
| Table 2 | Hillstrom arm 2, full 64k | 0.968 | S-Learner 1.000 |
| Table 2 | Criteo, 50k stratified subsample | 0.859 | X-Learner 1.000 |
| Table 7 | Criteo, full table | 0.939 | S-Learner 1.000 |

**Pass rule:** within ±0.03 of the paper (confirm tolerance with the instructor). Report raw and normalized Qini.

**Known traps:**
1. **Arm labels:** the paper text calls Hill(1) the men's email, but the code sets Hill(1) = women's email. Always report by arm name.
2. **Normalization:** normalized scores depend on the best baseline in each column. Different EconML versions can shift CausalPFN's normalized score. Use the authors' pinned environment (`env_reproduce.yaml`).
3. **Preprocessing:** use the authors' exact code first. No "improvements" until the reproduction passes.
4. **GPU memory:** the paper uses a 50k-row context. Test on the college GPU first (milestone M0).

## 6. Our models

All four are new models we train. M1–M3 share one transformer backbone, so M2 and M3 extend M1 instead of starting from scratch.

| Model | Type | Data | Fixes (paper limitation) | When |
|---|---|---|---|---|
| **M1: Retrieval uplift transformer** | New transformer | Criteo, Hillstrom | Drop on large tables (iii) | **Semester 1** |
| M2: Multi-campaign transformer | M1 + treatment embeddings | Hillstrom (3 arms) | Binary treatment only (iv) | Semester 2 |
| M3: Exposure-aware transformer | M1 + exposure head | Criteo | Assigned ≠ actually shown the ad (v) | Semester 2 |
| M4: Fine-tuned campaign LLM | LLM + LoRA | Hillstrom | Predict response to a campaign described in text, including unseen ones | Semester 2 |

### 6.1 Model 1 in detail (this semester)

```
customer features ──► feature tokens ──► Transformer encoder (3 layers) ──► customer vector
                                                                             │
   32–64 most similar TREATED customers (vector + outcome) ─► attention ─► P(visit | ad)
   32–64 most similar CONTROL customers (vector + outcome) ─► attention ─► P(visit | no ad)
                                                                             │
                                                uplift = P(visit | ad) − P(visit | no ad)
```

| Part | Decision |
|---|---|
| Stage A (M1a) | Encoder + two outcome heads, no retrieval. This is the guaranteed fallback result |
| Stage B (M1b) | Add attention over retrieved treated and control neighbors |
| Loss | Binary cross-entropy on the observed arm only; control examples weighted up (Criteo is 85% treated) |
| Leakage rule | A customer never retrieves itself or any test-fold customer. Enforced by a unit test |
| Retrieval | Start with scaled raw features; later refresh using the encoder's own vectors every few epochs |
| Model selection | Early stopping on **validation Qini**, not loss |
| Tuning | ~20 Optuna trials on a 500k-row Criteo subset |
| Final runs | 5 folds × 3 seeds, same folds as the paper |
| Tracking | Weights & Biases; every run logs the data snapshot version and git commit |

**M1 is compared against:** CausalPFN full-table (0.939), S-Learner (best, 1.000), CausalPFN 50k (0.859), and M1a vs. M1b (does retrieval help?).

**If M1b does not beat the baseline:** we report M1a and M1b against CausalPFN. That is still a trained model and a clear answer to the authors' open question.

## 7. Architecture and tech stack

```
                 GitHub Actions (push to main / manual run)
                                   │
  ┌──────────┐   ┌──────────────┐  │  ┌──────────────────────────┐   ┌────────────────────┐
  │ EXTRACT  │──►│ LOAD         │──┼─►│ TRANSFORM (dbt)          │──►│ EXPORT             │
  │ sklift → │   │ PUT + COPY   │  │  │ RAW → STAGING →          │   │ validated tables → │
  │ Parquet  │   │ into         │  │  │ VALIDATED → ANALYTICS    │   │ Parquet, ordered   │
  │ + MD5    │   │ Snowflake RAW│  │  │ + dbt tests              │   │ by row_id          │
  └──────────┘   └──────────────┘  │  └──────────────────────────┘   └─────────┬──────────┘
                                   │                                           │
                         College GPU server                                    ▼
  ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────────┐
  │ PREPARE              │──►│ BASELINE / M1 TRAIN  │──►│ EVALUATE                     │
  │ authors' preprocess  │   │ CausalPFN, learners, │   │ Qini, normalization,         │
  │ + folds + equality   │   │ M1a, M1b             │   │ bootstrap → results back     │
  │ check + neighbors    │   │ (W&B tracking)       │   │ to Snowflake ANALYTICS       │
  └──────────────────────┘   └──────────────────────┘   └──────────────────────────────┘
```

| Layer | Tool | Cost |
|---|---|---|
| Warehouse | **Snowflake** (X-Small warehouse, auto-suspend 60s) | Trial: $400 credits / 30 days. Ask about the Snowflake student program (reported 120-day accounts) |
| Transforms + data tests | dbt-core with `dbt-snowflake` (and `dbt-duckdb` as a fallback) | Free |
| Orchestration | GitHub Actions | Free tier (2,000 min/month private; more with GitHub Education) |
| Training | College GPU | Free |
| Experiment tracking | Weights & Biases (academic) | Free |
| Project management | Linear + GitHub integration | Free tier |

**Why `row_id` matters:** the authors' folds depend on row order. Extract assigns a `row_id` (original row position), and every export is sorted by it. Otherwise our folds won't match theirs and the reproduction fails.

**Snowflake trial risk:** if the trial ends mid-semester, the account is suspended. Mitigation: raw Parquet and snapshots are also kept on college storage, and every dbt model is portable SQL so we can switch the dbt target to DuckDB in one config change.

## 8. Team roles

Full task lists, handoffs, and backups for each person: [WORK_DIVISION.md](WORK_DIVISION.md).

| # | Role | Name | Main responsibility |
|---|---|---|---|
| 1 | **Data engineer (lead)** | Yashashree | Repo + Linear setup, extract and load, Snowflake, GitHub Actions, neighbor lists |
| 2 | Analytics engineer | _TBD_ | dbt models, data tests, randomization check, exports |
| 3 | Baseline engineer | _TBD_ | Model-ready data with the authors' preprocessing, CausalPFN reproduction |
| 4 | Model engineer | _TBD_ | Model 1 (M1a, M1b), training, tuning |
| 5 | Evaluation lead | _TBD_ | Metrics, X/S/T/DA learners, leakage test, results, report |

## 9. Milestones and checks

Milestones match Linear. Each one has a check; if the check fails, follow the fallback before moving on.

| Milestone | Done when | If it fails |
|---|---|---|
| **M0: Setup** | Everyone can clone, install, connect to Snowflake, and run a GPU job. The college GPU runs CausalPFN with a large context | Ask for a larger GPU, or agree a smaller-context protocol with the instructor |
| **M1: Data pipeline** | One GitHub Actions run loads both datasets into Snowflake and all dbt tests pass | Run dbt on the DuckDB target while fixing Snowflake |
| **M2: Model-ready data** | Our arrays are identical to the authors'; folds and neighbor lists exported | Stop and fix preprocessing first |
| **M3: Baseline reproduced** | Table 2 within tolerance; Table 7 Criteo drop reproduced | Check package versions and arm labels; use a 500k-row Criteo version to show the trend |
| **M4: Model 1a** | M1a trains on all folds and is evaluated | Simplify the model until it fits a small sample, then scale up |
| **M5: Model 1b** | Retrieval model trained, tuned, and evaluated with 3 seeds | Report M1a as the semester result |
| **M6: Report** | Results tables, write-up, Semester 2 proposal for M2–M4 | — |

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Snowflake trial expires | Warehouse unavailable | College storage copies + DuckDB fallback target |
| Reproduction off by more than tolerance | Weak baseline | Pinned environment, raw + normalized scores, check at M3 |
| GPU memory too small for 50k context | Can't match paper settings | Test at M0 |
| Leakage in retrieval | Falsely good M1 scores | Unit test, plus a review by Person 5 |
| Tree models beat M1 | Weaker headline | M1a vs. M1b and M1 vs. CausalPFN are still valid findings |
| Novelty overlap with new papers | Weaker contribution | Literature search before each deliverable |

## 11. How we work

Read [CONTRIBUTING.md](../CONTRIBUTING.md). In short:
- Every piece of work starts as a Linear issue.
- Branch names come from Linear ("Copy git branch name"), e.g. `yashashree/cue-12-extract-hillstrom`.
- PR titles include the issue ID, and the description says `Fixes CUE-12`, so Linear closes the issue on merge.
- No direct pushes to `main`. One reviewer per PR. CI must pass.

## 12. Open questions for the team

1. Submission deadlines.
2. College GPU model and memory, and how jobs are submitted (SLURM or direct).
3. Snowflake: 30-day trial or student account? Who owns the account?
4. Instructor confirmation that a trained transformer evaluated against the reproduced CausalPFN counts as "our model."

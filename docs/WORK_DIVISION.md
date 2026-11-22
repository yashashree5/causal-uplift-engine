# Work Division: Semester 1

Five roles, one owner each. Every role has its own tasks, the outputs it hands to others, and a backup person who reviews its PRs and can take over if needed.

Background on the project: [PROJECT_PLAN.md](PROJECT_PLAN.md). Issue list for Linear: [LINEAR_BACKLOG.md](LINEAR_BACKLOG.md).

## Summary

| # | Role | Name | Main responsibility | Backup |
|---|---|---|---|---|
| 1 | Data engineer (lead) | Yashashree | Getting raw data into Snowflake automatically | 2 |
| 2 | Analytics engineer | _TBD_ | Cleaning, testing, and exporting data in Snowflake with dbt | 1 |
| 3 | Baseline engineer | _TBD_ | Reproducing CausalPFN's published numbers | 5 |
| 4 | Model engineer | _TBD_ | Building and training Model 1 | 3 |
| 5 | Evaluation lead | _TBD_ | Metrics, comparison methods, results, and the report | 4 |

## How the work flows

```
Person 1: raw data → Snowflake RAW
          │
Person 2: STAGING → VALIDATED → export Parquet
          │
Person 3: authors' preprocessing + folds → snapshot ─────────────┬──────────────┐
          │                                                       │              │
Person 1: neighbor lists ───────────────► Person 4: Model 1 ◄────┘              │
                                                    │                            │
Person 3: CausalPFN predictions ──┐                 │                            │
Person 5: X/S/T/DA predictions ───┼──► Person 5: evaluation, results, report ◄───┘
Person 4: M1a / M1b predictions ──┘
```

People 3, 4, and 5 don't need to wait for the full pipeline. Person 3 can use the authors' loaders directly, and Persons 4 and 5 can start on a small Hillstrom sample.

---

## Person 1: Data engineer (lead)

**Owns:** project setup, data extraction, the Snowflake account, automation, and the neighbor lists for Model 1.

**Tasks:**
1. Set up GitHub: branch protection on `main`, repo secrets, PR template.
2. Set up Linear: team `CUE`, GitHub integration, labels, milestones, and issues from the backlog.
3. Create the Snowflake account and run `warehouse/snowflake_setup.sql` (database, warehouse, roles, resource monitor, CI service user).
4. Create Snowflake logins for all five members.
5. Finish `pipeline/extract.py`: download Hillstrom and Criteo, add `row_id`, write Parquet and the MD5 manifest.
6. Finish `pipeline/load_snowflake.py`: load the Parquet files into `RAW`.
7. Get the GitHub Actions `pipeline.yml` workflow running end to end.
8. Write `pipeline/neighbors.py`: for each fold, the nearest treated and control customers from the training rows only.
9. With Person 4, build the data loader that feeds neighbor lists into Model 1.
10. Write the pipeline and architecture section of the report, with Person 2.

**Delivers:**
- Raw tables in Snowflake for Person 2.
- Neighbor lists for Person 4.

**Done when:** one GitHub Actions run loads both datasets into Snowflake, and the neighbor lists pass Person 5's leakage test.

---

## Person 2: Analytics engineer

**Owns:** everything inside Snowflake after the raw load: dbt models, data tests, and exports.

**Tasks:**
1. Set up the dbt project with Snowflake and DuckDB connections, and check `dbt debug` works for every teammate.
2. Write and test the staging models `stg_hillstrom` and `stg_criteo`.
3. Add dbt tests: no nulls, allowed values, Hillstrom has 64,000 rows, Criteo row count as recorded, outcome rates in the expected range.
4. Build the randomization check: treated vs. control feature differences for Hillstrom and Criteo.
5. Build data summary tables for the report: group sizes, visit rates per group, overall effect per group.
6. Write `pipeline/export.py`: export validated tables to Parquet sorted by `row_id`, and check that `row_id` has no gaps.
7. Create a Snowflake table that Person 5's evaluation code writes results into.
8. Keep `docs/DATA_CONTRACTS.md` up to date.
9. Write the data and pipeline section of the report, with Person 1.

**Delivers:**
- Validated Parquet exports for Person 3.
- Summary tables for Person 5.

**Done when:** `dbt build` passes on Snowflake and on DuckDB, and the exported row order matches the extracted row order.

---

## Person 3: Baseline engineer

**Owns:** reproducing the CausalPFN paper's numbers, and preparing the exact data the models train on.

**Tasks:**
1. Add the CausalPFN repo under `third_party/` at commit `7da4afa` (done), and set up the authors' pinned environment on the college GPU.
2. Run the CausalPFN README example on the college GPU. Record the GPU model, memory, and the largest context that runs.
3. Write `pipeline/prepare.py`: the authors' preprocessing and 5-fold splits with seed 82718 for all four tasks.
4. Write the equality test: our arrays must be identical to the authors' arrays.
5. Run CausalPFN on both Hillstrom tasks (women's vs. none, men's vs. none).
6. Run CausalPFN on the Criteo 50k sample.
7. Run CausalPFN on full Criteo, or on 500k rows if full Criteo doesn't fit, and document which.
8. Save all predictions in the shared prediction format.
9. Write the baseline reproduction section of the report, with Person 5.

**Delivers:**
- Model-ready snapshots for Persons 4 and 5.
- CausalPFN predictions for Person 5.

**Done when:** CausalPFN predictions exist for all four tasks and the equality test passes.

---

## Person 4: Model engineer

**Owns:** Model 1, the retrieval uplift transformer.

**Tasks:**
1. Read the CausalPFN paper (Sections 3, 5, Appendix D) and the TabR and TARNet papers for background.
2. Build the feature tokenizer and transformer encoder (`models/m1/encoder.py`).
3. Build **M1a**: the encoder with two outcome heads and an arm-weighted loss.
4. Write the training loop: checkpoints every epoch, resume after a crash, W&B logging, early stopping on validation Qini.
5. Sanity check: M1a must fit a 1,000-row sample almost perfectly before running on full data.
6. Train M1a on Hillstrom and Criteo, all 5 folds.
7. Build **M1b**: attention over treated and control neighbors.
8. Add learned retrieval: rebuild neighbor lists from the encoder's own vectors between epochs.
9. Tune with Optuna on a 500k-row Criteo subset (about 20 trials), and commit the best settings to `configs/m1.yaml`.
10. Run the final M1a and M1b runs: 5 folds × 3 seeds on Hillstrom and Criteo.
11. Write the Model 1 section of the report.

**Delivers:** M1a and M1b predictions for Person 5.

**Done when:** predictions exist for every fold and seed, and each run in W&B records the data version and git commit.

---

## Person 5: Evaluation lead

**Owns:** the metrics, the comparison methods, the final results, and the report.

**Tasks:**
1. Set up the Weights & Biases team project.
2. Finish `eval/`: Qini (already matched to the authors' code), normalization as in the paper's tables, and bootstrap confidence intervals.
3. Write the leakage test for neighbor lists. It must fail on a deliberately leaky list and pass on the real one.
4. Run the comparison methods X-, S-, T-, and DA-Learner on all four tasks, using the authors' settings.
5. Build the reproduction table: our numbers vs. paper Tables 2 and 7, raw and normalized.
6. Build the final comparison table: M1a, M1b, CausalPFN, and the four learners, with confidence intervals.
7. Make the Qini curve figures.
8. Run the literature search for related work and novelty, with Person 4.
9. Lead the final report and assemble everyone's sections.
10. Draft the Semester 2 proposal for Models 2–4 with the whole team.

**Delivers:** evaluation code that everyone uses, the result tables, and the report.

**Done when:** every number in the report is produced by a script in the repo.

---

## Shared by everyone

- Review the PRs of the person you back up.
- Keep your Linear issues up to date.
- Write your own section of the report.
- Join the weekly sync and share progress, blockers, and next steps.
- Follow [CONTRIBUTING.md](../CONTRIBUTING.md): no data or credentials in Git, and all work goes through PRs.

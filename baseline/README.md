# Baseline Reproduction: CausalPFN

Runs **Hillstrom** and **Criteo** uplift benchmarks using the folds, comparison methods, and Qini code from the pinned CausalPFN submodule. The reports compare with Tables 2 and 7 and record the dataset used for each run.

## Benchmark reports

- [Hillstrom](../reports/baseline_results/hillstrom.md) covers both email arms and compares them with Table 2.
- [Criteo Uplift v2.1](../reports/baseline_results/criteo.md) covers separately drawn stratified samples of **50,000** and **2,500,000** rows. Both use the verified v2.1 archive as their source. Tables 2 and 7 provide reference scores from the paper's cohorts.

## Run the notebooks

- [Hillstrom notebook](../notebooks/baseline_hillstrom.ipynb) runs both email arms. It can download the original Hillstrom archive or read a verified copy supplied through `HILLSTROM_DATA`.
- [Criteo notebook](../notebooks/baseline_criteo.ipynb) runs the **50,000-row v2.1** benchmark. Save the [official v2.1 archive](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz) as `data/criteo-v21-source/criteo.csv.gz`, or set `CRITEO_V21_DATA_DIR` to the folder containing `criteo.csv.gz`.

Both notebooks show the fold log, Qini comparison, and curves. Set `CAUSALPFN_CHECKPOINT` to use a local copy of the published checkpoint.

## Reproduce the 2.5M Criteo v2.1 run

Use a CUDA GPU and the [baseline GPU requirements](requirements.txt). Initialize the pinned submodule with `git submodule update --init --recursive`. Save the [official Criteo v2.1 archive](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz) as `data/criteo-v21-source/criteo.csv.gz` and the [published CausalPFN checkpoint](https://huggingface.co/vdblm/causalpfn/blob/main/causalpfn_v0.pt) as `data/causalpfn_v0.pt`.

From the repository root, run these commands in Bash. Use the same output directory for both commands.

```bash
python -m baseline.prepare_criteo_v21 \
  --source data/criteo-v21-source/criteo.csv.gz \
  --output-dir outputs/criteo-2p5m-v21-rerun

LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python -m baseline.run_criteo_v21 \
  --checkpoint data/causalpfn_v0.pt \
  --output-dir outputs/criteo-2p5m-v21-rerun
```

The sampler verifies the archive checksum and freezes the stratified cohort and its five folds. The runner verifies the cohort and checkpoint, logs progress for all 25 method-folds, and saves predictions and aggregate Qini reports under the output directory. It can resume a run with the same configuration. The original run used PyTorch `2.3.1+cu121` and PyArrow `25.0.1`. The cohort and run manifests record the seeds, package versions, files, and checksums. [Original run results and settings](../reports/baseline_results/criteo.md#2500000-rows-from-criteo-v21)

## What runs

| Dataset | Task | Paper column | Rows | CausalPFN target |
|---|---|---|---|---|
| Hillstrom | `hill_womens_vs_none` | Table 2, Hill(1) | ~42.7k (two arms) | 0.992 |
| Hillstrom | `hill_mens_vs_none` | Table 2, Hill(2) | ~42.6k (two arms) | 0.968 |
| Criteo | `criteo_sub50k` | Table 2, Criteo | 50,000 | 0.859 |
| Criteo | `criteo_full` (`--include-full`) | Table 7, Criteo | full 10% sample (paper: ~2.5M) | 0.939 |

Methods per task: CausalPFN, X-, S-, DA-, T-Learner (EconML, `hpo=False`), each on the same 5 folds (seed 82718). Normalization divides by the best method in each task, so **all five methods must run** for the normalized numbers to be comparable with the paper.

## How to run (GPU server)

```bash
git clone --branch baseline-reproduction --recurse-submodules https://github.com/yashashree5/causal-uplift-engine.git
cd causal-uplift-engine
python3.10 -m venv .venv-baseline && source .venv-baseline/bin/activate
pip install -r baseline/requirements.txt
pip install -e . --no-deps

bash scripts/run_baseline_hillstrom.sh             # Hillstrom, both arms
bash scripts/run_baseline_criteo.sh                # Criteo 50k subsample
bash scripts/run_baseline_criteo.sh --include-full # + full Criteo (Table 7), slow
```

Outputs:
- Predictions: `outputs/predictions/<method>/<task>/seed82718_fold<k>.parquet`
- Run metadata: `outputs/predictions/<task>_run.json`
- Results: `reports/baseline_hillstrom.md|csv`, `reports/baseline_criteo.md|csv` (commit these)

The Criteo shell script uses the paper's historical scikit-uplift archive. That archive returned HTTP 404 during the v2.1 benchmark runs.

Re-running skips finished folds. To run one piece: `python -m baseline.run --task criteo_full --methods causalpfn`.

## Decisions and known issues

- **Row count:** each task prints its number of rows and warns if it differs from the expected count by more than 5% (Hillstrom arms: 42,693 and 42,613; Criteo full: the paper's ~2.5M). Record any warning in TEA-6, because a different sample size can explain a gap with the paper.
- **Pinned source `7da4afa` (Feb 2026), not the latest `b7ed2f1`.** The September 2026 release (0.1.5) replaced FAISS and changed how context rows are chosen on large tables ("prevent duplicate context rows"). That can change full-Criteo results, so we use the version closest to the paper. Running at `b7ed2f1` as a second check is useful.
- **Arm labels:** the code's Hill(1) is women's e-mail vs. none; the paper text says men's. We name tasks by arm.
- **Comparison methods use random forests without a fixed seed** (authors' code), so their scores vary slightly between runs. This moves CausalPFN's normalized score too. Report raw Qini alongside.
- **Data download:** the first run downloads Hillstrom and Criteo through scikit-uplift, and the CausalPFN weights from Hugging Face (`vdblm/causalpfn`). Make sure the cache directories have space.
- **GPU memory:** CausalPFN uses a context of 50,000 rows. If it runs out of memory, record the GPU and stop; don't lower the context without agreeing with the team, because results would no longer match the paper's setup.
- **Full Criteo** is slow: CausalPFN uses retrieval, and the random-forest learners train on ~1M+ rows per fold on CPU.
- `row_id` in prediction files is the index into the task's arrays (after arm filtering or subsampling), not the raw-table `row_id`.

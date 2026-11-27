# X5 RetailHero uplift: exploratory data analysis

Reusable, headless EDA for the five X5 tables (`clients`, `uplift_train`, `uplift_test`, `products`, `purchases`).
The notebook `x5_eda.ipynb` and the CLI `run_eda.py` run the **same code** (`x5_eda_lib.py`).

| File | Purpose |
|---|---|
| `x5_eda.ipynb` | Ten-section notebook; the **only** `.ipynb` in the EDA; carries the outputs of the **full-data** run (aggregates and figures only) |
| `x5_eda_lib.py` | Analysis code (DuckDB, bounded memory); `run_eda(settings)` is the entry point |
| `x5_eda_plots.py` | Figures, drawn from the result tables only |
| `run_eda.py` | CLI: run the analysis and write tables, figures and `summary.json` |
| `execute_notebook.py` | Run the notebook headlessly; `--in-place` saves its outputs into `x5_eda.ipynb`; exit 1 if a cell raised |
| `requirements.txt` | Pinned dependencies (includes `../requirements-x5.txt`) |
| `make_findings.py` | Generate `FINDINGS.md` from the full-data outputs (refuses sample outputs) |
| `FINDINGS.md` | Generated findings: status, one-page summary, findings table (Observation, Evidence, Recommended cleaning action, Caveat), key tables, open questions |
| `../configs/x5/eda.yaml` | All settings: input dirs, output root, segments, thresholds, DuckDB limits |

## Run it

```bash
pip install -r eda/requirements.txt        # Python 3.11

# sample mode: the 1% client sample from ingestion/sample_clients.py (see ingestion/SAMPLING.md)
python eda/run_eda.py --mode sample --input-dir raw_sample

# full mode: all rows. Plain CSV is read in parallel by DuckDB; .csv.gz works too but is single-threaded.
python ingestion/download_x5.py --extract              # once; see ingestion/README.md
python eda/run_eda.py --mode full --input-dir data/x5/extracted

# the notebook, run headlessly; results land in eda/outputs/<mode>/. --in-place also saves the outputs into x5_eda.ipynb
python eda/execute_notebook.py --mode full --input-dir data/x5/extracted --in-place
# or interactively: set X5_EDA_MODE / X5_EDA_INPUT_DIR and open eda/x5_eda.ipynb
```

`--mode` is a **label** that is printed in the notebook banner, recorded in `summary.json` (`run.mode`) and used for the
output folder. It does not change what is computed. A sample run describes the sample, not the full data; never quote
its numbers as full-data results.

### Configuration precedence

Input directory: `--input-dir` > `X5_EDA_INPUT_DIR` > `eda.input_dirs.<mode>` in `configs/x5/eda.yaml`.
Campaign cutoff: `--cutoff` (`X5_CAMPAIGN_CUTOFF` in the notebook) > `campaign.cutoff` in the YAML > unset.
Output directory: `--output-dir` (`X5_EDA_OUTPUT_DIR` in the notebook) > `<eda.output_root>/<mode>`.
Every other setting comes from the YAML. Relative paths are resolved against the repository root.

## Findings

```bash
python eda/make_findings.py     # reads eda/outputs/full, writes eda/FINDINGS.md
```

`FINDINGS.md` is **generated, never edited by hand**: every number is read from `eda/outputs/full/summary.json` and
`tables/*.csv`, and the generator refuses outputs of a `sample` run. It opens with the status block and a one-page summary
(about a dozen one-line cleaning decisions that point at the rows of the findings table), followed by the provenance, the row counts
against the work plan, the findings table, key tables and the open questions. The document says it is **not reviewed** by the
cleaning or baseline owners until they review it. It is regenerated after every full run (`--cutoff` once the campaign date
is known).

## Outputs (git-ignored, `eda/outputs/<mode>/`)

`summary.json` (all scalar results by section), `tables/*.csv`, `figures/*.png`, `work.duckdb` (typed working tables,
reused while the inputs and settings are unchanged; delete it or pass `--force` to rebuild).
Outputs are never committed: `eda/outputs/` is in `.gitignore`.

## Method notes

- **Text first.** CSVs are read as text and cast with `TRY_CAST`; values that fail to cast are counted
  (`files.cast_failures_total`, table `cast_failures`), not dropped silently. IDs are never cast, so leading zeros survive.
  (`TRY_CAST(NULLIF(x, ''))` is deliberate: a bare `TRY_CAST` on a DuckDB CSV column is pushed into the reader and
  raises on the first bad value instead of returning NULL; `tests/x5/test_x5_eda.py` covers this.)
- **Transactions.** `purchases` has one row per product. The key is `(client_id, transaction_id)`. `transaction_id` alone is
  *not* unique across clients in the real data (section 10). `purchase_sum` and the points columns are aggregated once per
  transaction, never per row; the notebook quantifies the overstatement of a row-wise sum.
- **Cutoff and leakage.** No campaign date is published. While `campaign.cutoff` is unset the leakage check is reported as
  `blocked` (not passed) and recency uses the **provisional** anchor `max(purchase timestamp)`.
- **Segments.** Age bands, spend tiers and recency bands use only attributes and purchase history. Intervals are Wald 95%
  intervals without multiplicity correction. A segment is `sparse` if an arm has fewer than `eda.min_arm_size` clients or a
  treatment x outcome cell has fewer than `eda.min_cell_events` clients.
- **Duplicates** in `purchases` are counted with a 64-bit row hash (a collision could only under-count).
- **Reproducibility.** Counts and tables are exact and repeatable. With `duckdb.threads > 1`, floating-point sums and
  standard deviations can differ in the last bits between runs (the tests compare with a relative tolerance of 1e-9).
- **Memory.** DuckDB runs with `duckdb.memory_limit` / `duckdb.threads` from the YAML and spills to `<output>/duckdb_tmp`.

## Verified runs

Measured on an 8-core, 16 GB laptop with `memory_limit: 6GB`, `threads: 6`:

| Mode | Input | Wall time | Work DB |
|---|---|---|---|
| full | `data/x5/extracted` (45,786,568 purchase rows) | about 66 s (typed tables 57 s, all sections about 10 s) | 1.6 GB |
| sample | `data/x5/raw_sample` (447,380 purchase rows) | about 5 s | 16 MB |

## Tests

`tests/x5/test_x5_eda.py` runs the analysis on a tiny hand-written dataset whose statistics are derived by hand in the test
(not random, not real X5 data), and executes the notebook headlessly on it.

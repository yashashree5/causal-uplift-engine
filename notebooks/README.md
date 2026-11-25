# RetailHero ABCD results notebook

Open [the executed notebook](retailhero_abcd_analysis.ipynb) to compare CausalPFN and PTONet across all 16 core configurations and the 10%, 20%, 30% and 50% SMS budgets. The saved figures and tables can be viewed directly on GitHub.

To run it locally, use Python 3.10 or newer and run these commands from the repository root.

```sh
python -m pip install -e ".[analysis]"
python -m jupyterlab notebooks/retailhero_abcd_analysis.ipynb
```

Select **Run All Cells** to refresh the analysis from the [published summaries](../reports/retailhero_abcd/README.md) on CPU. Set `REACH` and `INCLUDE_CONTROLS` near the top to change the configuration view. For individual budget profiles, choose `BUDGET_HOST` and `BUDGET_CONFIGS` in the budget comparison section. The final `EXPORT` cell saves PNGs and CSVs under `reports/retailhero_abcd/notebook-exports/`.

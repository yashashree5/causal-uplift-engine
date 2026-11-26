# RetailHero ABCD results notebook

Open [the executed notebook](retailhero_abcd_analysis.ipynb) to compare CausalPFN and PTONet across all 16 core configurations and the 10%, 20%, 30% and 50% SMS budgets. The saved figures and tables can be viewed directly on GitHub.

The direct Qwen comparison near the end shows how a frozen Qwen3 8B ranks the same 40,008 customers at all four budgets. Its table includes Original and ABCD for both hosts, the standalone LoRA score, and random targeting. The chart shows the difference from each of those six policies at the primary 30% budget, with paired 95% confidence intervals. [Direct LLM protocol and results](../reports/retailhero_abcd/DIRECT_LLM.md).

To run it locally, use Python 3.10 or newer and run these commands from the repository root.

```sh
python -m pip install -e ".[analysis]"
python -m jupyterlab notebooks/retailhero_abcd_analysis.ipynb
```

Select **Run All Cells** to refresh the analysis from the [published summaries](../reports/retailhero_abcd/README.md) on CPU. Set `REACH` and `INCLUDE_CONTROLS` near the top to change the configuration view. For individual budget profiles, choose `BUDGET_HOST` and `BUDGET_CONFIGS` in the budget comparison section. The `EXPORT` cell saves PNGs and CSVs under `reports/retailhero_abcd/notebook-exports/`.

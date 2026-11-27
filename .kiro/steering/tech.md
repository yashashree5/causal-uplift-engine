---
inclusion: always
---

# Technical rules

## Language and dependencies
- Python 3.11. Pin every dependency in a requirements file (`requirements-x5.txt` for the shared X5 tools; `eda/requirements.txt` and `streaming/producer/requirements.txt` add task-specific pins).
- Use DuckDB or Polars for the 45.8M-row purchases table; never load it with pandas. Keep memory bounded (DuckDB `memory_limit`, `threads`, spill directory from config).

## Configuration
- Every script takes paths, seeds, cutoff, bucket, project and topic from YAML under `configs/` (task files `extends: x5.yaml`). Nothing is hard-coded.
- Precedence is **CLI flag > environment variable > YAML > built-in default**, and is documented in each README.
- No credentials in the repo, ever. Use Application Default Credentials (`gcloud auth application-default login`) or `GOOGLE_APPLICATION_CREDENTIALS` pointing outside the repository. The emulator needs none (`PUBSUB_EMULATOR_HOST`).
- Never commit raw data, samples, generated outputs, model weights or `.env` files. `data/`, `raw_sample/`, `eda/outputs/` and `streaming/producer/{outputs,checkpoints}/` are git-ignored.

## Quality gates
- Tests: `pytest tests/x5 -q`. Lint: `ruff check --config configs/x5/ruff.toml <paths>` and `ruff format --check --config configs/x5/ruff.toml <paths>`.
- Tests use small hand-made fixtures whose expected values are derived by hand in the test. Fixture numbers are never reported as X5 results.
- State which kind of check was done: local test, emulator, full-data run, or real cloud. A missing cloud credential must not block local work, and must be reported as "not verified".

## Git and review
- One branch per task, one pull request per branch, reviewed by one teammate. Every branch and PR carries its Linear issue ID (for example `298-NN`); until the ID is known write `298-??`.
- Stacked branches: open each PR against its parent branch; document the merge order.
- Never push to `main`, merge, change branch protection or webhooks, or update Linear statuses. Do not commit or push unless the user asks.

## Style
- `from __future__ import annotations`; type hints; no `pass`-only or TODO-only code; no `NotImplementedError` stubs.
- Exit codes are integers with documented meanings; errors name the file or setting at fault.

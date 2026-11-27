# Design Document: FINDINGS generator

## Overview

`build_findings(results)` returns a list of `Finding(observation, evidence, action, caveat)` computed from `summary.json` and the result tables. `build_summary(results, findings)` returns the page-one list and resolves `(#n)` references by searching the findings for the matching observation text, so references cannot drift when findings are added. `render(...)` assembles status, summary, provenance, row-count check, findings table, key tables and open questions. `main` reads `eda/outputs/full` and writes `eda/FINDINGS.md` atomically.

## Decisions

1. **Generated, deterministic.** Text is a pure function of the outputs, so a reviewer can regenerate and diff.
2. **Refuse non-full runs.** A sample or fixture must never appear as a full-data result.
3. **Adaptive wording.** If a data assumption fails (for example `purchase_sum` differs between rows), the finding and the summary line say so instead of repeating the usual wording.
4. **No claims about reviewers.** Review status is stated as "not reviewed".
5. **Work-plan figures come from config** (`configs/x5/raw_landing.yaml`), not from code.

## Testing

`tests/x5/test_x5_findings.py` (13): refusal of sample outputs, all fields present, structure and status text, numbers read from results, adaptive finding, expectation table, CLI, escaping, summary length and references.

## Verification status

Local tests passed. `eda/FINDINGS.md` was generated from the full-data outputs (19 findings, 12-line summary). **Not reviewed** by the cleaning owner; that review is a definition-of-done item.

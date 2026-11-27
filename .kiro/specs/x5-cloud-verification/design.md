# Design Document: cloud verification and open decisions

## Overview

No new code is needed for items 1 to 3; the tools already support the cloud paths (`land_raw.py`, `sample_clients.py --upload`, `replay.py`). This spec lists the commands, the evidence to collect and the wording allowed in the report.

## Runbook

1. **Raw landing** (needs Bucket and ADC):
   `python ingestion/land_raw.py --bucket <bucket> --project <project> --strict-expectations`
   Evidence: landing report with `mode: cloud`, counts per table, then a second run reporting no changes.
2. **Sample upload:** `python ingestion/sample_clients.py --upload --bucket <bucket> --project <project>`.
3. **Emulator:** start the Pub/Sub emulator (container or `gcloud beta emulators pubsub start`), set `PUBSUB_EMULATOR_HOST=localhost:8085`, then
   `pytest tests/x5/test_x5_replay.py -k emulator` and
   `python streaming/producer/replay.py --project x5-local --create-topic-in-emulator --speed 1d/1m --start 2019-01-01 --end 2019-01-02`.
4. **Cloud demo:** `python streaming/producer/replay.py --project <project> --topic x5-transactions --speed 1d/1m --start 2019-01-01 --end 2019-01-02 --clients-sample raw_sample`; read the dead-letter subscription count.
5. **Cutoff:** set `campaign.cutoff` in `configs/x5/eda.yaml`, run `python eda/execute_notebook.py --mode full --input-dir data/x5/extracted --in-place` and `python eda/make_findings.py`.

## Allowed wording

"Local test", "emulator check", "full-data run" and "cloud verified" are different claims. A manifest-only run, a local-directory rehearsal and a dry run are never "cloud verified".

## Open decisions (owner)

- Event id of shared transaction ids: consumer and sink owner.
- Points-spent sign and cleaning in batch and stream: feature-library and consumer owners (the consistency check on 1,000 clients needs identical treatment).
- `items.amount` source: contract owner.
- Real campaign date: team (not in the published data).

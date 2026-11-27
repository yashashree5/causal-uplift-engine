# Design Document: replay producer

## Overview

`event_source.py` builds the ordered event table in DuckDB (group product rows, flag shared ids, check and reject, sort, number with `seq`) and caches it as `replay_events_<fingerprint>.duckdb`. `events.py` holds pure assembly, validation and speed parsing. `publishers.py` has the Pub/Sub and JSON-lines publishers behind one interface. `replay.py` contains settings, the `Replayer` loop (injectable clock and sleep), checkpoints, the run summary and the CLI.

## Decisions

1. **Event id deviation.** 28 transaction ids are shared by two clients (56 events). Publishing the contract value would let a consumer that deduplicates by `event_id` drop real transactions, so shared ids get the client id appended. The consumer owner should confirm.
2. **Cached, resumable stream.** The cache is keyed by input file (name, size, mtime), content hash of the client list, time filters and event options, and rebuilt automatically when any changes. `seq` makes resume a range scan.
3. **At-least-once.** Acknowledgement futures drive a contiguous acked prefix that is checkpointed; an unacknowledged event may repeat after a crash. Stable event ids let consumers deduplicate.
4. **Points sign as stored.** The raw stream then equals the raw batch data. `replay.event.points_spent_sign: absolute` flips it. Batch/stream consistency requires the shared feature library to clean both identically (open decision).
5. **`items.amount` is configurable** (`item_amount_column`, default `trn_sum_from_iss`, unverified).
6. **Emulator-only topic creation.** Cloud topics belong to the infrastructure owner.
7. **Unpaced dry run by default** so a demo file or test does not wait.

## Testing

`tests/x5/test_x5_replay.py` (72 tests: 71 pass, 1 skipped): contract, grouping across far-apart rows, order and tie-breaks, shuffled input, rejects, event-id modes, filters, speed parsing, fake-clock pacing, backpressure, failures, ack timeout, checkpoint and resume, dry run, config precedence, fake Pub/Sub client, credentials message. The emulator round-trip test is skipped unless `PUBSUB_EMULATOR_HOST` is set.

## Verification status

- Local tests passed.
- Full-data dry run to a JSON-lines file: 8,045,229 events, 0 rejects, about 32,600 events/s (cache build about 48 s, replay 247 s, about 4.5 GB peak RSS); every event valid, ids unique, no order violation, item rows 45,786,568.
- Paced check: 2 hours of data at `1h/10s` took 20.1 s, `max_lag_s` 0.01.
- **Emulator: not verified** (no Docker daemon, no `gcloud`). **Real Pub/Sub: not verified** (no project, topic or credentials).

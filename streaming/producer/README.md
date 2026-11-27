# X5 replay producer (simulated live stream)

X5 is a finished 2019 experiment, not a live feed. This producer **simulates** one: it reads the `purchases` table,
turns every `(client_id, transaction_id)` into one event, orders the events by time and publishes them to a Pub/Sub
topic at an accelerated clock. The same code can write the events to a JSON-lines file instead (`--dry-run`).

It does **not** provision a topic, bucket, subscription or IAM, and it does not clean data. Those belong to the
infrastructure and silver-layer owners.

## Install and run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r streaming/producer/requirements.txt

# 1. No cloud needed: write the first 1000 events as JSON lines (default output: streaming/producer/outputs/events_dry_run.jsonl)
python streaming/producer/replay.py --dry-run --max-events 1000 --input-dir data/x5/extracted

# 2. Against the Pub/Sub emulator (see "Verification status")
export PUBSUB_EMULATOR_HOST=localhost:8085
python streaming/producer/replay.py --project x5-local --topic x5-transactions --create-topic-in-emulator --speed 1d/1m \
    --start 2019-01-01 --end 2019-01-02

# 3. Real Pub/Sub: Application Default Credentials only, topic must already exist
gcloud auth application-default login          # or GOOGLE_APPLICATION_CREDENTIALS=<key file outside this repository>
python streaming/producer/replay.py --project <project-id> --topic <topic> --speed 1d/1m
```

Configuration is `configs/x5/replay.yaml` (extends `configs/x5/x5.yaml`). **Precedence: CLI flag > environment
variable > YAML > built-in default.** Environment variables: `GOOGLE_CLOUD_PROJECT`, `PUBSUB_EMULATOR_HOST`,
`X5_REPLAY_INPUT_DIR`. `gcp.project` is intentionally unset in the YAML. Nothing credential-like is stored in the
repository.

## Flags

| Flag | Meaning |
|---|---|
| `--config` | YAML file (default `configs/x5/replay.yaml`) |
| `--input-dir` | directory with `purchases.csv` or `purchases.csv.gz`. Plain CSV prepares faster than gzip (a gzip file is read by one thread) |
| `--speed` | see below; default `replay.speed` (`1d/1m`); with `--dry-run` the default is `max` |
| `--start`, `--end` | time window of the replay. `--start 2019-01-02` is inclusive. `--end 2019-01-03` (date only) includes that whole day; `--end 2019-01-03T06:00:00` is an exclusive time |
| `--clients-sample` | a `raw_sample/` directory (uses `selected_client_ids.csv`, else `clients.csv[.gz]`) or a CSV with a `client_id` column; replays only those clients |
| `--max-events N` | stop after N events |
| `--dry-run`, `--output` | write JSON lines instead of publishing (`--output -` = stdout) |
| `--project`, `--topic` | destination; topic defaults to `gcp.pubsub_topic` |
| `--emulator-host` | sets `PUBSUB_EMULATOR_HOST` for this run |
| `--create-topic-in-emulator` | create the topic. **Refused unless an emulator is configured** |
| `--event-id-collisions` | `disambiguate` (default), `keep`, `error` (see "Event ids") |
| `--resume`, `--reset-checkpoint`, `--checkpoint/--no-checkpoint` | checkpointing (see "Delivery semantics") |
| `--prepare-only`, `--force-prepare` | build (or rebuild) the ordered event cache and print its statistics |
| `--run-id` | label of the run summary file (default: UTC timestamp) |

Exit codes: `0` ok, `1` the stream could not be built or the internal order check failed, `2` usage/configuration,
`3` finished but some transactions were rejected (listed in `streaming/producer/outputs/rejects_*.jsonl`),
`4` a message was not acknowledged (resume with `--resume`), `5` publisher setup failed (no project, no credentials,
topic missing or not readable).

## `--speed` semantics

`--speed` is **data seconds per wall-clock second**. An event whose transaction time is `Δ` seconds after the first
replayed event is published `Δ / speed` wall seconds after the start.

| Value | Meaning |
|---|---|
| `1440` | 1440 data seconds per wall second = 1 day of data per minute |
| `1d/1m` | the same, written as *data duration / wall duration* (units `s m h d`); `1h/10s` = 360 |
| `max` | no pacing: publish as fast as acknowledgements allow (order is still preserved) |

If the publisher cannot keep up, events are published late rather than dropped. The run summary reports `max_lag_s`
(largest delay behind schedule) and `late_events` (events later than 1 s).

## Event contract (`schema_version` 1)

```json
{"event_id": "e857363b37", "client_id": "62d85233f5", "tx_ts": "2018-11-22T04:58:53",
 "purchase_sum": 244.0, "store_id": "05de709fef",
 "items": [{"product_id": "387b2930c5", "quantity": 2.0, "amount": 65.0}],
 "points": {"regular_received": 1.2, "express_received": 0.0, "regular_spent": 0.0, "express_spent": 0.0},
 "schema_version": 1}
```

Message attributes: `event_id`, `schema_version`. **Ordering key: `client_id`**, so one client's events arrive in order
(the topic's subscription must also enable message ordering to preserve it on the consumer side).

* **One event per `(client_id, transaction_id)`.** `purchases` has one row per product; `purchase_sum` and the points
  columns repeat on every row of a transaction. They are published **once** (checked to be identical across the rows;
  summing them would overstate spend about tenfold, see `eda/FINDINGS.md`).
* Ids are strings, leading zeros preserved. Values are published as they are in the source: no cleaning.
* `items` are sorted by `(product_id, quantity, amount)`. The total order of events is
  `(timestamp, client_id, transaction_id)`, independent of the input file order.
* A transaction that cannot become a valid event is **not published**; it is counted and written to the rejects file.
  Reasons: missing client/transaction id, missing or invalid timestamp, fields that differ between its rows, missing
  or non-finite (NaN/inf) numbers. `0` transactions were rejected on the real data (see below).

### Decisions that deviate from, or go beyond, the contract

1. **Event ids (`--event-id-collisions`).** The contract says `event_id` = `transaction_id`, but 28 transaction ids are
   shared by two different clients (56 events). Publishing them unchanged would make a consumer that deduplicates by
   `event_id` silently drop 28 real transactions. Default `disambiguate`: for those ids the event id is
   `<transaction_id>-<client_id>` (both members, deterministic, independent of any filter). `keep` publishes the
   contract value unchanged (duplicates reported in the summary); `error` refuses to start. **The consumer/sink owner
   should confirm this.**
2. **`points.*_spent` sign.** In the source, `regular_points_spent` and `express_points_spent` are zero or negative
   (504,597 of 8,045,229 events have a negative value). They are published as stored (`replay.event.points_spent_sign:
   as_source`); `absolute` publishes `abs()`. The contract does not say which is intended.
3. **`items[].amount` mapping is an assumption.** The work plan does not say which source column feeds `amount`. The
   only per-row amount-like columns are `trn_sum_from_iss` (used) and `trn_sum_from_red`. Set
   `replay.event.item_amount_column` to change it. Please get a decision from the feature/consumer owner; it is not
   verified against any specification.

## Delivery semantics

* **At-least-once, ordered per client.** A message counts as sent only when Pub/Sub acknowledges its future. After a
  crash an event that was published but not yet acknowledged may be sent again after `--resume`; `event_id` is stable,
  so consumers deduplicate on it. This is not exactly-once.
* **Backpressure.** At most `replay.max_outstanding` (5000) unacknowledged messages; the client library additionally
  blocks `publish()` at `batching.flow_control_messages` / `flow_control_bytes`.
* **Failures.** The first failed or timed-out (`ack_timeout_s`) message stops the run (exit 4). Acknowledgements that
  arrive after it are not counted in the checkpoint.
* **Checkpoint** (`streaming/producer/checkpoints/replay_<stream>_<destination>.json`, git-ignored) holds the last
  position whose predecessors were all acknowledged, plus the event key to detect a changed stream. It is written
  every `checkpoint_every` acknowledgements and at the end; on by default when publishing, off for `--dry-run` unless
  `--checkpoint` or `--resume` is given. `--resume` continues after it (no checkpoint = start at the beginning; a
  completed stream resumes to nothing); `--reset-checkpoint` deletes it. Every combination of input file, filters
  (`--start/--end/--clients-sample`), event options and destination has its own checkpoint.
* The ordered events are cached in `streaming/producer/outputs/replay_events_<fingerprint>.duckdb`, keyed by the input
  file (size, mtime) and all event options. It is rebuilt automatically when any of them changes.

## Expected runtime (measured)

Measured on the full dataset (8,045,229 events from 45,786,568 product rows, 400,162 clients, data from
2018-11-21T21:02 to 2019-03-18T23:40, about 117.1 days) on a MacBook (macOS, DuckDB limited to 6 GB / 6 threads, plain
`purchases.csv` input), **`--dry-run --speed max` to a local file**:

* building the ordered event cache: about 48 s (one-off per input and options);
* publishing 8,045,229 events to the file: 247 s (about 32,600 events/s); whole command 295 s, peak memory about 4.5 GB RSS;
* output validated: every line meets the contract, event ids unique, no time-order violation, item rows = 45,786,568.

A paced real-data check, `--speed 1h/10s` over 2019-01-01 10:00–12:00 (5,217 events), took 20.1 s (expected 20 s) with
`max_lag_s` = 0.01.

The wall time of a paced full replay is, by construction, the data span divided by the speed: **at `1d/1m` about 117
minutes** (not measured end to end; the pacing check above covers a 2-hour slice). The average event rate at `1d/1m` is
about 1,150 events/s (8,045,229 events over about 7,000 wall seconds); the evening peaks are higher. **Throughput to
real Pub/Sub or the emulator has not been measured** (no credentials, no emulator; see below).

## Verification status

| Check | Status |
|---|---|
| Unit and integration tests with fake clock/publisher/client (`tests/x5/test_x5_replay.py`) | passed (see the final report for the count) |
| JSONL dry run on a real 1% sample | done |
| JSONL dry run on the full dataset, incl. validation of every event | done (numbers above) |
| Paced run on real data (file target) | done for a 2-hour slice |
| Pub/Sub **emulator** | **not verified**: the Docker daemon could not be started here and `gcloud` is not installed. `test_emulator_roundtrip_preserves_per_client_order` runs automatically when `PUBSUB_EMULATOR_HOST` is set |
| Real Pub/Sub | **not verified**: no project, topic or credentials were available |

Dead letters: this producer has no dead-letter topic. Rejected transactions go to a local file. Nothing here says
anything about the dead-letter count of a downstream consumer.

## Files

* `replay.py`: CLI, settings, `Replayer` (pacing, backpressure, ack tracking), checkpoints, run summary
* `events.py`: pure event assembly/validation, speed parsing (importable by the consumer and the sink)
* `event_source.py`: DuckDB grouping, ordering, rejects and the resumable event cache
* `publishers.py`: Pub/Sub publisher (ordering, batching, flow control) and the JSON-lines publisher
* `requirements.txt`, `../../configs/x5/replay.yaml`, `../../tests/x5/test_x5_replay.py`

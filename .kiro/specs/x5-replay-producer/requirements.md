# Requirements Document

## Introduction

X5 has no live feed, so a producer **simulates** one: it reads `purchases`, builds one event per transaction and publishes the events in time order to the Pub/Sub topic `x5-transactions` at an accelerated clock. A JSON-lines dry run uses the same code without cloud. Tool: `streaming/producer/replay.py`. Linear 298-??.

## Glossary

- **Producer**: `streaming/producer/replay.py`.
- **Event**: the JSON message of schema_version 1 (see steering `structure.md`).
- **Speed**: data seconds per wall-clock second (`1440` = `1d/1m` = one data day per minute; `max` = no pacing).
- **Ordering_Key**: `client_id`.
- **Checkpoint**: file holding the last position whose predecessors were all acknowledged.
- **Reject**: a transaction that cannot become a valid Event.
- **Emulator**: the Pub/Sub emulator, selected by `PUBSUB_EMULATOR_HOST`.

## Requirements

### Requirement 1: Events

#### Acceptance Criteria

1. THE Producer SHALL emit one Event per `(client_id, transaction_id)` with exactly the contract fields in contract order.
2. THE Producer SHALL publish `purchase_sum` and the points values once per transaction (never summed over product rows) and SHALL check that they are identical across the rows of a transaction.
3. THE Producer SHALL keep ids as strings with leading zeros and SHALL publish source values without cleaning.
4. IF a transaction has a missing or non-finite value, an invalid timestamp, or fields that differ between its rows, THEN THE Producer SHALL NOT publish it, SHALL count it and SHALL list it with its reason in a rejects file.
5. THE Producer SHALL emit JSON that contains no NaN or Infinity.

### Requirement 2: Order and event ids

#### Acceptance Criteria

1. THE Producer SHALL publish Events in ascending `(tx_ts, client_id, transaction_id)` order independent of the input file order, with items sorted by `(product_id, quantity, amount)`.
2. WHERE a `transaction_id` is shared by several clients, THE Producer SHALL, in the default `disambiguate` mode, publish `event_id = <transaction_id>-<client_id>` for every member of that group; `keep` SHALL publish the contract value and report duplicates; `error` SHALL refuse to start.
3. THE event id of a shared transaction SHALL NOT depend on `--start`, `--end` or `--clients-sample`.

### Requirement 3: Pacing and filters

#### Acceptance Criteria

1. THE Producer SHALL publish an Event `delta / Speed` wall seconds after the first replayed Event, where `delta` is the data-time difference, and SHALL report `max_lag_s` and the number of events later than 1 s.
2. THE Producer SHALL accept `--speed` as a number, a ratio (`1d/1m`, `1h/10s`) or `max`; an invalid value SHALL exit 2.
3. THE Producer SHALL accept `--start`, `--end`, `--clients-sample` and `--max-events`; a date-only `--end` SHALL include that whole day.

### Requirement 4: Delivery

#### Acceptance Criteria

1. THE Producer SHALL set the Ordering_Key to `client_id`, enable message ordering, batch messages and use blocking flow control.
2. THE Producer SHALL bound the number of unacknowledged messages.
3. IF a message fails or its acknowledgement times out, THEN THE Producer SHALL stop, SHALL NOT count later acknowledgements in the Checkpoint and SHALL exit 4.
4. WHEN run with `--resume`, THE Producer SHALL continue after the Checkpoint; delivery SHALL be at-least-once with stable event ids, and the README SHALL say it is not exactly-once.
5. IF a Checkpoint does not match the stream, THEN THE Producer SHALL refuse to resume and exit 2.

### Requirement 5: Safety and configuration

#### Acceptance Criteria

1. THE Producer SHALL take project, topic, speed, paths and batching from `configs/x5/replay.yaml`; precedence CLI flag, then environment, then YAML.
2. THE Producer SHALL use Application Default Credentials only, SHALL NOT embed credentials, and SHALL fail with a clear message when none exist.
3. THE Producer SHALL create a topic only against the Emulator and SHALL refuse otherwise.
4. WHEN `--dry-run` is given, THE Producer SHALL write JSON lines and SHALL NOT contact Pub/Sub.
5. THE Producer SHALL write a run summary with counts, lag, throughput and exit code.

### Requirement 6: Documentation

#### Acceptance Criteria

1. THE README SHALL state the exact `--speed` semantics, expected runtime using measured values only, delivery semantics, the event-id deviation, the points-spent sign, the `items.amount` assumption, and which checks (local, emulator, cloud) were done.

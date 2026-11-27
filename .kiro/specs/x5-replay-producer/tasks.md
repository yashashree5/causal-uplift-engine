# Implementation Plan: replay producer

## Tasks

- [x] 1. Pure event assembly, validation and speed parsing (`streaming/producer/events.py`)
  - _Requirements: 1.1-1.3, 1.5, 2.2, 3.2_
- [x] 2. Ordered event source in DuckDB with rejects and a resumable cache (`event_source.py`)
  - _Requirements: 1.2, 1.4, 2.1-2.3_
- [x] 3. Publishers: Pub/Sub (ordering, batching, flow control) and JSON lines (`publishers.py`)
  - _Requirements: 4.1, 5.2-5.4_
- [x] 4. Replayer loop, checkpoints, summary and CLI (`replay.py`, `configs/x5/replay.yaml`)
  - _Requirements: 3.1-3.3, 4.2-4.5, 5.1, 5.5_
- [x] 5. Tests (`tests/x5/test_x5_replay.py`)
- [x] 6. README (`streaming/producer/README.md`) and pinned requirements
  - _Requirements: 6.1_
- [x] 7. Real-data dry run on the 1% sample and on the full data, with validation of every event; paced slice
  - _Requirements: 1.1-1.5, 2.1, 3.1_
- [ ] 8. Run against the Pub/Sub emulator (`PUBSUB_EMULATOR_HOST`), including `test_emulator_roundtrip_preserves_per_client_order`
  - Blocked: no running Docker daemon or `gcloud` emulator on the dev machine. See spec `x5-cloud-verification`.
- [ ] 9. Demo replay into the real topic with the sink and consumer; confirm nothing is dead-lettered
  - Blocked: needs the project, topic `x5-transactions`, credentials and the sink and consumer from the infrastructure owner.
- [ ] 10. Confirm the three contract decisions (event-id disambiguation, points-spent sign, `items.amount` source)
  - Blocked on the owners of the consumer and the feature library.

## Notes

- Branch: `ameya/x5-replay-producer`, stacked on `ameya/x5-download` only (independent of the EDA chain).

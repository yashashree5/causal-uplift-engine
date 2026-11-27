# Implementation Plan: cloud verification and open decisions

All tasks are blocked by an external dependency. Do not tick one until the evidence exists.

## Tasks

- [ ] 1. Land the five raw files and manifests in the Bucket; confirm counts; repeat to confirm idempotence
  - Blocked: Bucket, project and credentials from the infrastructure owner.
  - _Requirements: 1.1-1.4_
- [ ] 2. Upload `raw_sample/`
  - Blocked: same as 1.
  - _Requirements: 2.1_
- [ ] 3. Emulator round-trip test and a short paced replay
  - Blocked: a running Pub/Sub emulator (Docker daemon or `gcloud`) on the dev machine.
  - _Requirements: 3.1_
- [ ] 4. Cloud demo replay with sink and consumer; read the dead-letter count
  - Blocked: Topic, subscriptions, sink and consumer from the infrastructure owner.
  - _Requirements: 3.2, 3.3_
- [ ] 5. Set the campaign date; re-run the EDA; regenerate `eda/FINDINGS.md`
  - Blocked: the date is not published.
  - _Requirements: 4.1, 4.2_
- [ ] 6. Review of `eda/FINDINGS.md` by the cleaning owner
  - Blocked: a teammate's review.
  - _Requirements: 5.1_
- [ ] 7. Decisions on event id, points-spent sign and `items.amount` source; record them in the contract and README
  - Blocked: consumer and feature-library owners.
  - _Requirements: 5.2, 5.3_
- [ ] 8. Linear issue IDs on branches and PRs
  - Blocked: IDs not provided. Placeholders `TEA-??` are used.
  - _Requirements: 6.1_
- [ ] 9. Commit, push and open draft PRs in merge order
  - Waits for the user's explicit instruction.
  - _Requirements: 6.2_

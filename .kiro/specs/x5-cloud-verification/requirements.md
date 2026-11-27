# Requirements Document

## Introduction

Items of the data-collection workstream that cannot be completed locally because they depend on cloud resources, other owners or facts that are not published. This spec records what "done" means for each, so the work can be finished the moment the dependency exists. Nothing here may be reported as verified before it has actually run. Linear TEA-??.

## Glossary

- **Bucket**: `gs://<project-id>-x5/`, created by the infrastructure owner.
- **Topic**: Pub/Sub topic `x5-transactions` and its subscriptions, created by the infrastructure owner.
- **Cutoff**: the SMS campaign date.

## Requirements

### Requirement 1: Raw files and manifests in the Bucket

#### Acceptance Criteria

1. WHEN the Bucket and Application Default Credentials exist, THE landing run SHALL upload the five files unchanged to `raw/batch/<file>/ingest_date=<date>/` with a manifest next to each.
2. THE uploaded manifest row counts SHALL match the source: about 45.8M purchases, 400k clients, 200k `uplift_train`.
3. THE verification note SHALL state "cloud verified" only for a run that wrote to the real Bucket.
4. WHEN the same run is repeated, THE second run SHALL change nothing.

### Requirement 2: Sample in the Bucket

#### Acceptance Criteria

1. WHEN the Bucket exists, THE sample upload SHALL write `raw_sample/` and its metadata to the Bucket.

### Requirement 3: Producer against Pub/Sub

#### Acceptance Criteria

1. WHEN an Emulator is available, THE emulator round-trip test SHALL pass and a short paced replay SHALL be received in per-client order.
2. WHEN the Topic exists, THE producer SHALL publish a demo replay that the stream sink and consumer receive.
3. THE demo SHALL report the dead-letter count from the subscription; "nothing was dead-lettered" SHALL be stated only if that count was checked.

### Requirement 4: Campaign date and leakage

#### Acceptance Criteria

1. WHEN the Cutoff is published, THE team SHALL set `campaign.cutoff`, re-run the EDA and regenerate `eda/FINDINGS.md`.
2. THE leakage check SHALL then report passed or failed, with the number of purchases after the Cutoff.

### Requirement 5: Review and contract decisions

#### Acceptance Criteria

1. THE cleaning owner SHALL review `eda/FINDINGS.md`; until then the document SHALL say it is not reviewed.
2. THE owners of the consumer and the feature library SHALL decide: (a) the event id of the 28 shared transaction ids, (b) whether points spent are published as stored or as absolute values, (c) the source column of `items.amount`.
3. EACH decision SHALL be recorded in the shared contract and in `streaming/producer/README.md`.

### Requirement 6: Process

#### Acceptance Criteria

1. EACH branch and pull request SHALL carry its Linear issue ID.
2. THE stacked branches SHALL be committed, pushed and opened as draft pull requests only when the user asks, in the documented merge order.

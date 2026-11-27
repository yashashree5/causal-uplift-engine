#!/usr/bin/env python3
"""Replay X5 purchases as a simulated live event stream into Pub/Sub (or a JSON-lines file).

X5 is a finished 2019 experiment: there is no live feed. This producer *simulates* one. It reads ``purchases``,
builds one event per (client_id, transaction_id) in timestamp order and publishes it at an accelerated clock.

    python streaming/producer/replay.py --dry-run --max-events 1000          # no cloud, writes JSON lines
    python streaming/producer/replay.py --speed 1d/1m --start 2019-01-01 --end 2019-01-07 --project <id>
    PUBSUB_EMULATOR_HOST=localhost:8085 python streaming/producer/replay.py --project x5-local --create-topic-in-emulator

``--speed`` is data seconds per wall-clock second (``1440`` = ``1d/1m`` = one day of data per minute; ``max`` = no
pacing). See ``streaming/producer/README.md`` for flags, delivery guarantees and expected runtime.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingestion.x5_common import (  # noqa: E402
    ConfigError,
    atomic_write_json,
    config_get,
    load_config,
    read_json,
    resolve_path,
)
from streaming.producer import event_source as src  # noqa: E402
from streaming.producer.events import (  # noqa: E402
    EVENT_ID_COLLISION_MODES,
    POINTS_SIGN_MODES,
    SCHEMA_VERSION,
    EventError,
    SpeedError,
    assemble_event,
    encode_event,
    parse_speed,
    parse_when,
    wall_offset,
)
from streaming.producer.publishers import (  # noqa: E402
    EMULATOR_ENV,
    BatchConfig,
    JsonlPublisher,
    Publisher,
    PublisherSetupError,
    PubSubPublisher,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REJECTS = 3
EXIT_PUBLISH_FAILED = 4
EXIT_CLOUD_SETUP = 5
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "x5" / "replay.yaml"
DEFAULT_EMULATOR_PROJECT = "x5-local"


class PublishFailed(RuntimeError):
    """A message was not acknowledged; the checkpoint holds the last contiguously acknowledged event."""


class OrderError(RuntimeError):
    """The event source is not in time order (a bug or a corrupted work database)."""


# --------------------------------------------------------------------------- settings


@dataclasses.dataclass(frozen=True)
class ReplaySettings:
    source: src.SourceSettings
    speed: float
    outputs_dir: Path
    checkpoint_dir: Path
    checkpoint_every: int
    progress_every: int
    max_outstanding: int
    ack_timeout_s: float
    batch: BatchConfig
    schema_version: int
    points_spent_sign: str
    project: str | None
    topic: str

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        input_dir: str | Path | None = None,
        speed: Any = None,
        start: str | None = None,
        end: str | None = None,
        clients_sample: str | Path | None = None,
        project: str | None = None,
        topic: str | None = None,
        event_id_collisions: str | None = None,
        dry_run: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ReplaySettings:
        """Precedence: CLI argument > environment (``GOOGLE_CLOUD_PROJECT``) > YAML > default."""
        env = env or {}
        section = config.get("replay")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration has no 'replay' section (use configs/x5/replay.yaml)")
        event = section.get("event") or {}
        batching = section.get("batching") or {}
        try:
            chosen_speed = parse_speed(
                speed if speed is not None else ("max" if dry_run else section.get("speed", "1d/1m"))
            )
            collisions = event_id_collisions or str(event.get("event_id_collisions", "disambiguate"))
            sign = str(event.get("points_spent_sign", "as_source"))
            if sign not in POINTS_SIGN_MODES:
                raise ConfigError(f"replay.event.points_spent_sign must be one of {POINTS_SIGN_MODES}")
            if collisions not in EVENT_ID_COLLISION_MODES:
                raise ConfigError(f"event id collision mode must be one of {EVENT_ID_COLLISION_MODES}")
            schema_version = int(event.get("schema_version", SCHEMA_VERSION))
            if schema_version != SCHEMA_VERSION:
                raise ConfigError(
                    f"this producer emits schema_version {SCHEMA_VERSION}, config asks for {schema_version}"
                )
            outputs = resolve_path(section.get("outputs_dir", "streaming/producer/outputs"))
            source = src.SourceSettings(
                input_dir=resolve_path(
                    input_dir
                    or env.get("X5_REPLAY_INPUT_DIR")
                    or section.get("input_dir", "data/x5/downloads")
                ),
                work_dir=outputs,
                item_amount_column=str(event.get("item_amount_column", "trn_sum_from_iss")),
                event_id_collisions=collisions,
                start=parse_when(start) if start else None,
                end=parse_when(end, end_of_day=True) if end else None,
                clients_sample=resolve_path(clients_sample) if clients_sample else None,
                duckdb_settings=config.get("duckdb") or {},
            )
            settings = cls(
                source=source,
                speed=chosen_speed,
                outputs_dir=outputs,
                checkpoint_dir=resolve_path(section.get("checkpoint_dir", "streaming/producer/checkpoints")),
                checkpoint_every=int(section.get("checkpoint_every", 10000)),
                progress_every=int(section.get("progress_every", 100000)),
                max_outstanding=int(section.get("max_outstanding", 5000)),
                ack_timeout_s=float(section.get("ack_timeout_s", 120)),
                batch=BatchConfig(
                    max_messages=int(batching.get("max_messages", 100)),
                    max_bytes=int(batching.get("max_bytes", 1_000_000)),
                    max_latency_s=float(batching.get("max_latency_s", 0.05)),
                    flow_control_messages=int(batching.get("flow_control_messages", 10_000)),
                    flow_control_bytes=int(batching.get("flow_control_bytes", 50_000_000)),
                ),
                schema_version=schema_version,
                points_spent_sign=sign,
                project=project or env.get("GOOGLE_CLOUD_PROJECT") or config_get(config, "gcp.project"),
                topic=topic or str(config_get(config, "gcp.pubsub_topic", "x5-transactions")),
            )
        except (SpeedError, ValueError) as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(str(exc)) from exc
        if settings.max_outstanding < 1 or settings.checkpoint_every < 1 or settings.progress_every < 1:
            raise ConfigError("replay.max_outstanding, checkpoint_every and progress_every must be >= 1")
        return settings


# --------------------------------------------------------------------------- checkpoint


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "target"


class Checkpoint:
    """Last contiguously acknowledged position of one (stream, destination) pair."""

    def __init__(self, path: Path, fingerprint: str, target: str) -> None:
        self.path, self.fingerprint, self.target = path, fingerprint, target

    @classmethod
    def for_stream(cls, directory: Path, fingerprint: str, target: str) -> Checkpoint:
        return cls(directory / f"replay_{fingerprint[:16]}_{_slug(target)}.json", fingerprint, target)

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        data = read_json(self.path)
        if data.get("stream_fingerprint") != self.fingerprint or data.get("target") != self.target:
            raise ConfigError(
                f"checkpoint {self.path} belongs to a different stream or destination; use --reset-checkpoint"
            )
        return data

    def save(self, seq: int, key: list[str], acked_total: int, *, completed: bool = False) -> None:
        atomic_write_json(
            self.path,
            {
                "stream_fingerprint": self.fingerprint,
                "target": self.target,
                "position_seq": seq,
                "event_key": key,
                "events_acked_total": acked_total,
                "completed": completed,
                "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    def reset(self) -> bool:
        if self.path.exists():
            self.path.unlink()
            return True
        return False


# --------------------------------------------------------------------------- replay loop


@dataclasses.dataclass
class ReplayItem:
    seq: int
    tx_ts: datetime
    client_id: str
    event_id: str
    data: bytes
    key: list[str]


@dataclasses.dataclass
class RunStats:
    published: int = 0
    acked: int = 0
    failed: int = 0
    bytes_published: int = 0
    max_lag_s: float = 0.0
    late_events: int = 0
    first_tx_ts: str | None = None
    last_tx_ts: str | None = None
    acked_through_seq: int = 0
    wall_seconds: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Replayer:
    """Publish items in order, paced by ``speed``, with bounded outstanding messages and ack tracking.

    Delivery is at-least-once: an event published but not acknowledged when the process dies may be sent again after
    a resume. Event ids are stable, so a consumer can deduplicate.
    """

    LATE_THRESHOLD_S = 1.0

    def __init__(
        self,
        publisher: Publisher,
        *,
        speed: float,
        max_outstanding: int,
        ack_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_checkpoint: Callable[[int, list[str], int], None] | None = None,
        checkpoint_every: int = 10000,
        progress: Callable[[RunStats], None] | None = None,
        progress_every: int = 100000,
        max_events: int | None = None,
    ) -> None:
        self.publisher, self.speed = publisher, speed
        self.max_outstanding, self.ack_timeout_s = max_outstanding, ack_timeout_s
        self.clock, self.sleep = clock, sleep
        self.on_checkpoint, self.checkpoint_every = on_checkpoint, checkpoint_every
        self.progress, self.progress_every = progress, progress_every
        self.max_events = max_events
        self.stats = RunStats()
        self._outstanding: collections.deque[tuple[ReplayItem, Any]] = collections.deque()
        self._since_checkpoint = 0
        self._acked_key: list[str] | None = None
        self._failure: BaseException | None = None
        self._acked_base = 0

    # -- acks
    def _settle(self, item: ReplayItem, error: BaseException | None) -> None:
        if error is not None:
            if self._failure is None:
                self._failure = error
                self.stats.error = f"event {item.event_id} (seq {item.seq}): {error!r}"
            self.stats.failed += 1
            return
        if self._failure is not None:
            return  # acknowledged after an earlier failure: not part of the contiguous prefix
        self.stats.acked += 1
        self.stats.acked_through_seq = item.seq
        self._acked_key = item.key
        self._since_checkpoint += 1
        if self._since_checkpoint >= self.checkpoint_every:
            self._checkpoint()

    def _checkpoint(self) -> None:
        if self.on_checkpoint and self._acked_key is not None:
            self.on_checkpoint(self.stats.acked_through_seq, self._acked_key, self.stats.acked)
        self._since_checkpoint = 0

    def _reap(self) -> None:
        """Settle every already-finished head of the queue (never blocks)."""
        while self._outstanding and self._outstanding[0][1].done():
            item, future = self._outstanding.popleft()
            self._settle(item, future.exception(timeout=0))

    def _wait_head(self) -> None:
        item, future = self._outstanding.popleft()
        try:
            error = future.exception(timeout=self.ack_timeout_s)
        except Exception as exc:  # timeout while waiting for the acknowledgement
            error = exc
        self._settle(item, error)

    # -- main loop
    def run(self, items: Iterable[ReplayItem]) -> RunStats:
        started = self.clock()
        first_data: datetime | None = None
        wall0 = started
        previous: datetime | None = None
        try:
            for item in items:
                if self.max_events is not None and self.stats.published >= self.max_events:
                    break
                if self._failure is not None:
                    break
                if previous is not None and item.tx_ts < previous:
                    raise OrderError(
                        f"event {item.event_id} at {item.tx_ts} precedes the previous event at {previous}"
                    )
                previous = item.tx_ts
                if first_data is None:
                    first_data, wall0 = item.tx_ts, self.clock()
                    self.stats.first_tx_ts = item.tx_ts.isoformat()
                target = wall0 + wall_offset(item.tx_ts, first_data, self.speed)
                self._pace(target)
                lag = max(0.0, self.clock() - target) if not math.isinf(self.speed) else 0.0
                self.stats.max_lag_s = max(self.stats.max_lag_s, lag)
                if lag > self.LATE_THRESHOLD_S:
                    self.stats.late_events += 1
                while len(self._outstanding) >= self.max_outstanding and self._failure is None:
                    self._wait_head()  # backpressure: wait for the oldest acknowledgement
                if self._failure is not None:
                    break
                future = self.publisher.publish(
                    item.data,
                    item.client_id,
                    {"event_id": item.event_id, "schema_version": str(SCHEMA_VERSION)},
                )
                self._outstanding.append((item, future))
                self.stats.published += 1
                self.stats.bytes_published += len(item.data)
                self.stats.last_tx_ts = item.tx_ts.isoformat()
                self._reap()
                if self.progress and self.stats.published % self.progress_every == 0:
                    self.progress(self.stats)
        finally:
            self._drain()
            self.stats.wall_seconds = self.clock() - started
        if self._failure is not None:
            raise PublishFailed(self.stats.error or "publish failed")
        return self.stats

    def _pace(self, target: float) -> None:
        if math.isinf(self.speed):
            return
        while True:
            remaining = target - self.clock()
            if remaining <= 0:
                return
            self._reap()
            self.sleep(min(remaining, 0.5))

    def _drain(self) -> None:
        try:
            self.publisher.flush()
        finally:
            while self._outstanding:
                self._wait_head()
            self._checkpoint()


# --------------------------------------------------------------------------- orchestration


def build_items(
    rows: Iterable[Mapping[str, Any]], settings: ReplaySettings, rejected: list[dict[str, Any]]
) -> Iterable[ReplayItem]:
    for row in rows:
        try:
            event = assemble_event(
                {**row, "tx_ts": row["tx_ts"]},
                schema_version=settings.schema_version,
                points_spent_sign=settings.points_spent_sign,
            )
            data = encode_event(event)
        except (EventError, ValueError) as exc:
            rejected.append(
                {"client_id": row["client_id"], "transaction_id": row["transaction_id"], "reason": str(exc)}
            )
            continue
        yield ReplayItem(row["seq"], row["tx_ts"], row["client_id"], event["event_id"], data, src.key_of(row))


def make_publisher(
    settings: ReplaySettings, args: argparse.Namespace, env: Mapping[str, str], log: Callable[[str], None]
) -> Publisher:
    if args.dry_run:
        target = args.output or str(settings.outputs_dir / "events_dry_run.jsonl")
        log(f"dry run: writing events to {target if target != '-' else 'stdout'}; nothing is sent to Pub/Sub")
        return JsonlPublisher(target, append=args.resume)
    emulator = env.get(EMULATOR_ENV)
    project = settings.project or (DEFAULT_EMULATOR_PROJECT if emulator else None)
    if not project:
        raise PublisherSetupError(
            "no project: pass --project, set GOOGLE_CLOUD_PROJECT or gcp.project (or use --dry-run)"
        )
    publisher = PubSubPublisher(project, settings.topic, settings.batch)
    log(f"target: {'EMULATOR ' + emulator if emulator else 'Pub/Sub'} topic {publisher.topic_path}")
    if args.create_topic_in_emulator:
        log(
            "created topic in the emulator"
            if publisher.create_topic()
            else "topic already exists in the emulator"
        )
    publisher.check_topic()
    return publisher


def target_label(args: argparse.Namespace, settings: ReplaySettings, env: Mapping[str, str]) -> str:
    if args.dry_run:
        return "dry-run"
    project = settings.project or (DEFAULT_EMULATOR_PROJECT if env.get(EMULATOR_ENV) else "unknown-project")
    return f"{'emulator-' if env.get(EMULATOR_ENV) else ''}{project}-{settings.topic}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--input-dir",
        help="directory with purchases.csv[.gz] (env X5_REPLAY_INPUT_DIR; YAML replay.input_dir)",
    )
    parser.add_argument(
        "--speed",
        help="data seconds per wall second: a number (1440), a ratio (1d/1m, 1h/10s) or max (no pacing). "
        "Default: replay.speed (1d/1m); --dry-run defaults to max",
    )
    parser.add_argument(
        "--start", help="first transaction time to replay, inclusive (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end", help="last day (YYYY-MM-DD, whole day included) or exclusive end time (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--clients-sample",
        help="raw_sample/ directory (selected_client_ids.csv or clients.csv[.gz]) or a CSV of client_id",
    )
    parser.add_argument("--max-events", type=int, help="stop after this many events")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the events as JSON lines instead of publishing"
    )
    parser.add_argument(
        "--output", help="dry-run file ('-' = stdout); default <outputs_dir>/events_dry_run.jsonl"
    )
    parser.add_argument("--project", help="GCP project id (env GOOGLE_CLOUD_PROJECT; YAML gcp.project)")
    parser.add_argument("--topic", help="Pub/Sub topic id (YAML gcp.pubsub_topic)")
    parser.add_argument("--emulator-host", help=f"host:port of the Pub/Sub emulator (sets {EMULATOR_ENV})")
    parser.add_argument(
        "--create-topic-in-emulator", action="store_true", help="create the topic (emulator only)"
    )
    parser.add_argument(
        "--event-id-collisions", choices=EVENT_ID_COLLISION_MODES, help="shared transaction_id handling"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue after the last acknowledged event of the checkpoint"
    )
    parser.add_argument(
        "--reset-checkpoint", action="store_true", help="delete the checkpoint before starting"
    )
    parser.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write checkpoints (default: on when publishing, off for --dry-run)",
    )
    parser.add_argument("--prepare-only", action="store_true", help="build the ordered event table and exit")
    parser.add_argument(
        "--force-prepare", action="store_true", help="rebuild the event table even if it is cached"
    )
    parser.add_argument("--run-id", help="label for the run summary file (default: UTC timestamp)")
    return parser


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    publisher: Publisher | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ if env is None else env)
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    if args.emulator_host:
        env[EMULATOR_ENV] = args.emulator_host
        os.environ[EMULATOR_ENV] = args.emulator_host
    try:
        config = load_config(args.config)
        settings = ReplaySettings.from_config(
            config,
            input_dir=args.input_dir,
            speed=args.speed,
            start=args.start,
            end=args.end,
            clients_sample=args.clients_sample,
            project=args.project,
            topic=args.topic,
            event_id_collisions=args.event_id_collisions,
            dry_run=args.dry_run,
            env=env,
        )
        if args.max_events is not None and args.max_events < 1:
            raise ConfigError("--max-events must be >= 1")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        prepared = src.prepare_stream(settings.source, force=args.force_prepare, log=log)
    except (src.SourceError, ConfigError) as exc:
        print(f"cannot build the event stream: {exc}", file=sys.stderr)
        return EXIT_FAILED
    stats = prepared.stats
    log(
        f"stream: {stats['events']:,} events, {stats['clients']:,} clients, {stats['first_tx_ts']} to {stats['last_tx_ts']}; "
        f"{stats['rejected_transactions']:,} transactions rejected; "
        f"{stats['transaction_ids_shared_by_several_clients']:,} transaction ids shared by several clients "
        f"(event_id policy: {stats['event_id_collisions']})"
    )
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    rejects_path = settings.outputs_dir / f"rejects_{prepared.fingerprint[:16]}.jsonl"
    n_rejected_prepare = (
        src.export_rejects(prepared.db_path, settings.source, rejects_path)
        if stats["rejected_transactions"]
        else 0
    )
    if n_rejected_prepare:
        log(f"{n_rejected_prepare:,} rejected transactions are listed in {rejects_path}")
    if args.prepare_only:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return EXIT_OK

    use_checkpoint = args.checkpoint if args.checkpoint is not None else (not args.dry_run)
    use_checkpoint = use_checkpoint or args.resume
    checkpoint = Checkpoint.for_stream(
        settings.checkpoint_dir, prepared.fingerprint, target_label(args, settings, env)
    )
    if args.reset_checkpoint and checkpoint.reset():
        log(f"deleted checkpoint {checkpoint.path}")
    after_seq, acked_before = 0, 0
    if args.resume:
        try:
            saved = checkpoint.load()
        except ConfigError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if saved:
            row = src.event_at(prepared.db_path, settings.source, int(saved["position_seq"]))
            if row is None or [row[0].isoformat(), row[1], row[2]] != saved["event_key"]:
                print(
                    f"checkpoint {checkpoint.path} does not match the event stream; use --reset-checkpoint",
                    file=sys.stderr,
                )
                return EXIT_USAGE
            after_seq, acked_before = int(saved["position_seq"]), int(saved.get("events_acked_total", 0))
            log(f"resuming after seq {after_seq:,} ({acked_before:,} events were acknowledged before)")
        else:
            log("no checkpoint found: starting from the beginning")

    own_publisher = publisher is None
    try:
        publisher = publisher or make_publisher(settings, args, env, log)
    except PublisherSetupError as exc:
        print(f"cannot set up the publisher: {exc}", file=sys.stderr)
        return EXIT_CLOUD_SETUP

    rejected_runtime: list[dict[str, Any]] = []
    saved_position = {"seq": after_seq, "key": None, "acked": acked_before}

    def on_checkpoint(seq: int, key: list[str], acked: int) -> None:
        saved_position.update(seq=seq, key=key, acked=acked_before + acked)
        if use_checkpoint:
            checkpoint.save(seq, key, acked_before + acked)

    def progress(s: RunStats) -> None:
        log(
            f"published {s.published:,} acked {s.acked:,} failed {s.failed} data time {s.last_tx_ts} max lag {s.max_lag_s:.2f}s"
        )

    replayer = Replayer(
        publisher,
        speed=settings.speed,
        max_outstanding=settings.max_outstanding,
        ack_timeout_s=settings.ack_timeout_s,
        clock=clock,
        sleep=sleep,
        on_checkpoint=on_checkpoint,
        checkpoint_every=settings.checkpoint_every,
        progress=progress,
        progress_every=settings.progress_every,
        max_events=args.max_events,
    )
    exit_code = EXIT_OK
    error: str | None = None
    try:
        rows = src.iter_events(prepared.db_path, settings.source, after_seq=after_seq, limit=None)
        try:
            run_stats = replayer.run(build_items(rows, settings, rejected_runtime))
        finally:
            rows.close()  # release the DuckDB connection even when the replay stops early
        if use_checkpoint and args.max_events is None and run_stats.failed == 0:
            last = saved_position["key"]
            if last is not None:
                checkpoint.save(saved_position["seq"], last, saved_position["acked"], completed=True)
    except PublishFailed as exc:
        error, exit_code = str(exc), EXIT_PUBLISH_FAILED
        print(
            f"publishing failed: {exc}. Re-run with --resume to continue after the last acknowledged event.",
            file=sys.stderr,
        )
    except OrderError as exc:
        error, exit_code = str(exc), EXIT_FAILED
        print(f"internal error: {exc}", file=sys.stderr)
    finally:
        if own_publisher:
            publisher.close()
    run_stats = replayer.stats
    n_rejected = n_rejected_prepare + len(rejected_runtime)
    if rejected_runtime:
        with rejects_path.open("a", encoding="utf-8") as handle:
            for entry in rejected_runtime:
                handle.write(json.dumps({**entry, "stage": "assembly"}, sort_keys=True) + "\n")
    if exit_code == EXIT_OK and n_rejected:
        exit_code = EXIT_REJECTS
    summary = {
        "run_id": run_id,
        "mode": "dry-run" if args.dry_run else ("emulator" if env.get(EMULATOR_ENV) else "pubsub"),
        "topic": None if args.dry_run else settings.topic,
        "project": None
        if args.dry_run
        else (settings.project or (DEFAULT_EMULATOR_PROJECT if env.get(EMULATOR_ENV) else None)),
        "speed_data_seconds_per_wall_second": None if math.isinf(settings.speed) else settings.speed,
        "paced": not math.isinf(settings.speed),
        "resumed_after_seq": after_seq,
        "stream": stats,
        "run": run_stats.as_dict(),
        "rejected_transactions_total": n_rejected,
        "events_per_second": (run_stats.published / run_stats.wall_seconds)
        if run_stats.wall_seconds > 0
        else None,
        "exit_code": exit_code,
        "error": error,
        "delivery": "at-least-once; event ids are stable, consumers deduplicate by event_id",
    }
    summary_path = settings.outputs_dir / f"run_summary_{_slug(run_id)}.json"
    atomic_write_json(summary_path, summary)
    log(
        f"done: published {run_stats.published:,}, acknowledged {run_stats.acked:,}, failed {run_stats.failed}, "
        f"{n_rejected:,} rejected, max lag {run_stats.max_lag_s:.2f}s in {run_stats.wall_seconds:.1f}s; summary {summary_path}"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

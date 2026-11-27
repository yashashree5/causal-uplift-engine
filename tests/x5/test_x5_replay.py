"""Tests for the X5 replay producer (streaming/producer/).

All local: a hand-made purchases file whose events are derived by hand below, fake clocks, fake publishers and a fake
Pub/Sub client. The emulator test at the bottom runs only when PUBSUB_EMULATOR_HOST is set.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from streaming.producer import event_source as src
from streaming.producer import replay
from streaming.producer.events import (
    EventError,
    SpeedError,
    assemble_event,
    decode_event,
    encode_event,
    event_id_for,
    parse_duration,
    parse_speed,
    parse_when,
    validate_event,
    wall_offset,
)
from streaming.producer.publishers import (
    BatchConfig,
    JsonlPublisher,
    PublisherSetupError,
    PubSubPublisher,
)

COLUMNS = [
    "client_id",
    "transaction_id",
    "transaction_datetime",
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "purchase_sum",
    "store_id",
    "product_id",
    "product_quantity",
    "trn_sum_from_iss",
    "trn_sum_from_red",
]


def row(client, tx, ts, rr, er, rs, es, total, store, product, qty, amount, red="0.0"):
    return [client, tx, ts, rr, er, rs, es, total, store, product, qty, amount, red]


# Hand-made transactions. File order is deliberately NOT time order, and t3 / t4 rows are interleaved.
ROWS = [
    row("0001", "t1", "2019-01-01 10:00:00", "1.5", "0.0", "0.0", "0.0", "30.0", "s1", "p2", "1.0", "10.0"),
    row("0001", "t3", "2019-01-02 09:00:00", "0.0", "0.0", "0.0", "0.0", "50.0", "s1", "p9", "1.0", "20.0"),
    row("0003", "t4", "2019-01-02 08:00:00", "0.1", "0.2", "0.0", "0.0", "7.0", "s2", "p5", "1.0", "7.0"),
    row("0001", "t1", "2019-01-01 10:00:00", "1.5", "0.0", "0.0", "0.0", "30.0", "s1", "p1", "2.0", "20.0"),
    row("0003", "t4", "2019-01-02 08:00:00", "0.1", "0.2", "0.0", "0.0", "7.0", "s2", "p5", "2.0", "0.0"),
    row("0001", "t3", "2019-01-02 09:00:00", "0.0", "0.0", "0.0", "0.0", "50.0", "s1", "p7", "3.0", "30.0"),
    row("0002", "t2", "2019-01-01 10:00:00", "0.0", "0.0", "-5.0", "0.0", "12.5", "s3", "p1", "1.0", "12.5"),
    row("0005", "dup", "2019-01-03 12:05:00", "0.0", "0.0", "0.0", "0.0", "9.0", "s2", "p1", "1.0", "9.0"),
    row("0004", "dup", "2019-01-03 12:00:00", "0.0", "0.0", "0.0", "0.0", "4.0", "s1", "p1", "1.0", "4.0"),
    row("0001", "t5", "2019-01-05 23:59:59", "0.0", "0.0", "0.0", "0.0", "3.0", "s1", "p1", "1.0", "3.0"),
    # rejected: no timestamp / purchase_sum differs between rows / purchase_sum missing / NaN
    row("0006", "bad1", "", "0.0", "0.0", "0.0", "0.0", "1.0", "s1", "p1", "1.0", "1.0"),
    row("0007", "bad2", "2019-01-04 01:00:00", "0.0", "0.0", "0.0", "0.0", "1.0", "s1", "p1", "1.0", "1.0"),
    row("0007", "bad2", "2019-01-04 01:00:00", "0.0", "0.0", "0.0", "0.0", "2.0", "s1", "p2", "1.0", "1.0"),
    row("0008", "bad3", "2019-01-04 02:00:00", "0.0", "0.0", "0.0", "0.0", "", "s1", "p1", "1.0", "1.0"),
    row("0009", "bad4", "2019-01-04 03:00:00", "0.0", "0.0", "0.0", "0.0", "nan", "s1", "p1", "1.0", "1.0"),
]
GOOD_ORDER = [
    ("0001", "t1"),  # 2019-01-01 10:00:00, ties with 0002/t2 and wins by client_id
    ("0002", "t2"),
    ("0003", "t4"),  # 2019-01-02 08:00
    ("0001", "t3"),  # 2019-01-02 09:00
    ("0004", "dup"),  # 2019-01-03 12:00
    ("0005", "dup"),  # 2019-01-03 12:05
    ("0001", "t5"),  # 2019-01-05 23:59:59
]


def write_purchases(directory: Path, rows=ROWS, name="purchases.csv") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def write_config(tmp_path: Path, **replay_overrides) -> Path:
    section = {
        "input_dir": str(tmp_path / "in"),
        "speed": "max",
        "outputs_dir": str(tmp_path / "out"),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "checkpoint_every": 2,
        "progress_every": 1000,
        "max_outstanding": 10,
        "ack_timeout_s": 5,
        "event": {"item_amount_column": "trn_sum_from_iss", "event_id_collisions": "disambiguate"},
    }
    section.update(replay_overrides)
    path = tmp_path / "replay.yaml"
    path.write_text(yaml.safe_dump({"replay": section, "duckdb": {"memory_limit": "1GB", "threads": 2}}))
    return path


@pytest.fixture()
def env_dir(tmp_path):
    write_purchases(tmp_path / "in")
    return tmp_path


class Recorder:
    """Publisher that stores every message; futures complete immediately."""

    def __init__(self):
        self.messages: list[tuple[dict, str, dict]] = []
        self.closed = False

    def publish(self, data, ordering_key, attributes):
        self.messages.append((decode_event(data), ordering_key, dict(attributes)))
        return _Done()

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _Done:
    def done(self):
        return True

    def exception(self, timeout=None):
        return None


def run_main(tmp_path, *args, publisher=None, env=None, clock=None, sleep=None):
    argv = ["--config", str(write_config(tmp_path)), *args]
    kwargs = {}
    if clock:
        kwargs.update(clock=clock, sleep=sleep)
    return replay.main(argv, env if env is not None else {}, publisher=publisher, **kwargs)


def events_from_main(tmp_path, *args) -> list[dict]:
    rec = Recorder()
    code = run_main(tmp_path, *args, publisher=rec)
    assert code in (replay.EXIT_OK, replay.EXIT_REJECTS), code
    return [m[0] for m in rec.messages]


# --------------------------------------------------------------------------- event contract


def test_event_matches_contract_and_derived_values(env_dir):
    events = events_from_main(env_dir)
    assert [(e["client_id"], e["event_id"].split("-")[0]) for e in events] == [(c, t) for c, t in GOOD_ORDER]
    for event in events:
        assert validate_event(event) == []
    first = events[0]
    assert first == {
        "event_id": "t1",
        "client_id": "0001",
        "tx_ts": "2019-01-01T10:00:00",
        "purchase_sum": 30.0,  # repeated on both product rows: published once, never summed (would be 60)
        "store_id": "s1",
        "items": [
            {"product_id": "p1", "quantity": 2.0, "amount": 20.0},
            {"product_id": "p2", "quantity": 1.0, "amount": 10.0},
        ],  # sorted by product id, not file order
        "points": {
            "regular_received": 1.5,
            "express_received": 0.0,
            "regular_spent": 0.0,
            "express_spent": 0.0,
        },
        "schema_version": 1,
    }
    assert list(first) == [
        "event_id",
        "client_id",
        "tx_ts",
        "purchase_sum",
        "store_id",
        "items",
        "points",
        "schema_version",
    ]


def test_leading_zero_ids_are_strings_and_negative_points_pass_through(env_dir):
    events = {e["event_id"]: e for e in events_from_main(env_dir)}
    assert events["t2"]["client_id"] == "0002"
    assert events["t2"]["points"]["regular_spent"] == -5.0  # as_source (documented)


def test_points_spent_absolute_mode(env_dir):
    rec = Recorder()
    cfg = write_config(
        env_dir, event={"points_spent_sign": "absolute", "event_id_collisions": "disambiguate"}
    )
    assert replay.main(["--config", str(cfg)], {}, publisher=rec) == replay.EXIT_REJECTS
    by_id = {m[0]["event_id"]: m[0] for m in rec.messages}
    assert by_id["t2"]["points"]["regular_spent"] == 5.0


def test_transactions_are_grouped_even_when_rows_are_far_apart(env_dir):
    by_id = {e["event_id"]: e for e in events_from_main(env_dir)}
    assert [i["product_id"] for i in by_id["t3"]["items"]] == ["p7", "p9"]
    assert by_id["t3"]["purchase_sum"] == 50.0
    assert len(by_id["t4"]["items"]) == 2


def test_order_is_timestamp_then_client_then_transaction_and_contiguous_seq(env_dir):
    settings = replay.ReplaySettings.from_config(
        yaml.safe_load(write_config(env_dir).read_text()), env={}
    ).source
    prepared = src.prepare_stream(settings, log=lambda m: None)
    rows = list(src.iter_events(prepared.db_path, settings))
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    assert [(r["client_id"], r["transaction_id"]) for r in rows] == GOOD_ORDER
    stamps = [r["tx_ts"] for r in rows]
    assert stamps == sorted(stamps)


def test_shuffled_input_gives_identical_stream(tmp_path):
    import random

    shuffled = list(ROWS)
    random.Random(3).shuffle(shuffled)
    write_purchases(tmp_path / "a" / "in")
    write_purchases(tmp_path / "b" / "in", rows=shuffled)
    a = events_from_main(tmp_path / "a")
    b = events_from_main(tmp_path / "b")
    assert a == b


def test_rejected_transactions_are_reported_not_published(env_dir, capsys):
    rec = Recorder()
    code = run_main(env_dir, publisher=rec)
    assert code == replay.EXIT_REJECTS
    assert len(rec.messages) == len(GOOD_ORDER)
    rejects = [
        json.loads(line) for line in next((env_dir / "out").glob("rejects_*.jsonl")).read_text().splitlines()
    ]
    reasons = {r["transaction_id"]: r["reason"] for r in rejects}
    assert reasons == {
        "bad1": "missing_or_invalid_timestamp",
        "bad2": "inconsistent_or_missing_field_between_rows",
        "bad3": "inconsistent_or_missing_field_between_rows",
        "bad4": "non_finite_value",
    }
    summary = json.loads(next((env_dir / "out").glob("run_summary_*.json")).read_text())
    assert summary["rejected_transactions_total"] == 4
    assert summary["run"]["published"] == 7 and summary["exit_code"] == replay.EXIT_REJECTS


def test_clean_input_exits_zero(tmp_path):
    write_purchases(tmp_path / "in", rows=ROWS[:10])
    assert run_main(tmp_path, publisher=Recorder()) == replay.EXIT_OK


# --------------------------------------------------------------------------- event ids


def test_shared_transaction_ids_are_disambiguated_for_every_member(env_dir):
    ids = [e["event_id"] for e in events_from_main(env_dir)]
    assert "dup-0004" in ids and "dup-0005" in ids and "dup" not in ids
    assert len(ids) == len(set(ids))


def test_keep_mode_publishes_contract_value_and_reports_duplicates(env_dir):
    rec = Recorder()
    code = run_main(env_dir, "--event-id-collisions", "keep", publisher=rec)
    assert code == replay.EXIT_REJECTS
    ids = [m[0]["event_id"] for m in rec.messages]
    assert ids.count("dup") == 2
    summary = json.loads(next((env_dir / "out").glob("run_summary_*.json")).read_text())
    assert summary["stream"]["duplicate_event_ids"] == 1


def test_error_mode_refuses_to_start(env_dir, capsys):
    rec = Recorder()
    assert run_main(env_dir, "--event-id-collisions", "error", publisher=rec) == replay.EXIT_FAILED
    assert rec.messages == []
    assert "more than one client" in capsys.readouterr().err


def test_event_id_for_policy():
    assert event_id_for("t", "c", shared_transaction_id=False, mode="disambiguate") == "t"
    assert event_id_for("t", "c", shared_transaction_id=True, mode="disambiguate") == "t-c"
    assert event_id_for("t", "c", shared_transaction_id=True, mode="keep") == "t"
    with pytest.raises(ValueError):
        event_id_for("t", "c", shared_transaction_id=True, mode="bogus")


# --------------------------------------------------------------------------- filters


def test_start_and_end_filters(env_dir):
    ids = [e["event_id"] for e in events_from_main(env_dir, "--start", "2019-01-02", "--end", "2019-01-03")]
    assert ids == ["t4", "t3", "dup-0004", "dup-0005"]  # --end day is included whole
    ids = [e["event_id"] for e in events_from_main(env_dir, "--end", "2019-01-01T10:00:00")]
    assert ids == []  # exclusive time end
    ids = [e["event_id"] for e in events_from_main(env_dir, "--start", "2019-01-05T23:59:59")]
    assert ids == ["t5"]


def test_filter_does_not_change_event_ids_of_shared_transactions(env_dir):
    ids = [
        e["event_id"]
        for e in events_from_main(env_dir, "--clients-sample", str(_clients_csv(env_dir, ["0004"])))
    ]
    assert ids == ["dup-0004"]  # disambiguated although its twin (0005) is filtered out


def _clients_csv(tmp_path, clients):
    path = tmp_path / "selected_client_ids.csv"
    path.write_text("client_id\n" + "".join(f"{c}\n" for c in clients))
    return path


def test_clients_sample_directory_and_file(env_dir):
    sample = env_dir / "raw_sample"
    sample.mkdir()
    (sample / "selected_client_ids.csv").write_text("client_id\n0003\n0002\n")
    ids = [e["event_id"] for e in events_from_main(env_dir, "--clients-sample", str(sample))]
    assert ids == ["t2", "t4"]
    ids = [
        e["event_id"]
        for e in events_from_main(env_dir, "--clients-sample", str(sample / "selected_client_ids.csv"))
    ]
    assert ids == ["t2", "t4"]


def test_empty_clients_sample_is_an_error(env_dir, capsys):
    empty = _clients_csv(env_dir, [])
    assert run_main(env_dir, "--clients-sample", str(empty), publisher=Recorder()) == replay.EXIT_FAILED


def test_missing_purchases_file_is_a_clear_error(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    assert run_main(tmp_path, publisher=Recorder()) == replay.EXIT_FAILED
    assert "purchases" in capsys.readouterr().err


def test_missing_required_column(tmp_path, capsys):
    path = tmp_path / "in" / "purchases.csv"
    path.parent.mkdir()
    path.write_text("client_id,transaction_id\n1,2\n")
    assert run_main(tmp_path, publisher=Recorder()) == replay.EXIT_FAILED
    assert "lacks required columns" in capsys.readouterr().err


# --------------------------------------------------------------------------- speed and pacing


def test_parse_speed_forms():
    assert parse_speed("max") == math.inf
    assert parse_speed(1440) == 1440.0
    assert parse_speed("1d/1m") == 1440.0
    assert parse_speed("1h/10s") == 360.0
    assert parse_speed("0.5") == 0.5
    assert parse_duration("90s") == 90.0
    for bad in ("0", "-3", "fast", "1x/1m", "1d/0s", True, "nan"):
        with pytest.raises(SpeedError):
            parse_speed(bad)


def test_wall_offset_and_when():
    t0, t1 = datetime(2019, 1, 1), datetime(2019, 1, 2)
    assert wall_offset(t1, t0, 1440.0) == 60.0
    assert wall_offset(t1, t0, math.inf) == 0.0
    assert parse_when("2019-01-02", end_of_day=True) == datetime(2019, 1, 3)
    assert parse_when("2019-01-02T10:30") == datetime(2019, 1, 2, 10, 30)
    with pytest.raises(ValueError):
        parse_when("01/02/2019")


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TimedPublisher:
    def __init__(self, clock: FakeClock, fail_at: int | None = None, hold: bool = False):
        self.clock, self.fail_at, self.hold = clock, fail_at, hold
        self.log: list[tuple[float, str, str]] = []
        self.pending_after_publish: list[int] = []
        self.futures: list[Any] = []

    def publish(self, data, ordering_key, attributes):
        n = len(self.log) + 1
        self.log.append((self.clock.now, ordering_key, attributes["event_id"]))
        error = RuntimeError("boom") if self.fail_at == n else None
        future = ManualFuture(error, done=not self.hold)
        self.futures.append(future)
        self.pending_after_publish.append(sum(not f.is_done for f in self.futures))
        return future

    def flush(self):
        pass

    def close(self):
        pass


class ManualFuture:
    def __init__(self, error=None, done=True):
        self.error, self.is_done, self.waited = error, done, False

    def done(self):
        return self.is_done

    def exception(self, timeout=None):
        if not self.is_done:
            self.waited, self.is_done = True, True  # the replayer blocked until this ack
        return self.error


def make_items(spec):
    """spec: list of (client, 'YYYY-MM-DD HH:MM:SS')."""
    items = []
    for i, (client, ts) in enumerate(spec, 1):
        stamp = datetime.fromisoformat(ts)
        items.append(
            replay.ReplayItem(i, stamp, client, f"e{i}", b"{}", [stamp.isoformat(), client, f"e{i}"])
        )
    return items


def make_replayer(publisher, clock, **kw):
    defaults = dict(speed=86400.0, max_outstanding=10, ack_timeout_s=1.0, clock=clock, sleep=clock.sleep)
    defaults.update(kw)
    return replay.Replayer(publisher, **defaults)


def test_pacing_follows_data_time_scaled_by_speed():
    clock = FakeClock()
    pub = TimedPublisher(clock)
    items = make_items(
        [("a", "2019-01-01 00:00:00"), ("b", "2019-01-01 12:00:00"), ("a", "2019-01-03 00:00:00")]
    )
    stats = make_replayer(pub, clock, speed=86400.0).run(items)  # one data day per wall second
    assert [round(t - 1000.0, 6) for t, _, _ in pub.log] == [0.0, 0.5, 2.0]
    assert stats.published == stats.acked == 3 and stats.max_lag_s == 0.0 and stats.late_events == 0


def test_max_speed_does_not_sleep():
    clock = FakeClock()
    pub = TimedPublisher(clock)
    make_replayer(pub, clock, speed=math.inf).run(
        make_items([("a", "2019-01-01 00:00:00"), ("a", "2019-06-01 00:00:00")])
    )
    assert clock.sleeps == [] and len(pub.log) == 2


def test_late_events_are_measured_not_hidden():
    clock = FakeClock()

    class SlowPublisher(TimedPublisher):
        def publish(self, *a):
            clock.now += 5.0  # the broker is slow: 5 s per publish
            return super().publish(*a)

    pub = SlowPublisher(clock)
    items = make_items(
        [("a", "2019-01-01 00:00:00"), ("a", "2019-01-01 00:00:01"), ("a", "2019-01-01 00:00:02")]
    )
    stats = make_replayer(pub, clock, speed=1.0).run(items)
    assert stats.max_lag_s > 1.0 and stats.late_events >= 1


def test_replayer_rejects_out_of_order_input():
    clock = FakeClock()
    items = make_items([("a", "2019-01-02 00:00:00"), ("a", "2019-01-01 00:00:00")])
    with pytest.raises(replay.OrderError):
        make_replayer(TimedPublisher(clock), clock, speed=math.inf).run(items)


def test_ordering_key_is_client_id_and_event_id_attribute():
    clock = FakeClock()
    pub = TimedPublisher(clock)
    make_replayer(pub, clock, speed=math.inf).run(
        make_items([("c1", "2019-01-01 00:00:00"), ("c2", "2019-01-01 00:00:01")])
    )
    assert [(k, e) for _, k, e in pub.log] == [("c1", "e1"), ("c2", "e2")]


def test_backpressure_bounds_outstanding_messages():
    clock = FakeClock()
    pub = TimedPublisher(clock, hold=True)
    items = make_items([("a", f"2019-01-01 00:00:{i:02d}") for i in range(10)])
    stats = make_replayer(pub, clock, speed=math.inf, max_outstanding=3).run(items)
    assert max(pub.pending_after_publish) <= 3
    assert stats.acked == 10 and sum(f.waited for f in pub.futures) >= 7


def test_failure_stops_publishing_and_checkpoints_the_contiguous_acked_prefix():
    clock = FakeClock()
    pub = TimedPublisher(clock, fail_at=4)
    saved: list[tuple[int, list[str], int]] = []
    items = make_items([("a", f"2019-01-01 00:00:{i:02d}") for i in range(10)])
    replayer = make_replayer(
        pub,
        clock,
        speed=math.inf,
        on_checkpoint=lambda s, k, n: saved.append((s, k, n)),
        checkpoint_every=100,
    )
    with pytest.raises(replay.PublishFailed, match="seq 4"):
        replayer.run(items)
    assert len(pub.log) == 4  # nothing published after the failure was seen
    assert saved[-1][0] == 3 and saved[-1][2] == 3  # events 1-3 acknowledged; 4 failed
    assert replayer.stats.failed == 1


def test_acks_after_a_failure_do_not_advance_the_checkpoint():
    clock = FakeClock()

    class LateFail(TimedPublisher):
        pass

    pub = LateFail(clock, fail_at=2, hold=True)
    saved: list[int] = []
    items = make_items([("a", f"2019-01-01 00:00:{i:02d}") for i in range(3)])
    replayer = make_replayer(
        pub,
        clock,
        speed=math.inf,
        max_outstanding=10,
        on_checkpoint=lambda s, k, n: saved.append(s),
        checkpoint_every=1,
    )
    with pytest.raises(replay.PublishFailed):
        replayer.run(items)
    assert set(saved) == {1}  # event 3 was acknowledged too but follows the failed event 2


def test_ack_timeout_counts_as_failure():
    clock = FakeClock()

    class Hanging(ManualFuture):
        def exception(self, timeout=None):
            raise TimeoutError("no ack")

    class P(TimedPublisher):
        def publish(self, *a):
            super().publish(*a)
            return Hanging(done=False)

    with pytest.raises(replay.PublishFailed, match="TimeoutError"):
        make_replayer(P(clock), clock, speed=math.inf).run(make_items([("a", "2019-01-01 00:00:00")]))


def test_max_events_limits_publication():
    clock = FakeClock()
    pub = TimedPublisher(clock)
    stats = make_replayer(pub, clock, speed=math.inf, max_events=2).run(
        make_items([("a", f"2019-01-01 00:00:{i:02d}") for i in range(5)])
    )
    assert stats.published == 2


def test_progress_callback():
    clock = FakeClock()
    seen: list[int] = []
    make_replayer(
        TimedPublisher(clock),
        clock,
        speed=math.inf,
        progress=lambda s: seen.append(s.published),
        progress_every=2,
    ).run(make_items([("a", f"2019-01-01 00:00:{i:02d}") for i in range(5)]))
    assert seen == [2, 4]


# --------------------------------------------------------------------------- checkpoints and resume


def test_resume_continues_after_the_last_acknowledged_event(env_dir):
    first = Recorder()
    assert run_main(env_dir, "--max-events", "3", publisher=first) == replay.EXIT_REJECTS
    assert len(first.messages) == 3
    second = Recorder()
    assert run_main(env_dir, "--resume", publisher=second) == replay.EXIT_REJECTS
    combined = [m[0]["event_id"] for m in first.messages + second.messages]
    assert combined == ["t1", "t2", "t4", "t3", "dup-0004", "dup-0005", "t5"]
    assert len(combined) == len(set(combined)) == len(GOOD_ORDER)
    # a completed stream resumes to nothing
    third = Recorder()
    run_main(env_dir, "--resume", publisher=third)
    assert third.messages == []
    saved = json.loads(next((env_dir / "ckpt").glob("replay_*.json")).read_text())
    assert saved["completed"] is True and saved["events_acked_total"] == len(GOOD_ORDER)


def test_without_resume_the_stream_restarts(env_dir):
    run_main(env_dir, "--max-events", "2", publisher=Recorder())
    again = Recorder()
    run_main(env_dir, publisher=again)
    assert len(again.messages) == len(GOOD_ORDER)


def test_resume_without_checkpoint_starts_at_the_beginning(env_dir):
    rec = Recorder()
    run_main(env_dir, "--resume", publisher=rec)
    assert len(rec.messages) == len(GOOD_ORDER)


def test_reset_checkpoint(env_dir):
    run_main(env_dir, "--max-events", "2", publisher=Recorder())
    rec = Recorder()
    run_main(env_dir, "--resume", "--reset-checkpoint", publisher=rec)
    assert len(rec.messages) == len(GOOD_ORDER)


def test_checkpoint_of_a_different_stream_position_is_refused(env_dir, capsys):
    run_main(env_dir, "--max-events", "3", publisher=Recorder())
    path = next((env_dir / "ckpt").glob("replay_*.json"))
    data = json.loads(path.read_text())
    data["event_key"][1] = "9999"
    path.write_text(json.dumps(data))
    assert run_main(env_dir, "--resume", publisher=Recorder()) == replay.EXIT_USAGE
    assert "does not match" in capsys.readouterr().err


def test_checkpoints_are_separate_per_stream_filter(env_dir):
    run_main(env_dir, "--max-events", "2", publisher=Recorder())
    rec = Recorder()
    run_main(env_dir, "--resume", "--start", "2019-01-02", publisher=rec)
    assert [m[0]["event_id"] for m in rec.messages][
        0
    ] == "t4"  # other fingerprint: no checkpoint, starts at its beginning


def test_failed_run_exit_code_and_resume_hint(env_dir, capsys):
    clock = FakeClock()
    pub = TimedPublisher(clock, fail_at=3)
    code = run_main(env_dir, publisher=pub)
    assert code == replay.EXIT_PUBLISH_FAILED
    assert "--resume" in capsys.readouterr().err
    saved = json.loads(next((env_dir / "ckpt").glob("replay_*.json")).read_text())
    assert saved["position_seq"] == 2 and saved["completed"] is False
    rest = Recorder()
    run_main(env_dir, "--resume", publisher=rest)
    assert len(rest.messages) == len(GOOD_ORDER) - 2


# --------------------------------------------------------------------------- dry run and CLI


def test_dry_run_writes_valid_json_lines(env_dir):
    out = env_dir / "events.jsonl"
    code = run_main(env_dir, "--dry-run", "--output", str(out))
    assert code == replay.EXIT_REJECTS
    lines = out.read_text().splitlines()
    assert len(lines) == len(GOOD_ORDER)
    for line in lines:
        assert validate_event(json.loads(line)) == []
        assert "NaN" not in line and "Infinity" not in line
    assert not (env_dir / "ckpt").exists()  # dry runs do not checkpoint unless asked
    summary = json.loads(next((env_dir / "out").glob("run_summary_*.json")).read_text())
    assert summary["mode"] == "dry-run" and summary["paced"] is False


def test_dry_run_is_byte_identical_across_runs(env_dir):
    a, b = env_dir / "a.jsonl", env_dir / "b.jsonl"
    run_main(env_dir, "--dry-run", "--output", str(a))
    run_main(env_dir, "--dry-run", "--output", str(b), "--force-prepare")
    assert a.read_bytes() == b.read_bytes()


def test_dry_run_resume_appends(env_dir):
    out = env_dir / "events.jsonl"
    run_main(env_dir, "--dry-run", "--output", str(out), "--max-events", "3", "--checkpoint")
    run_main(env_dir, "--dry-run", "--output", str(out), "--resume")
    assert len(out.read_text().splitlines()) == len(GOOD_ORDER)


def test_dry_run_with_speed_is_paced(env_dir):
    clock = FakeClock()
    code = run_main(
        env_dir,
        "--dry-run",
        "--speed",
        "1d/10s",
        "--output",
        str(env_dir / "p.jsonl"),
        clock=clock,
        sleep=clock.sleep,
    )
    assert code == replay.EXIT_REJECTS
    # first event 2019-01-01 10:00, last 2019-01-05 23:59:59 = 4d 13:59:59 of data at 8640 data s per wall s
    assert sum(clock.sleeps) == pytest.approx((4 * 86400 + 14 * 3600 - 1) / 8640, rel=1e-3)


def test_prepare_only_builds_cache_without_publishing(env_dir, capsys):
    rec = Recorder()
    assert run_main(env_dir, "--prepare-only", publisher=rec) == replay.EXIT_OK
    assert rec.messages == []
    stats = json.loads(capsys.readouterr().out)
    assert stats["events"] == len(GOOD_ORDER) and stats["rejected_transactions"] == 4
    assert stats["transaction_ids_shared_by_several_clients"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--speed", "fast"],
        ["--start", "yesterday"],
        ["--max-events", "0"],
        ["--event-id-collisions", "bogus"],
    ],
)
def test_bad_arguments_exit_2(env_dir, args, capsys):
    try:
        code = run_main(env_dir, *args, publisher=Recorder())
    except SystemExit as exc:  # argparse choices
        code = exc.code
    assert code == replay.EXIT_USAGE


def test_missing_replay_section_is_a_config_error(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("gcp: {}\n")
    assert replay.main(["--config", str(cfg)], {}) == replay.EXIT_USAGE


def test_settings_precedence_cli_over_env_over_yaml(tmp_path):
    config = {"gcp": {"project": "yaml-proj", "pubsub_topic": "yaml-topic"}, "replay": {"speed": "60"}}
    s = replay.ReplaySettings.from_config(config, env={})
    assert (s.project, s.topic, s.speed) == ("yaml-proj", "yaml-topic", 60.0)
    s = replay.ReplaySettings.from_config(config, env={"GOOGLE_CLOUD_PROJECT": "env-proj"})
    assert s.project == "env-proj"
    s = replay.ReplaySettings.from_config(
        config, project="cli-proj", topic="cli-topic", speed="max", env={"GOOGLE_CLOUD_PROJECT": "env-proj"}
    )
    assert (s.project, s.topic, s.speed) == ("cli-proj", "cli-topic", math.inf)
    assert (
        replay.ReplaySettings.from_config(config, dry_run=True, env={}).speed == math.inf
    )  # dry runs are unpaced
    assert replay.ReplaySettings.from_config(config, dry_run=True, speed="30", env={}).speed == 30.0
    assert replay.ReplaySettings.from_config({"replay": {}}, dry_run=True, env={}).speed == math.inf


def test_repo_replay_config_loads_and_has_no_secrets():
    from ingestion.x5_common import load_config

    path = Path(__file__).resolve().parents[2] / "configs" / "x5" / "replay.yaml"
    config = load_config(path)
    settings = replay.ReplaySettings.from_config(config, env={})
    assert settings.speed == 1440.0 and settings.topic
    text = path.read_text().lower()
    assert "private_key" not in text and "password" not in text


# --------------------------------------------------------------------------- publishers


class FakePubSubClient:
    def __init__(self, existing=True):
        self.calls: list[tuple] = []
        self.existing = existing
        self.stopped = False

    def topic_path(self, project, topic):
        return f"projects/{project}/topics/{topic}"

    def publish(self, topic, data, ordering_key="", **attrs):
        self.calls.append((topic, data, ordering_key, attrs))
        return _Done()

    def get_topic(self, request):
        from google.api_core import exceptions

        if not self.existing:
            raise exceptions.NotFound("nope")

    def create_topic(self, request):
        self.calls.append(("create", request))

    def stop(self):
        self.stopped = True


def test_pubsub_publisher_passes_ordering_key_and_attributes():
    client = FakePubSubClient()
    pub = PubSubPublisher("p", "t", BatchConfig(), client=client)
    pub.publish(b"{}", "client-1", {"event_id": "e1", "schema_version": "1"})
    assert client.calls == [
        ("projects/p/topics/t", b"{}", "client-1", {"event_id": "e1", "schema_version": "1"})
    ]
    pub.close()
    assert client.stopped


def test_missing_topic_is_a_clear_setup_error():
    pub = PubSubPublisher("p", "t", BatchConfig(), client=FakePubSubClient(existing=False))
    with pytest.raises(PublisherSetupError, match="does not exist"):
        pub.check_topic()


def test_topic_creation_is_refused_outside_the_emulator(monkeypatch):
    monkeypatch.delenv("PUBSUB_EMULATOR_HOST", raising=False)
    client = FakePubSubClient()
    with pytest.raises(PublisherSetupError, match="refusing"):
        PubSubPublisher("p", "t", BatchConfig(), client=client).create_topic()
    assert not any(c[0] == "create" for c in client.calls)
    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", "localhost:1")
    assert PubSubPublisher("p", "t", BatchConfig(), client=client).create_topic() is True


@pytest.mark.parametrize(
    "bad", [{"max_messages": 0}, {"max_messages": 5000}, {"max_bytes": 20_000_000}, {"max_latency_s": -1}]
)
def test_batch_limits_are_validated(bad):
    with pytest.raises(PublisherSetupError):
        PubSubPublisher("p", "t", BatchConfig(**bad), client=FakePubSubClient())


def test_no_project_and_no_emulator_is_a_setup_error(env_dir, capsys):
    cfg = write_config(env_dir)
    assert replay.main(["--config", str(cfg)], {}) == replay.EXIT_CLOUD_SETUP
    assert "no project" in capsys.readouterr().err


def test_missing_credentials_message(env_dir, monkeypatch, capsys):
    import google.cloud.pubsub_v1 as pubsub_v1
    from google.auth import exceptions as auth_exceptions

    def boom(*args, **kwargs):
        raise auth_exceptions.DefaultCredentialsError("none")

    monkeypatch.setattr(pubsub_v1, "PublisherClient", boom)
    code = replay.main(["--config", str(write_config(env_dir)), "--project", "p"], {})
    assert code == replay.EXIT_CLOUD_SETUP
    assert "Application Default Credentials" in capsys.readouterr().err


def test_jsonl_publisher_append_and_stdout(tmp_path, capsysbinary):
    pub = JsonlPublisher(tmp_path / "x.jsonl")
    pub.publish(b'{"a":1}', "k", {})
    pub.close()
    again = JsonlPublisher(tmp_path / "x.jsonl", append=True)
    again.publish(b'{"a":2}', "k", {})
    again.close()
    assert (tmp_path / "x.jsonl").read_text().splitlines() == ['{"a":1}', '{"a":2}']
    out = JsonlPublisher("-")
    out.publish(b'{"b":1}', "k", {})
    out.close()
    assert capsysbinary.readouterr().out == b'{"b":1}\n'


# --------------------------------------------------------------------------- event assembly


def _base_row(**overrides):
    base = {
        "event_id": "e",
        "client_id": "c",
        "tx_ts": datetime(2019, 1, 1, 1, 2, 3),
        "purchase_sum": 1.0,
        "store_id": "s",
        "regular_points_received": 0.0,
        "express_points_received": 0.0,
        "regular_points_spent": 0.0,
        "express_points_spent": 0.0,
        "items": [{"product_id": "p", "quantity": 1.0, "amount": 1.0}],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "override",
    [
        {"purchase_sum": None},
        {"purchase_sum": float("nan")},
        {"store_id": ""},
        {"items": []},
        {"items": [{"product_id": "p", "quantity": None, "amount": 1.0}]},
        {"items": [{"product_id": "p", "quantity": 1.0, "amount": math.inf}]},
        {"regular_points_received": float("inf")},
    ],
)
def test_assemble_event_rejects_invalid_values(override):
    with pytest.raises(EventError):
        assemble_event(_base_row(**override))


def test_encode_event_is_compact_and_refuses_nan():
    event = assemble_event(_base_row())
    data = encode_event(event)
    assert b" " not in data and decode_event(data) == event and validate_event(event) == []
    event["purchase_sum"] = float("nan")
    with pytest.raises(ValueError):
        encode_event(event)


def test_validate_event_reports_problems():
    event = assemble_event(_base_row())
    assert validate_event({**event, "schema_version": 1}) == []
    bad = dict(event)
    bad["tx_ts"] = "yesterday"
    assert any("tx_ts" in p for p in validate_event(bad))
    assert validate_event({"event_id": "x"})


def test_build_items_reports_rows_that_cannot_be_assembled(env_dir):
    settings = replay.ReplaySettings.from_config(yaml.safe_load(write_config(env_dir).read_text()), env={})
    good = {**_base_row(), "seq": 1, "transaction_id": "t"}
    bad = {**_base_row(store_id=None), "seq": 2, "transaction_id": "u"}
    rejected: list[dict] = []
    items = list(replay.build_items([good, bad], settings, rejected))
    assert [i.event_id for i in items] == ["e"] and rejected[0]["transaction_id"] == "u"


# --------------------------------------------------------------------------- emulator (optional)


@pytest.mark.skipif(
    not os.environ.get("PUBSUB_EMULATOR_HOST"), reason="PUBSUB_EMULATOR_HOST not set (no emulator)"
)
def test_emulator_roundtrip_preserves_per_client_order(env_dir):
    from google.cloud import pubsub_v1

    project, topic, sub = "x5-test", f"x5-test-{int(time.time())}", f"x5-test-sub-{int(time.time())}"
    publisher_client = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = publisher_client.topic_path(project, topic)
    publisher_client.create_topic(request={"name": topic_path})
    sub_path = subscriber.subscription_path(project, sub)
    subscriber.create_subscription(
        request={"name": sub_path, "topic": topic_path, "enable_message_ordering": True}
    )
    code = replay.main(
        ["--config", str(write_config(env_dir)), "--project", project, "--topic", topic, "--run-id", "emu"],
        dict(os.environ),
    )
    assert code == replay.EXIT_REJECTS
    received: list[dict] = []
    deadline = time.time() + 30
    while len(received) < len(GOOD_ORDER) and time.time() < deadline:
        response = subscriber.pull(request={"subscription": sub_path, "max_messages": 100}, timeout=5)
        ack_ids = []
        for message in response.received_messages:
            received.append(decode_event(message.message.data))
            ack_ids.append(message.ack_id)
        if ack_ids:
            subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": ack_ids})
    assert sorted(e["event_id"] for e in received) == sorted(e["event_id"] for e in events_from_main(env_dir))
    client_1 = [e["tx_ts"] for e in received if e["client_id"] == "0001"]
    assert client_1 == sorted(client_1)


def test_cache_is_rebuilt_when_the_client_list_changes_but_not_its_size(env_dir):
    sample = env_dir / "sample.csv"
    sample.write_text("client_id\n0002\n")
    assert [e["event_id"] for e in events_from_main(env_dir, "--clients-sample", str(sample))] == ["t2"]
    sample.write_text("client_id\n0003\n")  # same name, same size, other content
    assert [e["event_id"] for e in events_from_main(env_dir, "--clients-sample", str(sample))] == ["t4"]

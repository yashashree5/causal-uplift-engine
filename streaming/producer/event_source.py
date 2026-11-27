"""Build the time-ordered transaction stream from the ``purchases`` CSV with DuckDB (bounded memory, disk spill).

``purchases`` has one row per product; the replay needs one event per ``(client_id, transaction_id)``, ordered by
timestamp. This module

1. groups the product rows (an external, spilling group-by and sort inside DuckDB),
2. rejects transactions that cannot become a valid event (missing/non-finite values, fields that differ between the
   rows of one transaction) and records why,
3. assigns event ids (``transaction_id`` is shared by two clients for some ids, see ``events.event_id_for``),
4. stores the ordered events in a work database keyed by ``seq`` so a replay can resume at any position.

Total order: ``(timestamp, client_id, transaction_id)`` ascending. Items inside an event are ordered by
``(product_id, quantity, amount)``. Both are independent of the order of the input file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ingestion.x5_common import ConfigError, resolve_path
from streaming.producer.events import EVENT_ID_COLLISION_MODES

STREAM_VERSION = 1
POINTS_COLUMNS = ("regular_points_received", "express_points_received", "regular_points_spent", "express_points_spent")
REQUIRED_PURCHASE_COLUMNS = (
    "client_id",
    "transaction_id",
    "transaction_datetime",
    *POINTS_COLUMNS,
    "purchase_sum",
    "store_id",
    "product_id",
    "product_quantity",
)


class SourceError(RuntimeError):
    """The purchases input cannot be turned into an event stream."""


@dataclasses.dataclass(frozen=True)
class SourceSettings:
    input_dir: Path
    work_dir: Path
    item_amount_column: str = "trn_sum_from_iss"
    event_id_collisions: str = "disambiguate"
    start: datetime | None = None  # inclusive
    end: datetime | None = None  # exclusive
    clients_sample: Path | None = None
    duckdb_settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_id_collisions not in EVENT_ID_COLLISION_MODES:
            raise ConfigError(f"event id collision mode must be one of {EVENT_ID_COLLISION_MODES}")
        if self.start and self.end and self.start >= self.end:
            raise ConfigError(f"--start ({self.start}) must be before --end ({self.end})")
        if not self.item_amount_column.isidentifier():
            raise ConfigError(f"invalid item amount column name: {self.item_amount_column!r}")


@dataclasses.dataclass
class PreparedStream:
    db_path: Path
    fingerprint: str
    stats: dict[str, Any]
    reused: bool


def find_purchases(input_dir: Path) -> Path:
    for name in ("purchases.csv.gz", "purchases.csv"):
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    raise SourceError(f"no purchases.csv.gz or purchases.csv in {input_dir}")


def load_client_ids_path(sample: Path) -> Path:
    """Resolve ``--clients-sample``: a directory (``selected_client_ids.csv`` or ``clients.csv[.gz]``) or a CSV file."""
    if sample.is_dir():
        for name in ("selected_client_ids.csv", "clients.csv.gz", "clients.csv"):
            if (sample / name).is_file():
                return sample / name
        raise SourceError(f"{sample} has no selected_client_ids.csv or clients.csv[.gz]")
    if sample.is_file():
        return sample
    raise SourceError(f"--clients-sample does not exist: {sample}")


def _q(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def fingerprint(settings: SourceSettings, purchases: Path, clients_file: Path | None) -> str:
    material = {
        "stream_version": STREAM_VERSION,
        "purchases": [purchases.name, purchases.stat().st_size, purchases.stat().st_mtime_ns],
        "clients": hashlib.sha256(clients_file.read_bytes()).hexdigest() if clients_file else None,
        "start": settings.start.isoformat() if settings.start else None,
        "end": settings.end.isoformat() if settings.end else None,
        "item_amount_column": settings.item_amount_column,
        "event_id_collisions": settings.event_id_collisions,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def connect(settings: SourceSettings, db_path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=read_only)
    cfg = settings.duckdb_settings
    if cfg.get("memory_limit"):
        con.execute(f"SET memory_limit={_q(cfg['memory_limit'])}")
    if cfg.get("threads"):
        con.execute(f"SET threads={int(cfg['threads'])}")
    temp = resolve_path(cfg["temp_directory"]) if cfg.get("temp_directory") else settings.work_dir / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={_q(temp)}")
    return con


def db_path_for(settings: SourceSettings, fp: str) -> Path:
    return settings.work_dir / f"replay_events_{fp[:16]}.duckdb"


# NOTE: TRY_CAST(NULLIF(x, '')) on purpose: a bare TRY_CAST on a column of a DuckDB CSV scan is pushed into the reader
# and raises on the first unparseable value instead of returning NULL (see eda/README.md).
def _num(column: str) -> str:
    return f"TRY_CAST(NULLIF({column}, '') AS DOUBLE)"


def _agg3(column: str) -> str:
    return f"min({column}) AS {column}_min, max({column}) AS {column}_max, count({column}) AS {column}_n"


def prepare_stream(settings: SourceSettings, *, force: bool = False, log=print) -> PreparedStream:
    """Build (or reuse) the ordered event table for these inputs and filters."""
    purchases = find_purchases(settings.input_dir)
    clients_file = load_client_ids_path(settings.clients_sample) if settings.clients_sample else None
    fp = fingerprint(settings, purchases, clients_file)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_path_for(settings, fp)
    if db_path.exists() and not force:
        con = connect(settings, db_path, read_only=True)
        try:
            row = con.execute("SELECT value FROM stream_meta WHERE key = 'stats'").fetchone()
            if row:
                log(f"reusing prepared event table {db_path.name}")
                return PreparedStream(db_path, fp, json.loads(row[0]), reused=True)
        except duckdb.Error:
            pass
        finally:
            con.close()
    tmp_path = db_path.with_suffix(".building")
    for stale in (tmp_path, Path(str(tmp_path) + ".wal")):
        stale.unlink(missing_ok=True)
    con = connect(settings, tmp_path)
    try:
        stats = _build(con, settings, purchases, clients_file, log)
        con.execute("CREATE TABLE stream_meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        con.execute("INSERT INTO stream_meta VALUES ('stats', ?), ('fingerprint', ?)", [json.dumps(stats), fp])
        con.execute("CHECKPOINT")
    except BaseException:
        con.close()
        tmp_path.unlink(missing_ok=True)
        raise
    con.close()
    db_path.unlink(missing_ok=True)
    tmp_path.replace(db_path)
    return PreparedStream(db_path, fp, stats, reused=False)


def _build(
    con: duckdb.DuckDBPyConnection, s: SourceSettings, purchases: Path, clients_file: Path | None, log
) -> dict[str, Any]:
    con.execute(
        f"CREATE VIEW src_purchases AS SELECT * FROM read_csv({_q(purchases)}, header=true, all_varchar=true, "
        "delim=',', quote='\"', escape='\"')"
    )
    have = {r[0] for r in con.execute("DESCRIBE SELECT * FROM src_purchases").fetchall()}
    missing = [c for c in (*REQUIRED_PURCHASE_COLUMNS, s.item_amount_column) if c not in have]
    if missing:
        raise SourceError(f"{purchases.name} lacks required columns {missing}")
    amount = _num(s.item_amount_column)
    numeric = ", ".join(f"{_num(c)} AS {c}" for c in (*POINTS_COLUMNS, "purchase_sum", "product_quantity"))
    con.execute(
        "CREATE VIEW typed AS SELECT client_id, transaction_id, "
        f"TRY_CAST(NULLIF(transaction_datetime, '') AS TIMESTAMP) AS ts, {numeric}, {amount} AS amount, "
        "store_id, product_id FROM src_purchases"
    )
    log("grouping product rows into transactions (external group-by; this scans purchases once)")
    consts = ["ts", "store_id", "purchase_sum", *POINTS_COLUMNS]
    aggs = ", ".join(_agg3(c) for c in consts)
    con.execute(
        f"CREATE TABLE tx_all AS SELECT client_id, transaction_id, count(*) AS n_rows, {aggs}, "
        "list(struct_pack(product_id := product_id, quantity := product_quantity, amount := amount) "
        "ORDER BY product_id, product_quantity, amount) AS items "
        "FROM typed GROUP BY client_id, transaction_id"
    )
    # transaction_id shared by several clients (computed on ALL data so event ids do not depend on the filters)
    con.execute(
        "CREATE TABLE tx_flagged AS SELECT *, "
        "count(*) OVER (PARTITION BY transaction_id) > 1 AS shared_transaction_id FROM tx_all"
    )
    n_groups, n_shared_ids, n_shared_groups = con.execute(
        "SELECT count(*), count(DISTINCT transaction_id) FILTER (WHERE shared_transaction_id), "
        "count(*) FILTER (WHERE shared_transaction_id) FROM tx_flagged"
    ).fetchone()
    if n_shared_ids and s.event_id_collisions == "error":
        raise SourceError(
            f"{n_shared_ids} transaction_id values are used by more than one client; refusing to publish duplicate "
            "event ids (use --event-id-collisions disambiguate to append the client id to those ids)"
        )

    consistent = " OR ".join(
        f"{c}_n < n_rows OR {c}_min <> {c}_max" for c in ("store_id", "purchase_sum", *POINTS_COLUMNS)
    )
    nonfinite = " OR ".join(f"NOT isfinite({c}_max)" for c in ("purchase_sum", *POINTS_COLUMNS))
    reason = (
        "CASE "
        "WHEN client_id IS NULL OR client_id = '' THEN 'missing_client_id' "
        "WHEN transaction_id IS NULL OR transaction_id = '' THEN 'missing_transaction_id' "
        "WHEN ts_n = 0 OR ts_n < n_rows THEN 'missing_or_invalid_timestamp' "
        "WHEN ts_min <> ts_max THEN 'inconsistent_timestamp' "
        f"WHEN {nonfinite} THEN 'non_finite_value' "
        f"WHEN {consistent} THEN 'inconsistent_or_missing_field_between_rows' "
        "WHEN list_bool_or(list_transform(items, lambda x: x.product_id IS NULL OR x.quantity IS NULL OR x.amount IS NULL "
        "OR NOT isfinite(x.quantity) OR NOT isfinite(x.amount))) THEN 'missing_or_non_finite_item_value' "
        "ELSE NULL END"
    )
    event_id = (
        "CASE WHEN shared_transaction_id AND ? = 'disambiguate' THEN transaction_id || '-' || client_id "
        "ELSE transaction_id END"
    )
    con.execute(f"CREATE TABLE tx_checked AS SELECT *, {reason} AS reject_reason FROM tx_flagged")

    filters, params = ["TRUE"], []
    if s.clients_sample:
        client_file = clients_file
        con.execute(
            f"CREATE TABLE sample_clients AS SELECT DISTINCT client_id FROM read_csv({_q(client_file)}, header=true, "
            "all_varchar=true, delim=',', quote='\"', escape='\"')"
        )
        if not con.execute("SELECT count(*) FROM sample_clients").fetchone()[0]:
            raise SourceError(f"--clients-sample {clients_file} contains no client ids")
        filters.append("client_id IN (SELECT client_id FROM sample_clients)")
    scope_clients = " AND ".join(filters)
    time_filters = []
    if s.start:
        time_filters.append("ts_min >= ?")
        params.append(s.start)
    if s.end:
        time_filters.append("ts_min < ?")
        params.append(s.end)
    in_time = " AND ".join(time_filters) or "TRUE"

    log("sorting events by (timestamp, client_id, transaction_id)")
    con.execute(
        "CREATE TABLE events AS SELECT row_number() OVER (ORDER BY ts_min, client_id, transaction_id) AS seq, "
        f"{event_id} AS event_id, client_id, transaction_id, ts_min AS tx_ts, store_id_max AS store_id, "
        "purchase_sum_max AS purchase_sum, regular_points_received_max AS regular_points_received, "
        "express_points_received_max AS express_points_received, regular_points_spent_max AS regular_points_spent, "
        "express_points_spent_max AS express_points_spent, items, shared_transaction_id "
        f"FROM tx_checked WHERE reject_reason IS NULL AND {scope_clients} AND {in_time} "
        "ORDER BY seq",
        [s.event_id_collisions, *params],
    )
    # rejected transactions inside the client scope (a missing timestamp can't be placed in a time range, so keep it)
    con.execute(
        "CREATE TABLE rejects AS SELECT client_id, transaction_id, n_rows, reject_reason AS reason FROM tx_checked "
        f"WHERE reject_reason IS NOT NULL AND {scope_clients} ORDER BY client_id, transaction_id"
    )
    n_events, first_ts, last_ts, n_rows_events, n_collision_events = con.execute(
        "SELECT count(*), min(tx_ts), max(tx_ts), coalesce(sum(len(items)), 0), "
        "count(*) FILTER (WHERE shared_transaction_id) FROM events"
    ).fetchone()
    n_clients = con.execute("SELECT count(DISTINCT client_id) FROM events").fetchone()[0]
    n_dup_ids = con.execute("SELECT count(*) - count(DISTINCT event_id) FROM events").fetchone()[0]
    n_rejects = con.execute("SELECT count(*) FROM rejects").fetchone()[0]
    reasons = dict(con.execute("SELECT reason, count(*) FROM rejects GROUP BY 1 ORDER BY 1").fetchall())
    total_rows = con.execute("SELECT count(*) FROM src_purchases").fetchone()[0]
    return {
        "stream_version": STREAM_VERSION,
        "purchase_product_rows": total_rows,
        "client_transactions_total": n_groups,
        "transaction_ids_shared_by_several_clients": n_shared_ids,
        "client_transactions_with_shared_transaction_id": n_shared_groups,
        "events": n_events,
        "events_product_rows": int(n_rows_events),
        "events_with_shared_transaction_id": n_collision_events,
        "duplicate_event_ids": n_dup_ids,
        "clients": n_clients,
        "first_tx_ts": first_ts.isoformat() if first_ts else None,
        "last_tx_ts": last_ts.isoformat() if last_ts else None,
        "rejected_transactions": n_rejects,
        "reject_reasons": reasons,
        "event_id_collisions": s.event_id_collisions,
        "item_amount_column": s.item_amount_column,
        "start": s.start.isoformat() if s.start else None,
        "end": s.end.isoformat() if s.end else None,
        "clients_sample": str(clients_file) if clients_file else None,
    }


# --------------------------------------------------------------------------- reading


EVENT_COLUMNS = (
    "seq",
    "event_id",
    "client_id",
    "transaction_id",
    "tx_ts",
    "store_id",
    "purchase_sum",
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "items",
)


def iter_events(
    db_path: Path, settings: SourceSettings, *, after_seq: int = 0, limit: int | None = None, page: int = 20000
) -> Iterator[dict[str, Any]]:
    """Yield event rows (dicts keyed by ``EVENT_COLUMNS``) in stream order with ``seq > after_seq``."""
    con = connect(settings, db_path, read_only=True)
    try:
        query = f"SELECT {', '.join(EVENT_COLUMNS)} FROM events WHERE seq > ? ORDER BY seq"
        params: list[Any] = [after_seq]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor = con.execute(query, params)
        while True:
            rows = cursor.fetchmany(page)
            if not rows:
                return
            for row in rows:
                yield dict(zip(EVENT_COLUMNS, row, strict=True))
    finally:
        con.close()


def export_rejects(db_path: Path, settings: SourceSettings, target: Path) -> int:
    """Write the rejected transactions as JSON lines; returns how many."""
    con = connect(settings, db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT client_id, transaction_id, n_rows, reason FROM rejects ORDER BY client_id, transaction_id"
        ).fetchall()
    finally:
        con.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for client_id, transaction_id, n_rows, reason in rows:
            handle.write(
                json.dumps(
                    {
                        "client_id": client_id,
                        "transaction_id": transaction_id,
                        "product_rows": n_rows,
                        "reason": reason,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(rows)


def key_of(row: Mapping[str, Any]) -> list[str]:
    """Stable identity of an event's position, stored in checkpoints to detect a changed stream."""
    return [row["tx_ts"].isoformat(), row["client_id"], row["transaction_id"]]


def event_at(db_path: Path, settings: SourceSettings, seq: int) -> Sequence[Any] | None:
    con = connect(settings, db_path, read_only=True)
    try:
        return con.execute("SELECT tx_ts, client_id, transaction_id FROM events WHERE seq = ?", [seq]).fetchone()
    finally:
        con.close()

"""Event assembly for the X5 replay producer: one message per (client_id, transaction_id).

Pure functions only (no I/O, no clock, no network) so the consumer, the sink and the tests can import them. The
event layout is the shared contract of the work plan ("Stream event"), ``schema_version`` 1::

    {"event_id": "<transaction_id>", "client_id": "...", "tx_ts": "2019-01-15T18:42:00",
     "purchase_sum": 512.3, "store_id": "...",
     "items": [{"product_id": "...", "quantity": 1.0, "amount": 99.0}],
     "points": {"regular_received": 5.1, "express_received": 0.0, "regular_spent": 0.0, "express_spent": 0.0},
     "schema_version": 1}

Values are the source values, unchanged: the producer does not clean data (cleaning is done by the silver layer).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

SCHEMA_VERSION = 1
EVENT_FIELDS = ("event_id", "client_id", "tx_ts", "purchase_sum", "store_id", "items", "points", "schema_version")
ITEM_FIELDS = ("product_id", "quantity", "amount")
POINTS_FIELDS = ("regular_received", "express_received", "regular_spent", "express_spent")
POINTS_SIGN_MODES = ("as_source", "absolute")
EVENT_ID_COLLISION_MODES = ("disambiguate", "keep", "error")

_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([smhd])\s*$")


class EventError(ValueError):
    """A row cannot be turned into a valid event."""


class SpeedError(ValueError):
    """The ``--speed`` value cannot be parsed."""


# --------------------------------------------------------------------------- speed and time


def parse_duration(text: str) -> float:
    """``'1d'`` -> 86400.0 seconds. Units: s, m, h, d."""
    match = _DURATION_RE.match(str(text).lower())
    if not match:
        raise SpeedError(f"not a duration (use e.g. 30s, 5m, 2h, 1d): {text!r}")
    seconds = float(match.group(1)) * _DURATION_UNITS[match.group(2)]
    if seconds <= 0:
        raise SpeedError(f"duration must be positive: {text!r}")
    return seconds


def parse_speed(value: Any) -> float:
    """Return the replay speed as *data seconds per wall-clock second* (``math.inf`` = no pacing).

    Accepted forms:

    * ``max`` / ``inf``: no pacing, publish as fast as the broker accepts (order is still preserved);
    * a number: data seconds per wall second, e.g. ``1440`` = 1 day of data per minute;
    * ``<data duration>/<wall duration>``, e.g. ``1d/1m`` (= 1440), ``1h/10s`` (= 360).
    """
    if isinstance(value, bool):
        raise SpeedError(f"invalid speed: {value!r}")
    if isinstance(value, (int, float)):
        speed = float(value)
    else:
        text = str(value).strip().lower()
        if text in {"max", "inf", "infinity"}:
            return math.inf
        if "/" in text:
            data, wall = text.split("/", 1)
            speed = parse_duration(data) / parse_duration(wall)
        else:
            try:
                speed = float(text)
            except ValueError as exc:
                raise SpeedError(
                    f"invalid speed {value!r}: use max, a number (data s per wall s) or e.g. 1d/1m"
                ) from exc
    if not (speed > 0) or math.isnan(speed):
        raise SpeedError(f"speed must be positive: {value!r}")
    return speed


def parse_when(text: str, *, end_of_day: bool = False) -> datetime:
    """Parse ``YYYY-MM-DD`` or ``YYYY-MM-DD[ T]HH:MM[:SS]`` into a naive datetime."""
    cleaned = str(text).strip().replace("T", " ")
    for pattern, date_only in (("%Y-%m-%d %H:%M:%S", False), ("%Y-%m-%d %H:%M", False), ("%Y-%m-%d", True)):
        try:
            parsed = datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
        return parsed + timedelta(days=1) if (date_only and end_of_day) else parsed
    raise ValueError(f"not a date or datetime (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS): {text!r}")


def wall_offset(data_time: datetime, first_data_time: datetime, speed: float) -> float:
    """Wall-clock seconds after the start at which ``data_time`` is due (0 for unpaced replay)."""
    if math.isinf(speed):
        return 0.0
    return max(0.0, (data_time - first_data_time).total_seconds() / speed)


# --------------------------------------------------------------------------- event ids


def event_id_for(transaction_id: str, client_id: str, *, shared_transaction_id: bool, mode: str) -> str:
    """The event id: the contract value ``transaction_id`` unless it is shared by several clients.

    ``transaction_id`` is not unique across clients in the real data. ``disambiguate`` (default) appends the client id
    for every member of a shared group, deterministically and independent of file order; ``keep`` publishes the
    contract value as is (duplicate ids, a consumer deduplicating by event_id would drop events).
    """
    if mode not in EVENT_ID_COLLISION_MODES:
        raise ValueError(f"event id collision mode must be one of {EVENT_ID_COLLISION_MODES}, got {mode!r}")
    if shared_transaction_id and mode == "disambiguate":
        return f"{transaction_id}-{client_id}"
    return transaction_id


# --------------------------------------------------------------------------- assembly


def _finite(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise EventError(f"{name} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise EventError(f"{name} is not finite ({value!r})")
    return number


def _text(value: Any, name: str) -> str:
    if value is None or str(value) == "":
        raise EventError(f"{name} is missing")
    return str(value)


def assemble_event(
    row: Mapping[str, Any], *, schema_version: int = SCHEMA_VERSION, points_spent_sign: str = "as_source"
) -> dict[str, Any]:
    """Build one contract event from an aggregated transaction row.

    ``row`` keys: ``event_id``, ``client_id``, ``tx_ts`` (datetime or ISO text), ``purchase_sum``, ``store_id``,
    ``items`` (list of mappings with ``product_id``, ``quantity``, ``amount``) and the four points values
    ``regular_points_received``, ``express_points_received``, ``regular_points_spent``, ``express_points_spent``.
    Raises :class:`EventError` for missing or non-finite values (JSON cannot represent NaN/Infinity).
    """
    if points_spent_sign not in POINTS_SIGN_MODES:
        raise ValueError(f"points_spent_sign must be one of {POINTS_SIGN_MODES}")
    ts = row["tx_ts"]
    tx_ts = ts.isoformat() if isinstance(ts, datetime) else _text(ts, "tx_ts")
    items = row.get("items")
    if not items:
        raise EventError("transaction has no items")

    def spent(name: str) -> float:
        value = _finite(row[name], name)
        return abs(value) if points_spent_sign == "absolute" else value

    return {
        "event_id": _text(row["event_id"], "event_id"),
        "client_id": _text(row["client_id"], "client_id"),
        "tx_ts": tx_ts,
        "purchase_sum": _finite(row["purchase_sum"], "purchase_sum"),
        "store_id": _text(row["store_id"], "store_id"),
        "items": [
            {
                "product_id": _text(item["product_id"], "item.product_id"),
                "quantity": _finite(item["quantity"], "item.quantity"),
                "amount": _finite(item["amount"], "item.amount"),
            }
            for item in items
        ],
        "points": {
            "regular_received": _finite(row["regular_points_received"], "regular_points_received"),
            "express_received": _finite(row["express_points_received"], "express_points_received"),
            "regular_spent": spent("regular_points_spent"),
            "express_spent": spent("express_points_spent"),
        },
        "schema_version": int(schema_version),
    }


def encode_event(event: Mapping[str, Any]) -> bytes:
    """Compact UTF-8 JSON in contract field order. ``allow_nan=False`` makes NaN/Infinity an error, never a message."""
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode_event(data: bytes | str) -> dict[str, Any]:
    return json.loads(data)


def validate_event(event: Mapping[str, Any]) -> list[str]:
    """Return the list of contract violations (empty = valid). Used by tests and by consumers."""
    problems: list[str] = []
    if list(event.keys()) != list(EVENT_FIELDS):
        problems.append(f"fields must be exactly {list(EVENT_FIELDS)} in order, got {list(event.keys())}")
        return problems
    for name in ("event_id", "client_id", "store_id"):
        if not isinstance(event[name], str) or not event[name]:
            problems.append(f"{name} must be a non-empty string")
    if not isinstance(event["tx_ts"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?", event["tx_ts"]
    ):
        problems.append("tx_ts must be ISO-8601 without zone, e.g. 2019-01-15T18:42:00")
    if not isinstance(event["purchase_sum"], float) or not math.isfinite(event["purchase_sum"]):
        problems.append("purchase_sum must be a finite float")
    items = event["items"]
    if not isinstance(items, Sequence) or isinstance(items, str) or not items:
        problems.append("items must be a non-empty list")
    else:
        for i, item in enumerate(items):
            if list(item.keys()) != list(ITEM_FIELDS):
                problems.append(f"items[{i}] fields must be {list(ITEM_FIELDS)}")
                continue
            if not isinstance(item["product_id"], str) or not item["product_id"]:
                problems.append(f"items[{i}].product_id must be a non-empty string")
            for name in ("quantity", "amount"):
                if not isinstance(item[name], float) or not math.isfinite(item[name]):
                    problems.append(f"items[{i}].{name} must be a finite float")
    points = event["points"]
    if not isinstance(points, Mapping) or list(points.keys()) != list(POINTS_FIELDS):
        problems.append(f"points fields must be {list(POINTS_FIELDS)}")
    else:
        for name in POINTS_FIELDS:
            if not isinstance(points[name], float) or not math.isfinite(points[name]):
                problems.append(f"points.{name} must be a finite float")
    if event["schema_version"] != SCHEMA_VERSION or isinstance(event["schema_version"], bool):
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    return problems

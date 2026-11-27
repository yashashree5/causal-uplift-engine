"""Reusable analysis code behind ``eda/x5_eda.ipynb`` and ``eda/run_eda.py``.

All heavy work is done by DuckDB inside a bounded-memory, disk-backed work database
(``<output_dir>/work.duckdb``). The five CSV tables are read as text first and cast with ``TRY_CAST(NULLIF(x, ''))``;
the number of values that fail to cast is recorded rather than hidden. IDs are never cast, so leading
zeros survive.

Results are collected in :class:`Results` (scalars per section plus named tables) and written to
``<output_dir>`` as ``summary.json``, ``tables/*.csv`` and ``figures/*.png``. Nothing here fabricates
numbers: every value is computed from the input files of the run and the run is labelled with its mode
(``sample`` or ``full``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import duckdb
import pandas as pd

from ingestion.sample_clients import find_source
from ingestion.x5_common import ConfigError, atomic_write_json, atomic_write_text, config_get, resolve_path

EDA_VERSION = 5
MODES = ("sample", "full")
ID_COLUMNS = {
    "clients": ["client_id"],
    "uplift_train": ["client_id"],
    "uplift_test": ["client_id"],
    "products": ["product_id", "level_1", "level_2", "level_3", "level_4", "brand_id", "vendor_id"],
    "purchases": ["client_id", "transaction_id", "store_id", "product_id"],
}
POINTS = (
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
)
NO_PURCHASES = "no_purchases"
INVALID_AGE = "invalid_or_missing"

REQUIRED_COLUMNS = {
    "clients": ["client_id", "first_issue_date", "first_redeem_date", "age", "gender"],
    "uplift_train": ["client_id", "treatment_flg", "target"],
    "uplift_test": ["client_id"],
    "products": ["product_id"],
    "purchases": [
        "client_id",
        "transaction_id",
        "transaction_datetime",
        *POINTS,
        "purchase_sum",
        "store_id",
        "product_id",
        "product_quantity",
        "trn_sum_from_iss",
        "trn_sum_from_red",
    ],
}


class EdaError(RuntimeError):
    """A problem with the inputs or the configuration of an EDA run."""


# --------------------------------------------------------------------------- settings


@dataclasses.dataclass(frozen=True)
class EdaSettings:
    mode: str
    input_dir: Path
    output_dir: Path
    alpha: float
    min_arm_size: int
    min_cell_events: int
    money_tolerance: float
    valid_age_min: int
    valid_age_max: int
    age_band_edges: tuple[int, ...]
    recency_band_edges_days: tuple[int, ...]
    spend_tier_quantiles: tuple[float, ...]
    quantiles: tuple[float, ...]
    cutoff: str | None
    duckdb_settings: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        mode: str,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        cutoff: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> EdaSettings:
        """Resolve settings. Precedence: explicit argument > environment > YAML > default."""
        env = env or {}
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
        section = config.get("eda")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration has no 'eda' section (use configs/x5/eda.yaml)")
        chosen_input = (
            input_dir or env.get("X5_EDA_INPUT_DIR") or config_get(config, f"eda.input_dirs.{mode}")
        )
        if not chosen_input:
            raise ConfigError(f"no input directory for mode '{mode}' (eda.input_dirs.{mode} or --input-dir)")
        root = output_dir or Path(str(config_get(config, "eda.output_root", "eda/outputs"))) / mode
        segments = section.get("segments") or {}
        try:
            settings = cls(
                mode=mode,
                input_dir=resolve_path(chosen_input),
                output_dir=resolve_path(root),
                alpha=float(section.get("alpha", 0.05)),
                min_arm_size=int(section.get("min_arm_size", 200)),
                min_cell_events=int(section.get("min_cell_events", 20)),
                money_tolerance=float(section.get("money_tolerance", 0.01)),
                valid_age_min=int(section.get("valid_age_min", 16)),
                valid_age_max=int(section.get("valid_age_max", 100)),
                age_band_edges=tuple(
                    int(x) for x in segments.get("age_band_edges", [16, 25, 35, 45, 55, 65, 101])
                ),
                recency_band_edges_days=tuple(
                    int(x) for x in segments.get("recency_band_edges_days", [0, 30, 90])
                ),
                spend_tier_quantiles=tuple(
                    float(x) for x in segments.get("spend_tier_quantiles", [0.25, 0.5, 0.75])
                ),
                quantiles=tuple(float(x) for x in section.get("quantiles", [0.01, 0.25, 0.5, 0.75, 0.99])),
                cutoff=cutoff or config_get(config, "campaign.cutoff"),
                duckdb_settings=config.get("duckdb") or {},
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid eda configuration: {exc}") from exc
        if not 0 < settings.alpha < 1:
            raise ConfigError("eda.alpha must be in (0, 1)")
        for name, edges in (
            ("age_band_edges", settings.age_band_edges),
            ("recency_band_edges_days", settings.recency_band_edges_days),
        ):
            if len(edges) < 2 or list(edges) != sorted(set(edges)):
                raise ConfigError(f"eda.segments.{name} must be strictly increasing with at least 2 values")
        if any(not 0 < q < 1 for q in settings.spend_tier_quantiles) or list(
            settings.spend_tier_quantiles
        ) != sorted(set(settings.spend_tier_quantiles)):
            raise ConfigError(
                "eda.segments.spend_tier_quantiles must be strictly increasing values in (0, 1)"
            )
        if settings.cutoff is not None:
            parse_cutoff(settings.cutoff)
        return settings


def parse_cutoff(value: Any) -> str:
    """Return the cutoff as ``YYYY-MM-DD HH:MM:SS``; raise :class:`ConfigError` if it is not a date."""
    text = str(value).strip()
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(text, pattern))
        except ValueError:
            continue
    raise ConfigError(f"campaign cutoff must look like YYYY-MM-DD or YYYY-MM-DD HH:MM:SS, got {value!r}")


# --------------------------------------------------------------------------- results container


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: NaN/inf -> None, numpy/pandas scalars -> Python, tuples -> lists."""
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class Results:
    """Scalars per section plus named tables, with (de)serialisation."""

    def __init__(self) -> None:
        self.sections: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, pd.DataFrame] = {}

    def section(self, name: str) -> dict[str, Any]:
        return self.sections.setdefault(name, {})

    def add_table(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        self.tables[name] = frame
        return frame

    def write(self, output_dir: Path) -> None:
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            target = tables_dir / f"{name}.csv"
            atomic_write_text(target, frame.to_csv(index=False))
        index = {
            name: {"rows": int(len(f)), "columns": list(map(str, f.columns))}
            for name, f in self.tables.items()
        }
        payload = {"sections": _clean(self.sections), "tables": index}
        atomic_write_json(output_dir / "summary.json", payload)

    @classmethod
    def load(cls, output_dir: Path) -> Results:
        path = output_dir / "summary.json"
        if not path.is_file():
            raise EdaError(f"no EDA results at {output_dir} (missing summary.json); run eda/run_eda.py first")
        payload = json.loads(path.read_text())
        results = cls()
        results.sections = payload["sections"]
        for name in payload["tables"]:
            results.tables[name] = pd.read_csv(
                output_dir / "tables" / f"{name}.csv", dtype=str, keep_default_na=False
            )
            results.tables[name] = _restore_numeric(results.tables[name])
        return results


def _restore_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert columns that are entirely numeric back to numbers. ID-like columns stay text."""
    out = frame.copy()
    for column in out.columns:
        if column.endswith("_id") or column in {
            "column",
            "table",
            "segment",
            "segment_kind",
            "stat",
            "label",
            "flag",
        }:
            continue
        converted = pd.to_numeric(out[column], errors="coerce")
        blanks = out[column].isin(["", "nan", "NaN"])
        if bool((converted.notna() | blanks).all()):
            out[column] = converted
    return out


# --------------------------------------------------------------------------- statistics helpers


def uplift_stats(
    n_t: int, y_t: int, n_c: int, y_c: int, *, alpha: float, min_arm_size: int, min_cell_events: int
) -> dict[str, Any]:
    """Difference in outcome rate (treated - control) with a normal-approximation CI and sparse-cell flags."""
    rate_t = y_t / n_t if n_t else math.nan
    rate_c = y_c / n_c if n_c else math.nan
    uplift = rate_t - rate_c if n_t and n_c else math.nan
    variance = (
        rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c if n_t and n_c else math.nan
    )  # unpooled (Wald) variance
    se = math.sqrt(variance) if variance == variance else math.nan
    crit = NormalDist().inv_cdf(1 - alpha / 2)
    z = uplift / se if se and se == se and se > 0 else math.nan
    p_value = math.erfc(abs(z) / math.sqrt(2)) if z == z else math.nan
    cells = (y_t, n_t - y_t, y_c, n_c - y_c)
    sparse = n_t < min_arm_size or n_c < min_arm_size or min(cells) < min_cell_events
    return {
        "n": n_t + n_c,
        "n_treated": n_t,
        "n_control": n_c,
        "conversions_treated": y_t,
        "conversions_control": y_c,
        "rate_treated": rate_t,
        "rate_control": rate_c,
        "uplift": uplift,
        "se": se,
        "ci_low": uplift - crit * se if se == se else math.nan,
        "ci_high": uplift + crit * se if se == se else math.nan,
        "z": z,
        "p_value": p_value,
        "sparse": bool(sparse),
    }


def smd(mean_t: float, var_t: float, mean_c: float, var_c: float) -> float:
    """Standardised mean difference (pooled SD); NaN when undefined."""
    pooled = math.sqrt((var_t + var_c) / 2) if var_t == var_t and var_c == var_c else math.nan
    if not pooled or pooled != pooled:
        return math.nan
    return (mean_t - mean_c) / pooled


def band_labels(edges: Sequence[int], *, open_ended: bool, width: int = 3) -> list[str]:
    """Sortable labels for consecutive edge intervals, e.g. ``[0,15,30]`` -> ``000-014``, ``015-029``."""
    labels = [f"{lo:0{width}d}-{hi - 1:0{width}d}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    if open_ended:
        labels.append(f">={edges[-1]:0{width}d}")
    return labels


def _case_expr(expr: str, edges: Sequence[int], labels: Sequence[str], *, open_ended: bool) -> str:
    parts = []
    for (lo, hi), label in zip(zip(edges[:-1], edges[1:], strict=True), labels, strict=False):
        parts.append(f"WHEN {expr} >= {lo} AND {expr} < {hi} THEN '{label}'")
    if open_ended:
        parts.append(f"WHEN {expr} >= {edges[-1]} THEN '{labels[-1]}'")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def _q(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- context / connection


@dataclasses.dataclass
class Context:
    settings: EdaSettings
    con: duckdb.DuckDBPyConnection
    results: Results
    stages: dict[str, float] = dataclasses.field(default_factory=dict)

    def sql(self, query: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    def one(self, query: str, params: Sequence[Any] | None = None) -> tuple[Any, ...]:
        row = self.con.execute(query, params or []).fetchone()
        return row if row is not None else ()


def connect(settings: EdaSettings) -> duckdb.DuckDBPyConnection:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.output_dir / "work.duckdb"))
    cfg = settings.duckdb_settings
    if cfg.get("memory_limit"):
        con.execute(f"SET memory_limit={_q(cfg['memory_limit'])}")
    if cfg.get("threads"):
        con.execute(f"SET threads={int(cfg['threads'])}")
    temp = (
        resolve_path(cfg["temp_directory"])
        if cfg.get("temp_directory")
        else settings.output_dir / "duckdb_tmp"
    )
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={_q(temp)}")
    return con


# --------------------------------------------------------------------------- inputs / typed tables


def input_files(settings: EdaSettings) -> dict[str, Path]:
    if not settings.input_dir.is_dir():
        raise EdaError(f"input directory does not exist: {settings.input_dir}")
    files = {}
    for table in REQUIRED_COLUMNS:
        try:
            files[table] = find_source(table, settings.input_dir)
        except Exception as exc:  # SamplingError: no file for this table
            raise EdaError(str(exc)) from exc
    return files


def input_fingerprint(settings: EdaSettings, files: Mapping[str, Path]) -> str:
    material = {
        "eda_version": EDA_VERSION,
        "files": {t: [p.name, p.stat().st_size] for t, p in sorted(files.items())},
        "cutoff": settings.cutoff,
        "age": [settings.valid_age_min, settings.valid_age_max],
        "segments": [
            settings.age_band_edges,
            settings.recency_band_edges_days,
            settings.spend_tier_quantiles,
        ],
        "tolerance": settings.money_tolerance,
        "alpha": settings.alpha,
        "min_arm_size": settings.min_arm_size,
        "min_cell_events": settings.min_cell_events,
        "quantiles": settings.quantiles,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=list).encode()).hexdigest()


def _register_raw(ctx: Context, files: Mapping[str, Path]) -> None:
    for table, path in files.items():
        ctx.con.execute(
            f"CREATE OR REPLACE TEMP VIEW src_{table} AS SELECT * FROM read_csv({_q(path)}, header=true, "
            "all_varchar=true, delim=',', quote='\"', escape='\"')"
        )
        have = {r[0] for r in ctx.con.execute(f"DESCRIBE SELECT * FROM src_{table}").fetchall()}
        missing = [c for c in REQUIRED_COLUMNS[table] if c not in have]
        if missing:
            raise EdaError(f"{files[table].name} is missing required columns {missing}; found {sorted(have)}")


_CAST_CHECKS = [
    ("clients", "first_issue_date", "TIMESTAMP"),
    ("clients", "first_redeem_date", "TIMESTAMP"),
    ("clients", "age", "BIGINT"),
    ("uplift_train", "treatment_flg", "BIGINT"),
    ("uplift_train", "target", "BIGINT"),
    ("purchases", "transaction_datetime", "TIMESTAMP"),
    *[("purchases", c, "DOUBLE") for c in (*POINTS, "purchase_sum", "product_quantity")],
    ("purchases", "trn_sum_from_iss", "DOUBLE"),
    ("purchases", "trn_sum_from_red", "DOUBLE"),
    ("products", "netto", "DOUBLE"),
    ("products", "segment_id", "DOUBLE"),
]


# NOTE: ``TRY_CAST(NULLIF(col, ''))`` rather than ``TRY_CAST(col)`` on purpose. A bare TRY_CAST on a column of a
# DuckDB CSV scan is pushed into the reader and raises on the first unparseable value instead of returning NULL;
# NULLIF (a no-op here, empty fields already read as NULL) keeps the cast in the query so bad values are counted.


def prepare(ctx: Context, files: Mapping[str, Path], fingerprint: str, *, force: bool = False) -> bool:
    """Build typed tables in the work DB. Returns True if (re)built, False if the existing ones are current."""
    con = ctx.con
    con.execute("CREATE TABLE IF NOT EXISTS eda_meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    current = con.execute("SELECT value FROM eda_meta WHERE key='fingerprint'").fetchone()
    if current and current[0] == fingerprint and not force:
        return False
    _register_raw(ctx, files)
    for table in (
        "clients_t",
        "uplift_train_t",
        "uplift_test_t",
        "products_t",
        "purchases_t",
        "tx",
        "cx",
        "cast_failures",
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        "CREATE TABLE clients_t AS SELECT client_id, TRY_CAST(NULLIF(first_issue_date, '') AS TIMESTAMP) AS first_issue_ts, "
        "TRY_CAST(NULLIF(first_redeem_date, '') AS TIMESTAMP) AS first_redeem_ts, TRY_CAST(NULLIF(age, '') AS BIGINT) AS age, gender "
        "FROM src_clients"
    )
    con.execute(
        "CREATE TABLE uplift_train_t AS SELECT client_id, TRY_CAST(NULLIF(treatment_flg, '') AS BIGINT) AS treatment_flg, "
        "TRY_CAST(NULLIF(target, '') AS BIGINT) AS target FROM src_uplift_train"
    )
    con.execute("CREATE TABLE uplift_test_t AS SELECT * FROM src_uplift_test")
    con.execute("CREATE TABLE products_t AS SELECT * FROM src_products")
    numeric = ", ".join(
        f"TRY_CAST(NULLIF({c}, '') AS DOUBLE) AS {c}"
        for c in (*POINTS, "purchase_sum", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red")
    )
    con.execute(
        "CREATE TABLE purchases_t AS SELECT client_id, transaction_id, "
        "TRY_CAST(NULLIF(transaction_datetime, '') AS TIMESTAMP) AS ts, "
        f"{numeric}, store_id, product_id FROM src_purchases"
    )
    rows = []
    present = {
        t: {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM src_{t}").fetchall()} for t in REQUIRED_COLUMNS
    }
    for table, column, dtype in _CAST_CHECKS:
        if column not in present[table]:  # optional columns (e.g. products.netto) may be absent
            continue
        n_non_null, n_failed = con.execute(
            f"SELECT count({column}), count(*) FILTER (WHERE {column} IS NOT NULL AND TRY_CAST(NULLIF({column}, '') AS {dtype}) IS NULL) "
            f"FROM src_{table}"
        ).fetchone()
        rows.append((table, column, dtype, n_non_null, n_failed))
    con.execute(
        "CREATE TABLE cast_failures (table_name VARCHAR, column_name VARCHAR, target_type VARCHAR, "
        "n_non_null BIGINT, n_failed BIGINT)"
    )
    con.executemany("INSERT INTO cast_failures VALUES (?, ?, ?, ?, ?)", rows)
    _build_tx(ctx)
    _build_cx(ctx)
    con.execute("INSERT OR REPLACE INTO eda_meta VALUES ('fingerprint', ?)", [fingerprint])
    return True


def _agg3(column: str, alias: str) -> str:
    return f"min({column}) AS {alias}_min, max({column}) AS {alias}_max, count({column}) AS {alias}_n"


def _build_tx(ctx: Context) -> None:
    """One row per (client_id, transaction_id) with min/max/non-null counts of the columns that should be constant.

    transaction_id alone is not unique across clients in the real data (see ``semantics``), so the pair is the key.
    """
    constant = ["client_id", "store_id", "ts", "purchase_sum", *POINTS]
    aggs = ", ".join(_agg3(c, c) for c in constant)
    ctx.con.execute(
        f"CREATE TABLE tx AS SELECT client_id AS client_key, transaction_id, count(*) AS n_rows, {aggs}, "
        "count(DISTINCT product_id) AS n_products, "
        "sum(trn_sum_from_iss) AS sum_iss, count(trn_sum_from_iss) AS n_iss, "
        "sum(trn_sum_from_red) AS sum_red, count(trn_sum_from_red) AS n_red, "
        "sum(product_quantity) AS sum_qty, min(product_quantity) AS min_qty "
        "FROM purchases_t GROUP BY client_id, transaction_id"
    )


def _build_cx(ctx: Context) -> None:
    """One row per client in `clients_t`, with pre-campaign purchase aggregates from transaction level."""
    ctx.con.execute(
        "CREATE TABLE cx AS "
        "WITH agg AS (SELECT client_id_min AS client_id, count(*) AS n_tx, sum(purchase_sum_max) AS total_spend, "
        "min(ts_min) AS first_ts, max(ts_min) AS last_ts, count(DISTINCT store_id_min) AS n_stores "
        "FROM tx GROUP BY client_id_min) "
        "SELECT c.client_id, c.first_issue_ts, c.first_redeem_ts, c.age, c.gender, "
        "coalesce(a.n_tx, 0) AS n_tx, a.total_spend, a.first_ts, a.last_ts, coalesce(a.n_stores, 0) AS n_stores "
        "FROM clients_t c LEFT JOIN agg a USING (client_id)"
    )


# --------------------------------------------------------------------------- sections


def _timed(ctx: Context, name: str, func) -> None:
    started = time.monotonic()
    func(ctx)
    ctx.stages[name] = round(time.monotonic() - started, 3)


def _provenance_summary(input_dir: Path, files: Mapping[str, Path]) -> dict[str, Any]:
    """Link the input files to ``download_provenance.json`` (written by ingestion/download_x5.py) if it is found.

    The match is by file size against the downloaded artifact or its extracted CSV member; checksums are not
    recomputed here (the downloader already verified them).
    """
    for candidate in (input_dir, input_dir.parent / "downloads", input_dir.parent):
        path = candidate / "download_provenance.json"
        if path.is_file():
            break
    else:
        return {"found": False}
    try:
        artifacts = json.loads(path.read_text())["artifacts"]
    except (OSError, ValueError, KeyError):
        return {"found": False, "unreadable": str(path)}
    tables = {}
    revisions = set()
    for table, file in files.items():
        entry = artifacts.get(table)
        if not entry:
            continue
        size = file.stat().st_size
        member_sizes = [m.get("bytes") for m in entry.get("extracted_members", [])]
        source = entry.get("source") or {}
        revisions.add(source.get("revision_resolved"))
        tables[table] = {
            "artifact_sha256": entry.get("sha256"),
            "input_size_matches_artifact": size == entry.get("bytes"),
            "input_size_matches_extracted_member": size in member_sizes,
        }
    return {
        "found": True,
        "file": str(path),
        "revision_resolved": sorted(r for r in revisions if r),
        "repo_id_resolved": sorted(
            {(a.get("source") or {}).get("repo_id_resolved") for a in artifacts.values() if a.get("source")}
            - {None}
        ),
        "tables": tables,
    }


def section_run_info(ctx: Context, files: Mapping[str, Path], fingerprint: str) -> None:
    s = ctx.settings
    info = ctx.results.section("run")
    info.update(
        {
            "eda_version": EDA_VERSION,
            "mode": s.mode,
            "mode_description": (
                "FULL data: every row of the five X5 tables"
                if s.mode == "full"
                else "SAMPLE: client sample of the X5 tables; statistics describe the sample, not the full data"
            ),
            "input_dir": str(s.input_dir),
            "input_files": {t: {"file": p.name, "bytes": p.stat().st_size} for t, p in files.items()},
            "fingerprint": fingerprint,
            "duckdb_version": duckdb.__version__,
            "python_version": platform.python_version(),
            "duckdb_memory_limit": s.duckdb_settings.get("memory_limit"),
            "duckdb_threads": s.duckdb_settings.get("threads"),
            "settings": {
                "alpha": s.alpha,
                "min_arm_size": s.min_arm_size,
                "min_cell_events": s.min_cell_events,
                "money_tolerance": s.money_tolerance,
                "valid_age_range": [s.valid_age_min, s.valid_age_max],
                "age_band_edges": list(s.age_band_edges),
                "recency_band_edges_days": list(s.recency_band_edges_days),
                "spend_tier_quantiles": list(s.spend_tier_quantiles),
                "campaign_cutoff": s.cutoff,
            },
        }
    )
    info["download_provenance"] = _provenance_summary(s.input_dir, files)
    sample_meta = s.input_dir / "sample_metadata.json"
    if sample_meta.is_file():
        meta = json.loads(sample_meta.read_text())
        info["sample_metadata"] = {
            k: meta.get(k)
            for k in ("fraction", "seed", "selected_client_count", "selected_ids_checksum", "method")
        }


def section_files(ctx: Context) -> None:
    """File statistics, nulls, duplicates, key checks and unmatched IDs."""
    out = ctx.results.section("files")
    con = ctx.con
    rows = []
    for table, view in (
        ("clients", "clients_t"),
        ("uplift_train", "uplift_train_t"),
        ("uplift_test", "uplift_test_t"),
        ("products", "products_t"),
        ("purchases", "purchases_t"),
    ):
        columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()]
        exprs = ", ".join(f"count({c})" for c in columns)
        n_rows = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
        non_null = con.execute(f"SELECT {exprs} FROM {view}").fetchone()
        for column, n_nn in zip(columns, non_null, strict=True):
            rows.append({"table": table, "column": column, "n_rows": n_rows, "n_null": n_rows - n_nn})
        out[f"{table}_rows"] = n_rows
        out[f"{table}_columns"] = columns
    nulls = pd.DataFrame(rows)
    nulls["null_share"] = nulls["n_null"] / nulls["n_rows"].where(nulls["n_rows"] > 0)
    ctx.results.add_table("null_counts", nulls)

    cast = ctx.sql(
        'SELECT table_name AS "table", column_name AS "column", target_type, n_non_null, n_failed FROM cast_failures '
        "ORDER BY 1, 2"
    )
    ctx.results.add_table("cast_failures", cast)
    out["cast_failures_total"] = int(cast["n_failed"].sum())

    # ---- key uniqueness
    keys = {}
    for table, view, key in (
        ("clients", "clients_t", "client_id"),
        ("uplift_train", "uplift_train_t", "client_id"),
        ("uplift_test", "uplift_test_t", "client_id"),
        ("products", "products_t", "product_id"),
    ):
        n, distinct = con.execute(f"SELECT count({key}), count(DISTINCT {key}) FROM {view}").fetchone()
        keys[table] = {
            "key": key,
            "rows_with_key": n,
            "distinct_keys": distinct,
            "duplicate_rows": n - distinct,
        }
    out["key_uniqueness"] = keys

    # ---- purchases: duplicates (hash based: a 64-bit hash collision would only under-count)
    n_p, n_hash_all, n_pairs = con.execute(
        "SELECT count(*), count(DISTINCT hash(client_id, transaction_id, ts, regular_points_received, "
        "express_points_received, regular_points_spent, express_points_spent, purchase_sum, store_id, product_id, "
        "product_quantity, trn_sum_from_iss, trn_sum_from_red)), "
        "count(DISTINCT hash(transaction_id, product_id)) FROM purchases_t"
    ).fetchone()
    purchases = {"rows": n_p, "exact_duplicate_rows_hash_based": n_p - n_hash_all}
    purchases["rows_beyond_first_per_transaction_product_pair"] = n_p - n_pairs
    n_tx, n_tx_clients, n_clients, n_products, n_stores = con.execute(
        "SELECT count(DISTINCT transaction_id), count(DISTINCT (client_id, transaction_id)), count(DISTINCT client_id), "
        "count(DISTINCT product_id), count(DISTINCT store_id) FROM purchases_t"
    ).fetchone()
    purchases.update(
        {
            "distinct_transaction_id": n_tx,
            "distinct_client_transaction_pairs": n_tx_clients,
            "distinct_client_id": n_clients,
            "distinct_product_id": n_products,
            "distinct_store_id": n_stores,
        }
    )
    out["purchases"] = purchases

    # ---- cross-table relationships (clients present on one side but not the other)
    rel = {}

    def one(sql: str) -> int:
        return int(con.execute(sql).fetchone()[0])

    rel["train_clients_not_in_clients"] = one(
        "SELECT count(DISTINCT client_id) FROM uplift_train_t WHERE client_id NOT IN (SELECT client_id FROM clients_t)"
    )
    rel["test_clients_not_in_clients"] = one(
        "SELECT count(DISTINCT client_id) FROM uplift_test_t WHERE client_id NOT IN (SELECT client_id FROM clients_t)"
    )
    rel["train_test_overlap_clients"] = one(
        "SELECT count(*) FROM (SELECT DISTINCT client_id FROM uplift_train_t) t "
        "WHERE client_id IN (SELECT client_id FROM uplift_test_t)"
    )
    rel["clients_in_neither_train_nor_test"] = one(
        "SELECT count(*) FROM clients_t WHERE client_id NOT IN (SELECT client_id FROM uplift_train_t) "
        "AND client_id NOT IN (SELECT client_id FROM uplift_test_t)"
    )
    rel["purchase_clients_not_in_clients"] = one(
        "SELECT count(DISTINCT client_id) FROM purchases_t WHERE client_id NOT IN (SELECT client_id FROM clients_t)"
    )
    rel["purchase_rows_with_client_not_in_clients"] = one(
        "SELECT count(*) FROM purchases_t WHERE client_id NOT IN (SELECT client_id FROM clients_t)"
    )
    rel["purchase_product_ids_not_in_products"] = one(
        "SELECT count(DISTINCT product_id) FROM purchases_t WHERE product_id NOT IN (SELECT product_id FROM products_t)"
    )
    rel["purchase_rows_with_product_not_in_products"] = one(
        "SELECT count(*) FROM purchases_t WHERE product_id IS NOT NULL "
        "AND product_id NOT IN (SELECT product_id FROM products_t)"
    )
    rel["purchase_rows_with_null_product_id"] = one(
        "SELECT count(*) FROM purchases_t WHERE product_id IS NULL"
    )
    rel["products_never_purchased"] = one(
        "SELECT count(*) FROM products_t WHERE product_id NOT IN (SELECT DISTINCT product_id FROM purchases_t WHERE product_id IS NOT NULL)"
    )
    out["relationships"] = rel

    # ---- clients without purchases, by population
    frame = ctx.sql(
        "WITH pop AS (SELECT c.client_id, c.n_tx, "
        "CASE WHEN t.client_id IS NOT NULL THEN 'uplift_train' WHEN s.client_id IS NOT NULL THEN 'uplift_test' "
        "ELSE 'neither' END AS population, t.treatment_flg "
        "FROM cx c LEFT JOIN (SELECT DISTINCT client_id, treatment_flg FROM uplift_train_t) t USING (client_id) "
        "LEFT JOIN (SELECT DISTINCT client_id FROM uplift_test_t) s USING (client_id)) "
        "SELECT population, treatment_flg, count(*) AS n_clients, "
        "count(*) FILTER (WHERE n_tx = 0) AS n_without_purchases FROM pop GROUP BY 1, 2 ORDER BY 1, 2"
    )
    frame["share_without_purchases"] = frame["n_without_purchases"] / frame["n_clients"]
    ctx.results.add_table("clients_without_purchases", frame)
    out["clients_without_purchases_total"] = int(frame["n_without_purchases"].sum())


def section_timestamps(ctx: Context) -> None:
    out = ctx.results.section("timestamps")
    con = ctx.con
    first, last, n_null, n = con.execute(
        "SELECT min(ts), max(ts), count(*) FILTER (WHERE ts IS NULL), count(*) FROM purchases_t"
    ).fetchone()
    cast = ctx.results.tables["cast_failures"]
    failed = cast[(cast["table"] == "purchases") & (cast["column"] == "transaction_datetime")]["n_failed"]
    out.update(
        {
            "purchase_ts_min": first,
            "purchase_ts_max": last,
            "purchase_rows": n,
            "purchase_ts_null_or_invalid_rows": n_null,
            "purchase_ts_unparseable_non_null_values": int(failed.iloc[0]) if len(failed) else None,
            "timezone_note": "timestamps are naive local retail time as published; no time zone is given",
        }
    )
    if first is not None:
        out["purchase_span_days"] = (last - first).total_seconds() / 86400
    out["clients_first_issue_ts_min"], out["clients_first_issue_ts_max"] = con.execute(
        "SELECT min(first_issue_ts), max(first_issue_ts) FROM clients_t"
    ).fetchone()
    out["clients_first_redeem_ts_min"], out["clients_first_redeem_ts_max"] = con.execute(
        "SELECT min(first_redeem_ts), max(first_redeem_ts) FROM clients_t"
    ).fetchone()
    (
        out["clients_first_redeem_before_first_issue"],
        out["clients_first_redeem_null"],
        out["clients_first_issue_after_first_purchase"],
        out["clients_first_redeem_after_last_purchase_ts"],
    ) = con.execute(
        "SELECT count(*) FILTER (WHERE first_redeem_ts < first_issue_ts), count(*) FILTER (WHERE first_redeem_ts IS NULL), "
        "count(*) FILTER (WHERE first_ts IS NOT NULL AND first_issue_ts > first_ts), "
        "count(*) FILTER (WHERE first_redeem_ts > (SELECT max(ts) FROM purchases_t)) FROM cx"
    ).fetchone()
    monthly = ctx.sql(
        "SELECT coalesce(strftime(date_trunc('month', ts_min), '%Y-%m'), '<no_timestamp>') AS month, count(*) AS n_transactions, "
        "count(DISTINCT client_id_min) AS n_clients, sum(purchase_sum_max) AS spend, "
        "min(ts_min) AS first_ts, max(ts_min) AS last_ts FROM tx GROUP BY 1 ORDER BY 1"
    )
    ctx.results.add_table("transactions_by_month", monthly)
    weekly = ctx.sql(
        "SELECT CAST(date_trunc('week', ts_min) AS DATE) AS week_start, count(*) AS n_transactions, "
        "count(DISTINCT client_id_min) AS n_clients FROM tx WHERE ts_min IS NOT NULL GROUP BY 1 ORDER BY 1"
    )
    weekly["week_start"] = weekly["week_start"].astype(str)
    ctx.results.add_table("transactions_by_week", weekly)
    hours = ctx.sql(
        "SELECT hour(ts_min) AS hour_of_day, count(*) AS n_transactions FROM tx WHERE ts_min IS NOT NULL GROUP BY 1 ORDER BY 1"
    )
    ctx.results.add_table("transactions_by_hour", hours)
    clients_by_month = ctx.sql(
        "SELECT strftime(date_trunc('month', first_issue_ts), '%Y-%m') AS month, count(*) AS n_clients "
        "FROM clients_t WHERE first_issue_ts IS NOT NULL GROUP BY 1 ORDER BY 1"
    )
    ctx.results.add_table("clients_first_issue_by_month", clients_by_month)


def anchor_for(ctx: Context) -> tuple[str, str]:
    """Recency anchor ``(timestamp_text, label)``: the campaign cutoff if configured, else max purchase ts."""
    if ctx.settings.cutoff:
        return parse_cutoff(ctx.settings.cutoff), "campaign cutoff from configuration"
    value = ctx.con.execute("SELECT max(ts) FROM purchases_t").fetchone()[0]
    if value is None:
        raise EdaError("no valid purchase timestamps: cannot anchor recency")
    return value.strftime("%Y-%m-%d %H:%M:%S"), "PROVISIONAL: max purchase timestamp (campaign date unknown)"


def section_leakage(ctx: Context) -> None:
    """Purchase history vs campaign cutoff. Without a cutoff the check is blocked, not passed."""
    out = ctx.results.section("leakage")
    anchor, label = anchor_for(ctx)
    out["recency_anchor"] = anchor
    out["recency_anchor_label"] = label
    cutoff = ctx.settings.cutoff
    if not cutoff:
        out["status"] = "blocked"
        out["reason"] = (
            "campaign.cutoff is not set: no campaign date is published with the dataset, so purchases cannot be "
            "split into before/after the SMS. Features built from the full purchase history may include post-campaign "
            "behaviour. Set campaign.cutoff once the real date is confirmed."
        )
        return
    cutoff = parse_cutoff(cutoff)
    out["status"] = "checked"
    out["cutoff"] = cutoff
    n_rows, n_post, n_tx, n_tx_post = ctx.one(
        "SELECT (SELECT count(*) FROM purchases_t), (SELECT count(*) FROM purchases_t WHERE ts >= ?::TIMESTAMP), "
        "(SELECT count(*) FROM tx), (SELECT count(*) FROM tx WHERE ts_min >= ?::TIMESTAMP)",
        [cutoff, cutoff],
    )
    out.update(
        {
            "purchase_rows": n_rows,
            "purchase_rows_at_or_after_cutoff": n_post,
            "transactions": n_tx,
            "transactions_at_or_after_cutoff": n_tx_post,
            "clients_first_redeem_at_or_after_cutoff": ctx.one(
                "SELECT count(*) FROM clients_t WHERE first_redeem_ts >= ?::TIMESTAMP", [cutoff]
            )[0],
        }
    )
    out["note"] = "rows at/after the cutoff must be excluded from any pre-campaign feature"


def _describe(ctx: Context, relation: str, expr: str, where: str = "TRUE") -> dict[str, Any]:
    """Quantile summary of a numeric expression (non-null values satisfying ``where``)."""
    qs = list(ctx.settings.quantiles)
    row = ctx.con.execute(
        f"SELECT count(*), count({expr}), avg({expr}), stddev_samp({expr}), min({expr}), max({expr}), "
        f"count(*) FILTER (WHERE {expr} < 0), count(*) FILTER (WHERE {expr} = 0), count(*) FILTER (WHERE {expr} > 0), "
        f"quantile_cont({expr}, {qs}) FROM {relation} WHERE {where}"
    ).fetchone()
    names = ("n_rows", "n_non_null", "mean", "std", "min", "max", "n_negative", "n_zero", "n_positive")
    result = dict(zip(names, row[:-1], strict=True))
    result["quantiles"] = {str(q): v for q, v in zip(qs, row[-1] or [None] * len(qs), strict=True)}
    return result


def _hist(
    ctx: Context,
    relation: str,
    expr: str,
    *,
    width: float,
    where: str = "TRUE",
    lo: float | None = None,
    hi: float | None = None,
) -> pd.DataFrame:
    clauses = [where, f"{expr} IS NOT NULL"]
    if lo is not None:
        clauses.append(f"{expr} >= {lo}")
    if hi is not None:
        clauses.append(f"{expr} < {hi}")
    frame = ctx.sql(
        f"SELECT floor({expr} / {width}) * {width} AS bin_start, count(*) AS n FROM {relation} "
        f"WHERE {' AND '.join(clauses)} GROUP BY 1 ORDER BY 1"
    )
    frame["bin_width"] = width
    return frame


def section_distributions(ctx: Context) -> None:
    out = ctx.results.section("distributions")
    anchor, _ = anchor_for(ctx)
    # anchor is a validated timestamp text produced by anchor_for(), never raw user input
    ctx.con.execute(
        "CREATE OR REPLACE TEMP VIEW cx_feat AS SELECT *, "
        f"CASE WHEN last_ts IS NULL THEN NULL ELSE (epoch(TIMESTAMP '{anchor}') - epoch(last_ts)) / 86400.0 END "
        "AS recency_days, "
        f"CASE WHEN first_issue_ts IS NULL THEN NULL ELSE (epoch(TIMESTAMP '{anchor}') - epoch(first_issue_ts)) / 86400.0 END "
        "AS tenure_days FROM cx"
    )
    out["spend_per_transaction"] = _describe(ctx, "tx", "purchase_sum_max")
    out["product_rows_per_transaction"] = _describe(ctx, "tx", "n_rows")
    out["distinct_products_per_transaction"] = _describe(ctx, "tx", "n_products")
    out["transactions_per_client_with_purchases"] = _describe(ctx, "cx", "n_tx", "n_tx > 0")
    out["total_spend_per_client_with_purchases"] = _describe(ctx, "cx", "total_spend", "n_tx > 0")
    out["stores_per_client_with_purchases"] = _describe(ctx, "cx", "n_stores", "n_stores > 0")
    out["recency_days_with_purchases"] = _describe(ctx, "cx_feat", "recency_days", "n_tx > 0")
    out["tenure_days_at_anchor"] = _describe(ctx, "cx_feat", "tenure_days")
    out["age_all_values"] = _describe(ctx, "clients_t", "age")
    out["age_valid_range"] = _describe(
        ctx, "clients_t", "age", f"age BETWEEN {ctx.settings.valid_age_min} AND {ctx.settings.valid_age_max}"
    )
    out["age_outside_valid_range"] = ctx.one(
        f"SELECT count(*) FILTER (WHERE age < {ctx.settings.valid_age_min}), "
        f"count(*) FILTER (WHERE age > {ctx.settings.valid_age_max}), count(*) FILTER (WHERE age IS NULL) FROM clients_t"
    )
    out["age_outside_valid_range"] = dict(
        zip(("below_min", "above_max", "null"), out["age_outside_valid_range"], strict=True)
    )
    out["valid_age_range"] = [ctx.settings.valid_age_min, ctx.settings.valid_age_max]
    out["quantity_per_row"] = _describe(ctx, "purchases_t", "product_quantity")
    out["quantity_fractional_rows"] = ctx.one(
        "SELECT count(*) FROM purchases_t WHERE product_quantity IS NOT NULL AND product_quantity <> floor(product_quantity)"
    )[0]
    gender = ctx.sql(
        "SELECT coalesce(gender, '<null>') AS gender, count(*) AS n_clients, "
        "count(*) FILTER (WHERE n_tx = 0) AS n_without_purchases FROM cx GROUP BY 1 ORDER BY 1"
    )
    gender["share"] = gender["n_clients"] / gender["n_clients"].sum()
    ctx.results.add_table("gender_counts", gender)

    # ---- points sign conventions (transaction level: the value is repeated on every product row)
    rows = []
    for column in POINTS:
        info = ctx.sql(
            f"SELECT '{column}' AS column_name, count(*) AS n_transactions, count({column}_max) AS n_non_null, "
            f"min({column}_min) AS min, max({column}_max) AS max, "
            f"count(*) FILTER (WHERE {column}_max < 0) AS n_negative, count(*) FILTER (WHERE {column}_max = 0) AS n_zero, "
            f"count(*) FILTER (WHERE {column}_max > 0) AS n_positive, avg({column}_max) AS mean FROM tx"
        )
        rows.append(info)
    points = pd.concat(rows, ignore_index=True).rename(columns={"column_name": "column"})
    ctx.results.add_table("points_sign_conventions", points)
    out["purchase_sum_sign"] = {
        "n_negative": out["spend_per_transaction"]["n_negative"],
        "n_zero": out["spend_per_transaction"]["n_zero"],
        "n_positive": out["spend_per_transaction"]["n_positive"],
    }

    # ---- histograms (written as tables; figures are drawn from them)
    ctx.results.add_table(
        "hist_spend_per_transaction", _hist(ctx, "tx", "purchase_sum_max", width=100, lo=-1000, hi=20000)
    )
    ctx.results.add_table(
        "hist_transactions_per_client", _hist(ctx, "cx", "n_tx", width=1, where="n_tx > 0", hi=200)
    )
    ctx.results.add_table(
        "hist_age",
        _hist(
            ctx, "clients_t", "age", width=1, lo=ctx.settings.valid_age_min, hi=ctx.settings.valid_age_max + 1
        ),
    )
    ctx.results.add_table(
        "hist_recency_days", _hist(ctx, "cx_feat", "recency_days", width=5, where="n_tx > 0")
    )
    ctx.results.add_table(
        "hist_stores_per_client", _hist(ctx, "cx", "n_stores", width=1, where="n_stores > 0", hi=40)
    )
    ctx.results.add_table(
        "hist_total_spend_per_client",
        _hist(ctx, "cx", "total_spend", width=1000, where="n_tx > 0", lo=-1000, hi=100000),
    )


def _train_features_view(ctx: Context) -> list[str]:
    """Create ``train_feat``: one row per uplift_train client with segment labels. Returns spend tier labels."""
    s = ctx.settings
    age_labels = band_labels(s.age_band_edges, open_ended=False)
    rec_labels = band_labels(s.recency_band_edges_days, open_ended=True)
    age_case = _case_expr("age", s.age_band_edges, age_labels, open_ended=False)
    # ages outside the valid range are not allowed to land in a band even if the edges would cover them
    age_case = age_case.replace(
        "CASE ", f"CASE WHEN age < {s.valid_age_min} OR age > {s.valid_age_max} THEN NULL ", 1
    )
    rec_case = _case_expr("recency_days", s.recency_band_edges_days, rec_labels, open_ended=True)
    q = list(s.spend_tier_quantiles)
    edges = ctx.con.execute(
        "SELECT quantile_cont(f.total_spend, ?) FROM cx f JOIN uplift_train_t t USING (client_id) WHERE f.n_tx > 0",
        [q],
    ).fetchone()[0]
    edges = [float(e) for e in edges]
    tier_labels = [f"spend_Q{i + 1}" for i in range(len(edges) + 1)]
    tier_case = (
        "CASE "
        + " ".join(
            [f"WHEN total_spend <= {edges[0]!r} THEN '{tier_labels[0]}'"]
            + [f"WHEN total_spend <= {e!r} THEN '{tier_labels[i + 1]}'" for i, e in enumerate(edges[1:])]
            + [f"ELSE '{tier_labels[-1]}'"]
        )
        + " END"
    )
    ctx.con.execute(
        "CREATE OR REPLACE TEMP VIEW train_feat AS "
        "SELECT t.client_id, t.treatment_flg, t.target, f.age, f.gender, f.tenure_days, f.n_tx, f.total_spend, "
        "f.recency_days, f.n_stores, f.n_tx > 0 AS has_purchases, "
        f"coalesce({age_case}, '{INVALID_AGE}') AS age_band, "
        f"CASE WHEN f.n_tx = 0 THEN '{NO_PURCHASES}' ELSE coalesce({rec_case}, '{NO_PURCHASES}') END AS recency_band, "
        f"CASE WHEN f.n_tx = 0 OR f.total_spend IS NULL THEN '{NO_PURCHASES}' ELSE {tier_case} END AS spend_tier, "
        "coalesce(f.gender, '<null>') AS gender_label "
        "FROM (SELECT DISTINCT client_id, treatment_flg, target FROM uplift_train_t) t "
        "LEFT JOIN cx_feat f USING (client_id)"
    )
    ctx.results.section("segments")["spend_tier_edges"] = dict(zip(tier_labels[:-1], edges, strict=True))
    return tier_labels


def section_treatment(ctx: Context) -> None:
    s = ctx.settings
    out = ctx.results.section("treatment")
    ctx.con.execute(
        "CREATE OR REPLACE TEMP VIEW train_clean AS SELECT * FROM uplift_train_t WHERE treatment_flg IN (0, 1) AND target IN (0, 1)"
    )
    bad = ctx.one(
        "SELECT count(*) FILTER (WHERE treatment_flg IS NULL OR treatment_flg NOT IN (0, 1)), "
        "count(*) FILTER (WHERE target IS NULL OR target NOT IN (0, 1)), count(*) FROM uplift_train_t"
    )
    out["rows_with_invalid_treatment_flg"], out["rows_with_invalid_target"], out["train_rows"] = bad
    counts = ctx.sql(
        "SELECT treatment_flg, count(*) AS n, sum(target) AS conversions FROM train_clean GROUP BY 1 ORDER BY 1"
    )
    by = {int(r.treatment_flg): (int(r.n), int(r.conversions)) for r in counts.itertuples()}
    n_c, y_c = by.get(0, (0, 0))
    n_t, y_t = by.get(1, (0, 0))
    total = n_c + n_t
    out["treated_share"] = n_t / total if total else math.nan
    out["overall"] = uplift_stats(
        n_t, y_t, n_c, y_c, alpha=s.alpha, min_arm_size=s.min_arm_size, min_cell_events=s.min_cell_events
    )
    # sample-ratio check against an even split (a design expectation to verify, not a published fact)
    if total:
        z = (n_t - total / 2) / math.sqrt(total * 0.25)
        out["share_vs_half"] = {"z": z, "p_value": math.erfc(abs(z) / math.sqrt(2))}
    ctx.results.add_table("treatment_outcome", counts.assign(rate=counts["conversions"] / counts["n"]))

    tier_labels = _train_features_view(ctx)
    out["spend_tier_labels"] = tier_labels

    # ---- covariate balance between arms (randomisation check; pre-treatment variables only)
    covariates = {
        "age_valid": f"CASE WHEN age BETWEEN {s.valid_age_min} AND {s.valid_age_max} THEN age END",
        "gender_female": "CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END",
        "gender_male": "CASE WHEN gender = 'M' THEN 1.0 ELSE 0.0 END",
        "gender_unknown": "CASE WHEN gender = 'U' THEN 1.0 ELSE 0.0 END",
        "tenure_days": "tenure_days",
        "has_purchases": "CASE WHEN has_purchases THEN 1.0 ELSE 0.0 END",
        "n_transactions": "n_tx",
        "total_spend": "total_spend",
        "recency_days": "recency_days",
        "n_stores": "n_stores",
    }
    rows = []
    for name, expr in covariates.items():
        stats = ctx.sql(
            f"SELECT treatment_flg, avg(v) AS mean, var_samp(v) AS var, count(v) AS n FROM "
            f"(SELECT treatment_flg, {expr} AS v FROM train_feat WHERE treatment_flg IN (0, 1)) GROUP BY 1 ORDER BY 1"
        )
        d = {int(r.treatment_flg): r for r in stats.itertuples()}
        if 0 in d and 1 in d:
            rows.append(
                {
                    "covariate": name,
                    "mean_control": d[0].mean,
                    "mean_treated": d[1].mean,
                    "n_control": int(d[0].n),
                    "n_treated": int(d[1].n),
                    "smd": smd(d[1].mean, d[1].var, d[0].mean, d[0].var),
                }
            )
    balance = pd.DataFrame(rows)
    ctx.results.add_table("covariate_balance", balance)
    out["max_abs_smd"] = float(balance["smd"].abs().max()) if len(balance) else math.nan


def section_segments(ctx: Context) -> None:
    s = ctx.settings
    out = ctx.results.section("segments")
    specs = {
        "age_band": (band_labels(s.age_band_edges, open_ended=False) + [INVALID_AGE]),
        "gender_label": None,
        "spend_tier": ctx.results.section("treatment")["spend_tier_labels"] + [NO_PURCHASES],
        "recency_band": band_labels(s.recency_band_edges_days, open_ended=True) + [NO_PURCHASES],
    }
    frames = []
    for kind, order in specs.items():
        data = ctx.sql(
            f"SELECT {kind} AS segment, "
            "count(*) FILTER (WHERE treatment_flg = 1) AS n_t, coalesce(sum(target) FILTER (WHERE treatment_flg = 1), 0) AS y_t, "
            "count(*) FILTER (WHERE treatment_flg = 0) AS n_c, coalesce(sum(target) FILTER (WHERE treatment_flg = 0), 0) AS y_c "
            f"FROM train_feat WHERE treatment_flg IN (0, 1) AND target IN (0, 1) GROUP BY 1"
        )
        data = data.sort_values(
            "segment",
            key=lambda col: col.map({v: i for i, v in enumerate(order or sorted(col))}) if order else col,
        )
        for r in data.itertuples():
            stats = uplift_stats(
                int(r.n_t),
                int(r.y_t),
                int(r.n_c),
                int(r.y_c),
                alpha=s.alpha,
                min_arm_size=s.min_arm_size,
                min_cell_events=s.min_cell_events,
            )
            frames.append(
                {"segment_kind": "gender" if kind == "gender_label" else kind, "segment": r.segment, **stats}
            )
    table = pd.DataFrame(frames)
    total_n = table[table["segment_kind"] == "age_band"]["n"].sum()
    table["share_of_train"] = table["n"] / total_n if total_n else math.nan
    ctx.results.add_table("uplift_by_segment", table)
    out["n_segments"] = int(len(table))
    out["n_sparse_segments"] = int(table["sparse"].sum())
    out["definitions"] = {
        "age_band": f"client age in [{s.valid_age_min}, {s.valid_age_max}] split at {list(s.age_band_edges)}; others = {INVALID_AGE}",
        "recency_band": "days from last purchase to the recency anchor (see leakage.recency_anchor_label), edges "
        f"{list(s.recency_band_edges_days)}; clients without purchases = {NO_PURCHASES}",
        "spend_tier": "client total spend (sum of purchase_sum once per transaction) in quantile tiers of train clients "
        f"with purchases, quantiles {list(s.spend_tier_quantiles)}; no purchases = {NO_PURCHASES}",
        "caveat": "purchase history may extend past the campaign date (unknown); segments are descriptive, "
        "not pre-campaign-clean until a cutoff is applied",
    }


def section_ids(ctx: Context) -> None:
    out = ctx.results.section("id_formats")
    views = {
        "clients": "clients_t",
        "uplift_train": "uplift_train_t",
        "uplift_test": "uplift_test_t",
        "products": "products_t",
        "purchases": "purchases_t",
    }
    rows = []
    for table, columns in ID_COLUMNS.items():
        for column in columns:
            if column not in {
                r[0] for r in ctx.con.execute(f"DESCRIBE SELECT * FROM {views[table]}").fetchall()
            }:
                continue
            row = ctx.con.execute(
                "SELECT count(*), min(length(v)), max(length(v)), "
                "count(*) FILTER (WHERE regexp_full_match(v, '[0-9a-f]+')), "
                "count(*) FILTER (WHERE regexp_full_match(v, '[0-9]+')), count(*) FILTER (WHERE v LIKE '0%'), "
                "count(*) FILTER (WHERE v <> trim(v) OR v = ''), "
                "count(*) FILTER (WHERE regexp_full_match(v, '[0-9]+e[0-9]+')) "
                f"FROM (SELECT DISTINCT {column} AS v FROM {views[table]} WHERE {column} IS NOT NULL)"
            ).fetchone()
            n = row[0]
            rows.append(
                {
                    "table": table,
                    "column": column,
                    "distinct_values": n,
                    "min_length": row[1],
                    "max_length": row[2],
                    "share_lowercase_hex": row[3] / n if n else math.nan,
                    "share_digits_only": row[4] / n if n else math.nan,
                    "share_leading_zero": row[5] / n if n else math.nan,
                    "n_blank_or_padded": row[6],
                    "n_scientific_notation_like": row[7],
                }
            )
    ctx.results.add_table("id_formats", pd.DataFrame(rows))
    seg = (
        ctx.one(
            "SELECT count(segment_id), count(*) FILTER (WHERE regexp_full_match(segment_id, '[0-9]+\\.0')) FROM products_t"
        )
        if "segment_id" in columns_of(ctx, "products_t")
        else (None, None)
    )
    out["products_segment_id_non_null"], out["products_segment_id_integer_valued_floats"] = seg
    out["interpretation"] = (
        "IDs are opaque strings. Columns whose distinct values are all lowercase hex of one fixed length look "
        "hashed (anonymised); they must be treated as text, never as numbers."
    )


def columns_of(ctx: Context, relation: str) -> list[str]:
    return [r[0] for r in ctx.con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def section_uplift_test(ctx: Context) -> None:
    out = ctx.results.section("uplift_test")
    columns = [r[0] for r in ctx.con.execute("DESCRIBE SELECT * FROM uplift_test_t").fetchall()]
    labels = [c for c in columns if c in {"treatment_flg", "target"}]
    out["columns"] = columns
    out["label_columns_present"] = labels
    out["has_labels"] = bool(labels)
    out["clients"] = ctx.one("SELECT count(DISTINCT client_id) FROM uplift_test_t")[0]
    out["clients_with_purchases"] = ctx.one(
        "SELECT count(*) FROM cx WHERE n_tx > 0 AND client_id IN (SELECT client_id FROM uplift_test_t)"
    )[0]
    out["note"] = (
        "uplift_test has no outcome or treatment columns: no evaluation labels exist in the published data. "
        "Model evaluation must hold out part of uplift_train."
        if not labels
        else "uplift_test carries label columns; verify they are not leaked into training."
    )


def section_semantics(ctx: Context) -> None:
    """Does ``purchase_sum`` repeat per product row? Are the grouping keys safe? What are the line amounts?"""
    s = ctx.settings
    tol = s.money_tolerance
    out = ctx.results.section("semantics")
    n_tx, n_rows = ctx.one("SELECT count(*), sum(n_rows) FROM tx")
    out["transactions"] = n_tx
    out["transactions_definition"] = "distinct (client_id, transaction_id) pairs"
    out["product_rows"] = int(n_rows or 0)
    out["transactions_with_several_rows"] = ctx.one("SELECT count(*) FROM tx WHERE n_rows > 1")[0]
    out["transactions_with_one_row"] = ctx.one("SELECT count(*) FROM tx WHERE n_rows = 1")[0]

    def inconsistent(column: str) -> str:
        return f"count(*) FILTER (WHERE {column}_min <> {column}_max OR ({column}_n > 0 AND {column}_n < n_rows))"

    constants = ["store_id", "ts", "purchase_sum", *POINTS]
    row = ctx.con.execute(
        f"SELECT {', '.join(inconsistent(c) for c in constants)} FROM tx WHERE n_rows > 1"
    ).fetchone()
    out["multi_row_transactions_where_value_differs_between_rows"] = dict(zip(constants, row, strict=True))
    out["multi_row_transactions_total"] = out["transactions_with_several_rows"]

    # naive row-sum vs once-per-transaction sum of purchase_sum
    naive, dedup = ctx.one(
        "SELECT (SELECT sum(purchase_sum) FROM purchases_t), (SELECT sum(purchase_sum_max) FROM tx)"
    )
    out["sum_purchase_sum_over_rows"] = naive
    out["sum_purchase_sum_once_per_transaction"] = dedup
    out["row_sum_overstatement_factor"] = naive / dedup if dedup else math.nan

    # transaction_id as a key
    shared = ctx.one(
        "SELECT count(*), coalesce(sum(n_groups), 0), coalesce(sum(n_rows), 0) FROM "
        "(SELECT transaction_id, count(*) AS n_groups, sum(n_rows) AS n_rows FROM tx GROUP BY 1 HAVING count(*) > 1)"
    )
    out["transaction_ids_shared_by_more_than_one_client"] = shared[0]
    out["client_transactions_with_shared_transaction_id"] = shared[1]
    out["product_rows_with_shared_transaction_id"] = shared[2]
    out["client_transactions_with_more_than_one_store"] = ctx.one(
        "SELECT count(*) FROM tx WHERE store_id_min <> store_id_max"
    )[0]
    out["client_transactions_with_more_than_one_timestamp"] = ctx.one(
        "SELECT count(*) FROM tx WHERE ts_min <> ts_max"
    )[0]
    out["client_transactions_with_null_client"] = ctx.one(
        "SELECT count(*) FROM tx WHERE client_id_n < n_rows"
    )[0]
    out["rows_with_null_transaction_id"] = ctx.one(
        "SELECT count(*) FROM purchases_t WHERE transaction_id IS NULL"
    )[0]
    out["distinct_client_ts_store_triples"] = ctx.one(
        "SELECT count(*) FROM (SELECT DISTINCT client_id_min, ts_min, store_id_min FROM tx)"
    )[0]

    # line amounts: trn_sum_from_iss / trn_sum_from_red against purchase_sum
    row = ctx.con.execute(
        "SELECT count(*) FILTER (WHERE n_iss > 0), count(*) FILTER (WHERE n_red > 0), "
        "count(*) FILTER (WHERE n_iss > 0 AND abs(sum_iss - purchase_sum_max) <= ?), "
        "count(*) FILTER (WHERE n_iss > 0 AND abs(sum_iss + coalesce(sum_red, 0) - purchase_sum_max) <= ?), "
        "count(*) FILTER (WHERE n_iss > 0 AND sum_iss > purchase_sum_max + ?), "
        "count(*) FILTER (WHERE n_iss > 0 AND sum_iss < purchase_sum_max - ?), "
        "count(*) FILTER (WHERE n_iss = 0) "
        "FROM tx WHERE purchase_sum_n > 0",
        [tol, tol, tol, tol],
    ).fetchone()
    out["line_amount_vs_purchase_sum"] = dict(
        zip(
            (
                "transactions_with_iss",
                "transactions_with_red",
                "sum_iss_equals_purchase_sum",
                "sum_iss_plus_red_equals_purchase_sum",
                "sum_iss_above_purchase_sum",
                "sum_iss_below_purchase_sum",
                "transactions_without_any_iss",
            ),
            row,
            strict=True,
        )
    )
    out["line_amount_vs_purchase_sum"]["tolerance"] = tol
    ratio = ctx.con.execute(
        "SELECT quantile_cont(sum_iss / purchase_sum_max, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) FROM tx "
        "WHERE n_iss > 0 AND purchase_sum_max > 0"
    ).fetchone()[0]
    out["sum_iss_over_purchase_sum_quantiles"] = dict(
        zip(("0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"), ratio or [], strict=False)
    )
    rel = ctx.con.execute(
        "SELECT count(*), count(*) FILTER (WHERE abs(sum_iss / purchase_sum_max - 1) <= 1e-6), "
        "count(*) FILTER (WHERE abs(sum_iss / purchase_sum_max - 1) <= 0.01), "
        "count(*) FILTER (WHERE abs(sum_iss / purchase_sum_max - 1) <= 0.05), "
        "count(*) FILTER (WHERE abs(sum_iss / purchase_sum_max - 1) <= 0.10) "
        "FROM tx WHERE n_iss > 0 AND purchase_sum_max > 0"
    ).fetchone()
    out["sum_iss_relative_agreement_with_purchase_sum"] = dict(
        zip(
            ("transactions_compared", "within_1e-6", "within_1pct", "within_5pct", "within_10pct"),
            rel,
            strict=True,
        )
    )
    out["transactions_with_nonpositive_purchase_sum"] = ctx.one(
        "SELECT count(*) FROM tx WHERE purchase_sum_max <= 0"
    )[0]
    out["rows_with_trn_sum_from_iss"], out["rows_with_trn_sum_from_red"], out["rows_total"] = ctx.one(
        "SELECT count(trn_sum_from_iss), count(trn_sum_from_red), count(*) FROM purchases_t"
    )
    out["rows_with_both_iss_and_red"] = ctx.one(
        "SELECT count(*) FROM purchases_t WHERE trn_sum_from_iss IS NOT NULL AND trn_sum_from_red IS NOT NULL"
    )[0]
    out["rows_with_negative_iss"], out["rows_with_zero_iss"] = ctx.one(
        "SELECT count(*) FILTER (WHERE trn_sum_from_iss < 0), count(*) FILTER (WHERE trn_sum_from_iss = 0) FROM purchases_t"
    )
    out["rows_with_nan_or_inf_numeric"] = ctx.one(
        "SELECT count(*) FROM purchases_t WHERE "
        + " OR ".join(
            f"isnan({c}) OR isinf({c})"
            for c in (*POINTS, "purchase_sum", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red")
        )
    )[0]
    shape = ctx.sql(
        "SELECT least(n_rows, 10) AS product_rows_capped_at_10, count(*) AS n_transactions FROM tx GROUP BY 1 ORDER BY 1"
    )
    ctx.results.add_table("rows_per_transaction", shape)
    out["interpretation"] = {
        "purchase_sum_and_points_are_transaction_level": out["multi_row_transactions_total"] > 0
        and all(v == 0 for v in out["multi_row_transactions_where_value_differs_between_rows"].values()),
        "rule": "aggregate purchase_sum and the four points columns ONCE per (client_id, transaction_id), not per "
        "product row, and use (client_id, transaction_id) as the transaction key",
    }


SECTIONS = (
    ("files", section_files),
    ("timestamps", section_timestamps),
    ("leakage", section_leakage),
    ("distributions", section_distributions),
    ("treatment", section_treatment),
    ("segments", section_segments),
    ("id_formats", section_ids),
    ("uplift_test", section_uplift_test),
    ("semantics", section_semantics),
)


# --------------------------------------------------------------------------- orchestration


def run_eda(settings: EdaSettings, *, force: bool = False, figures: bool = True, log=print) -> Results:
    """Run all sections and write the outputs. Returns the :class:`Results`."""
    files = input_files(settings)
    fingerprint = input_fingerprint(settings, files)
    results = Results()
    con = connect(settings)
    ctx = Context(settings=settings, con=con, results=results)
    try:
        started = time.monotonic()
        log(f"[{settings.mode}] preparing typed tables from {settings.input_dir}")
        rebuilt = _timed_prepare(ctx, files, fingerprint, force)
        log(
            f"[{settings.mode}] typed tables {'built' if rebuilt else 'reused (inputs and settings unchanged)'}"
        )
        section_run_info(ctx, files, fingerprint)
        for name, func in SECTIONS:
            log(f"[{settings.mode}] section: {name}")
            _timed(ctx, name, func)
        results.section("run")["stage_seconds"] = ctx.stages
        results.section("run")["total_seconds"] = round(time.monotonic() - started, 3)
        results.write(settings.output_dir)
        if figures:
            from eda.x5_eda_plots import make_figures

            make_figures(results, settings.output_dir / "figures")
    finally:
        con.close()
    return results


def _timed_prepare(ctx: Context, files: Mapping[str, Path], fingerprint: str, force: bool) -> bool:
    started = time.monotonic()
    rebuilt = prepare(ctx, files, fingerprint, force=force)
    ctx.stages["prepare"] = round(time.monotonic() - started, 3)
    if not rebuilt:
        _register_raw(ctx, files)  # src_* views are needed by a few sections
    return rebuilt


def flatten(values: Mapping[str, Any], prefix: str = "") -> pd.DataFrame:
    """Flatten nested scalars into ``key``/``value`` rows (for notebook display)."""
    rows: list[dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        else:
            rows.append({"key": path, "value": node})

    walk(values, prefix)
    return pd.DataFrame(rows, columns=["key", "value"])

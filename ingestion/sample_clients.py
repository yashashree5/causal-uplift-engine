#!/usr/bin/env python3
"""Reproducible 1% **client** sample of the X5 tables.

This samples *clients*, not purchase rows: 1% of the unique ``client_id`` values are selected and
then every row belonging to a selected client is kept, in every table that has a ``client_id``
(``clients``, ``uplift_train``, ``uplift_test``, ``purchases``). A sample of 1% of purchase rows
would split transactions and give clients partial histories; that is not what this does.

Selection method (``hash-rank``)
-------------------------------
1. Universe = distinct non-empty ``client_id`` values of the ``clients`` table (N of them).
2. Each client gets the key ``sha256("<seed>:<client_id>")`` (hex digest).
3. Sort clients by ``(key, client_id)`` and keep the first ``K = round(fraction * N)``.

The result depends only on the *set* of client IDs, the seed and the fraction, so it is independent
of input row order, duplicate rows, file splitting and DuckDB thread scheduling. Exactly K clients
are selected (not "about 1%"). A pure-Python reference implementation of the same rule is used in
the tests.

Other tables
------------
* ``uplift_train`` / ``uplift_test`` / ``purchases``: all rows of the selected clients.
  ``uplift_test`` keeps only its source columns (``client_id``): it stays unlabeled.
* ``products`` has no ``client_id``. It is included as the products *referenced by the sampled
  purchases* (``product_id`` in the sampled purchase rows), so every sampled purchase can be
  joined to a product row if the source had one.

All columns are read and written as text, so the source schema, string IDs (leading zeros) and
number formatting are preserved exactly.

Exit codes: 0 ok, 1 validation failure, 2 usage/config error, 3 remote conflict or integrity failure,
4 cloud not configured.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python ingestion/sample_clients.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

from ingestion import land_raw  # noqa: E402
from ingestion.object_store import (  # noqa: E402
    CloudConfigError,
    ObjectStoreError,
    local_checksums,
)
from ingestion.x5_common import (  # noqa: E402
    ConfigError,
    atomic_write_json,
    config_get,
    configured_tables,
    data_root,
    downloads_dir,
    file_digests,
    load_config,
    read_json,
    resolve_path,
)

EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_USAGE = 2
EXIT_REMOTE = 3
EXIT_CLOUD_CONFIG = 4

SAMPLE_METHOD = (
    "hash-rank: sort unique client_ids of the universe table by (sha256('<seed>:<client_id>') hex, client_id) "
    "and keep the first K = round(fraction * N)"
)
METADATA_VERSION = 1
IDS_FILE = "selected_client_ids.csv"
METADATA_FILE = "sample_metadata.json"

CLIENT_TABLES = ("clients", "uplift_train", "uplift_test", "purchases")


class SamplingError(Exception):
    """The sample cannot be produced (bad inputs or settings)."""


# --------------------------------------------------------------------------- pure helpers


def sample_size(n_unique: int, fraction: float) -> int:
    """K = round(fraction * N) with ``round`` = round half up; must be at least 1."""
    if not 0 < fraction <= 1:
        raise SamplingError(f"fraction must be in (0, 1], got {fraction}")
    if n_unique < 1:
        raise SamplingError("the universe table contains no client IDs")
    k = int(fraction * n_unique + 0.5)
    if k < 1:
        raise SamplingError(f"fraction {fraction} of {n_unique} client IDs rounds to an empty sample")
    return min(k, n_unique)


def sampling_key(seed: int, client_id: str) -> str:
    return hashlib.sha256(f"{seed}:{client_id}".encode()).hexdigest()


def reference_select(client_ids: Iterable[str], fraction: float, seed: int) -> list[str]:
    """Pure-Python implementation of the hash-rank rule. Returns the selected IDs sorted ascending."""
    unique = {cid for cid in client_ids if cid not in (None, "")}
    k = sample_size(len(unique), fraction)
    ranked = sorted(unique, key=lambda cid: (sampling_key(seed, cid), cid))
    return sorted(ranked[:k])


def ids_checksum(sorted_ids: Iterable[str]) -> str:
    """SHA-256 over the ascending-sorted IDs, one per line, with a trailing newline."""
    digest = hashlib.sha256()
    for cid in sorted_ids:
        digest.update(cid.encode() + b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------- DuckDB access


def connect(config: Mapping[str, Any]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    settings = config.get("duckdb") or {}
    if settings.get("memory_limit"):
        con.execute(f"SET memory_limit='{settings['memory_limit']}'")
    if settings.get("threads"):
        con.execute(f"SET threads={int(settings['threads'])}")
    if settings.get("temp_directory"):
        temp = resolve_path(settings["temp_directory"])
        temp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp}'")
    return con


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def find_source(table: str, input_dir: Path) -> Path:
    for name in (f"{table}.csv.gz", f"{table}.csv"):
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    raise SamplingError(
        f"no input file for table '{table}' in {input_dir} (expected {table}.csv.gz or {table}.csv)"
    )


def register_sources(
    con: duckdb.DuckDBPyConnection, tables: Iterable[str], input_dir: Path
) -> dict[str, Path]:
    sources = {}
    for table in tables:
        path = find_source(table, input_dir)
        sources[table] = path
        con.execute(
            f"CREATE OR REPLACE VIEW src_{table} AS SELECT * FROM read_csv({_q(path)}, header=true, "
            "all_varchar=true, delim=',', quote='\"', escape='\"')"
        )
    return sources


def columns_of(con: duckdb.DuckDBPyConnection, relation: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def select_client_ids(
    con: duckdb.DuckDBPyConnection, universe_table: str, fraction: float, seed: int
) -> tuple[int, int, int]:
    """Create temp table ``selected`` and return ``(n_universe, k, rows_without_client_id)``."""
    if "client_id" not in columns_of(con, f"src_{universe_table}"):
        raise SamplingError(f"universe table '{universe_table}' has no client_id column")
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE universe AS "
        f"SELECT DISTINCT client_id FROM src_{universe_table} WHERE client_id IS NOT NULL AND client_id <> ''"
    )
    n_universe = con.execute("SELECT count(*) FROM universe").fetchone()[0]
    missing = con.execute(
        f"SELECT count(*) FROM src_{universe_table} WHERE client_id IS NULL OR client_id = ''"
    ).fetchone()[0]
    k = sample_size(n_universe, fraction)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE selected AS "
        "SELECT client_id FROM (SELECT client_id, sha256(? || ':' || client_id) AS h FROM universe) "
        "ORDER BY h, client_id LIMIT ?",
        [str(seed), k],
    )
    return n_universe, k, missing


def selected_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute("SELECT client_id FROM selected ORDER BY client_id").fetchall()]


# --------------------------------------------------------------------------- sampling


@dataclasses.dataclass
class SampleSettings:
    fraction: float
    seed: int
    universe_table: str
    output_dir: Path
    compression: str
    object_prefix: str
    upload_chunk_bytes: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SampleSettings:
        section = config.get("sampling")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration has no 'sampling' section (use configs/x5/sample.yaml)")
        try:
            fraction = float(section["fraction"])
            seed = int(section["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("sampling.fraction (float) and sampling.seed (int) are required") from exc
        if not 0 < fraction <= 1:
            raise ConfigError(f"sampling.fraction must be in (0, 1], got {fraction}")
        compression = str(section.get("compression", "gzip"))
        if compression not in {"gzip", "none"}:
            raise ConfigError("sampling.compression must be 'gzip' or 'none'")
        prefix = str(section.get("object_prefix", "raw_sample")).strip("/")
        if not prefix or ".." in prefix.split("/"):
            raise ConfigError(f"invalid sampling.object_prefix: {prefix!r}")
        chunk = int(section.get("upload_chunk_bytes", 8 * 1024 * 1024))
        if chunk <= 0 or chunk % (256 * 1024):
            raise ConfigError("sampling.upload_chunk_bytes must be a positive multiple of 262144")
        return cls(
            fraction=fraction,
            seed=seed,
            universe_table=str(section.get("universe_table", "clients")),
            output_dir=resolve_path(section.get("output_dir", "raw_sample")),
            compression=compression,
            object_prefix=prefix,
            upload_chunk_bytes=chunk,
        )


def _output_name(table: str, compression: str) -> str:
    return f"{table}.csv.gz" if compression == "gzip" else f"{table}.csv"


def _copy(con: duckdb.DuckDBPyConnection, query: str, target: Path, compression: str) -> int:
    option = ", COMPRESSION gzip" if compression == "gzip" else ""
    con.execute(f"COPY ({query}) TO {_q(target)} (FORMAT CSV, HEADER true{option})")
    return con.execute(
        f"SELECT count(*) FROM read_csv({_q(target)}, header=true, all_varchar=true)"
    ).fetchone()[0]


def write_sample(
    con: duckdb.DuckDBPyConnection,
    sources: Mapping[str, Path],
    settings: SampleSettings,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Write the sampled tables and the ID list into ``out_dir``. Returns per-file info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}

    ids_path = out_dir / IDS_FILE
    con.execute(
        f"COPY (SELECT client_id FROM selected ORDER BY client_id) TO {_q(ids_path)} (FORMAT CSV, HEADER true)"
    )

    for table in CLIENT_TABLES:
        if table not in sources:
            continue
        target = out_dir / _output_name(table, settings.compression)
        # Rows keep their source-file order. The explicit row number makes that order independent of
        # DuckDB's thread scheduling (a bare semi-join may emit chunks in a different order per run).
        query = (
            f"SELECT * EXCLUDE (_src_row) FROM (SELECT *, row_number() OVER () AS _src_row FROM src_{table}) "
            "WHERE client_id IN (SELECT client_id FROM selected) ORDER BY _src_row"
        )
        if table == "purchases":
            con.execute(f"CREATE OR REPLACE TEMP TABLE sample_purchases AS {query}")
            query = "SELECT * FROM sample_purchases"
        files[table] = {"path": target, "rows": _copy(con, query, target, settings.compression)}

    if "products" in sources:
        if "purchases" not in sources:
            raise SamplingError(
                "products can only be sampled together with purchases (it is defined by them)"
            )
        target = out_dir / _output_name("products", settings.compression)
        query = (
            "SELECT * EXCLUDE (_src_row) FROM (SELECT *, row_number() OVER () AS _src_row FROM src_products) "
            "WHERE product_id IN (SELECT DISTINCT product_id FROM sample_purchases WHERE product_id IS NOT NULL) "
            "ORDER BY _src_row"
        )
        files["products"] = {"path": target, "rows": _copy(con, query, target, settings.compression)}
    return files


def _read(out: Path) -> str:
    return f"read_csv({_q(out)}, header=true, all_varchar=true)"


def validate_sample(
    con: duckdb.DuckDBPyConnection,
    sources: Mapping[str, Path],
    files: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    *,
    expected_k: int,
    check_orphans: bool,
) -> dict[str, Any]:
    """Re-read the written files and check uniqueness, referential consistency and schema."""
    problems: list[str] = []
    checks: dict[str, Any] = {}

    ids_file = out_dir / IDS_FILE
    n_ids, n_distinct = con.execute(
        f"SELECT count(*), count(DISTINCT client_id) FROM {_read(ids_file)}"
    ).fetchone()
    checks["selected_ids"] = {"rows": n_ids, "distinct": n_distinct, "expected": expected_k}
    if n_ids != n_distinct:
        problems.append(f"selected client IDs are not unique ({n_ids} rows, {n_distinct} distinct)")
    if n_ids != expected_k:
        problems.append(f"selected {n_ids} client IDs, expected {expected_k}")

    for table, info in files.items():
        out = Path(info["path"])
        source_columns = columns_of(con, f"src_{table}")
        sample_columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {_read(out)}").fetchall()]
        entry: dict[str, Any] = {"rows": info["rows"], "schema_preserved": sample_columns == source_columns}
        if sample_columns != source_columns:
            problems.append(f"{table}: columns differ from the source ({sample_columns} vs {source_columns})")
        if table in CLIENT_TABLES:
            outside = con.execute(
                f"SELECT count(*) FROM {_read(out)} WHERE client_id NOT IN (SELECT client_id FROM selected)"
            ).fetchone()[0]
            entry["rows_outside_selected_clients"] = outside
            if outside:
                problems.append(f"{table}: {outside} row(s) belong to clients that were not selected")
            entry["distinct_clients"] = con.execute(
                f"SELECT count(DISTINCT client_id) FROM {_read(out)}"
            ).fetchone()[0]
        checks[table] = entry

    if "uplift_test" in files:
        test_columns = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM {_read(Path(files['uplift_test']['path']))}"
            ).fetchall()
        ]
        leaked = [c for c in test_columns if c in {"treatment_flg", "target"}]
        checks["uplift_test"]["label_columns_present"] = leaked
        if leaked:
            problems.append(f"uplift_test sample contains label column(s) {leaked}")

    clients_out = Path(files["clients"]["path"]) if "clients" in files else None
    if clients_out is not None:
        for table in ("uplift_train", "uplift_test", "purchases"):
            if table in files:
                missing = con.execute(
                    f"SELECT count(DISTINCT client_id) FROM {_read(Path(files[table]['path']))} "
                    f"WHERE client_id NOT IN (SELECT client_id FROM {_read(clients_out)})"
                ).fetchone()[0]
                checks[table]["clients_missing_from_sampled_clients"] = missing
                if missing:
                    problems.append(f"{table}: {missing} client(s) are not in the sampled clients table")
        checks["selected_clients_missing_from_clients_output"] = con.execute(
            f"SELECT count(*) FROM selected WHERE client_id NOT IN (SELECT client_id FROM {_read(clients_out)})"
        ).fetchone()[0]
        if checks["selected_clients_missing_from_clients_output"]:
            problems.append("some selected client IDs have no row in the sampled clients table")

    if "uplift_train" in files and "uplift_test" in files:
        overlap = con.execute(
            f"SELECT count(*) FROM {_read(Path(files['uplift_train']['path']))} WHERE client_id IN "
            f"(SELECT client_id FROM {_read(Path(files['uplift_test']['path']))})"
        ).fetchone()[0]
        checks["train_test_overlap_clients"] = overlap
        if overlap:
            problems.append(f"{overlap} sampled client(s) appear in both uplift_train and uplift_test")

    if "products" in files and "purchases" in files:
        products_out = Path(files["products"]["path"])
        purchases_out = Path(files["purchases"]["path"])
        referenced = con.execute(
            f"SELECT count(DISTINCT product_id) FROM {_read(purchases_out)} WHERE product_id IS NOT NULL"
        ).fetchone()[0]
        in_products = con.execute(
            f"SELECT count(DISTINCT product_id) FROM {_read(purchases_out)} WHERE product_id IN "
            f"(SELECT product_id FROM src_products)"
        ).fetchone()[0]
        unreferenced = con.execute(
            f"SELECT count(*) FROM {_read(products_out)} WHERE product_id NOT IN "
            f"(SELECT DISTINCT product_id FROM {_read(purchases_out)})"
        ).fetchone()[0]
        checks["products"].update(
            distinct_products_referenced_by_sampled_purchases=referenced,
            referenced_products_found_in_products=in_products,
            referenced_products_missing_from_products=referenced - in_products,
            products_rows_not_referenced_by_sample=unreferenced,
        )
        if unreferenced:
            problems.append(f"products sample has {unreferenced} row(s) not referenced by sampled purchases")
        if checks["products"]["rows"] != in_products:
            # duplicate product_id rows in the source would make rows > distinct ids
            dupes = con.execute(
                f"SELECT count(*) - count(DISTINCT product_id) FROM {_read(products_out)}"
            ).fetchone()[0]
            if checks["products"]["rows"] - dupes != in_products:
                problems.append("products sample row count does not match the referenced products found")

    if check_orphans:
        orphans: dict[str, int] = {}
        for table in ("uplift_train", "uplift_test", "purchases"):
            if table in sources and "clients" in sources:
                orphans[table] = con.execute(
                    f"SELECT count(DISTINCT client_id) FROM src_{table} "
                    f"WHERE client_id NOT IN (SELECT client_id FROM universe)"
                ).fetchone()[0]
        checks["source_client_ids_absent_from_universe"] = orphans

    return {"ok": not problems, "problems": problems, "checks": checks}


def _find_manifest(root: Path, table: str, artifact_md5: str | None) -> dict[str, Any] | None:
    base = root / "manifests" / table
    if not base.is_dir() or artifact_md5 is None:
        return None
    for folder in sorted(base.glob("ingest_date=*"), reverse=True):
        path = folder / "manifest.json"
        if path.is_file():
            try:
                manifest = read_json(path)
            except (OSError, ValueError):
                continue
            if manifest.get("md5") == artifact_md5:
                return {
                    "ingest_date": manifest["ingest_date"],
                    "row_count": manifest["row_count"],
                    "md5": manifest["md5"],
                }
    return None


def source_provenance(
    config: Mapping[str, Any], root: Path, sources: Mapping[str, Path], input_dir: Path
) -> dict[str, Any]:
    """Link every input to its download provenance (when present) and, if found, its manifest."""
    prov_name = str(config_get(config, "paths.provenance_file", "download_provenance.json"))
    prov_path = downloads_dir(config, root) / prov_name
    records = (
        read_json(prov_path).get("artifacts", {})
        if prov_path.is_file() and input_dir == downloads_dir(config, root)
        else {}
    )
    out: dict[str, Any] = {}
    for table, path in sources.items():
        record = records.get(table)
        digests = file_digests(path, ("md5", "sha256"))
        entry: dict[str, Any] = {"artifact": path.name, "bytes": path.stat().st_size, **digests}
        if record:
            entry["provenance_matches_artifact"] = record.get("sha256") == digests["sha256"]
            entry["source"] = {
                k: (record.get("source") or {}).get(k)
                for k in ("name", "url", "repo_id_resolved", "revision_requested", "revision_resolved")
            }
            entry["retrieved_at"] = record.get("retrieved_at")
        else:
            entry["provenance_matches_artifact"] = None
        entry["manifest"] = _find_manifest(root, table, digests["md5"])
        out[table] = entry
    return out


def build_metadata(
    settings: SampleSettings,
    *,
    n_universe: int,
    k: int,
    rows_without_client_id: int,
    ids: list[str],
    files: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = {}
    for table, info in files.items():
        path = Path(info["path"])
        outputs[table] = {
            "file": path.name,
            "rows": info["rows"],
            "bytes": path.stat().st_size,
            **file_digests(path, ("md5",)),
        }
    return {
        "metadata_version": METADATA_VERSION,
        "dataset": "x5_retailhero_uplift",
        "sampling_unit": "client_id (unique); all rows of a selected client are kept, never a fraction of rows",
        "method": SAMPLE_METHOD,
        "fraction": settings.fraction,
        "seed": settings.seed,
        "universe": {
            "table": settings.universe_table,
            "unique_client_ids": n_universe,
            "rows_without_client_id": rows_without_client_id,
        },
        "selected_client_count": k,
        "selected_ids_file": IDS_FILE,
        "selected_ids_checksum": {
            "algorithm": "sha256 over ascending-sorted client_ids, one per line, trailing newline",
            "value": ids_checksum(ids),
        },
        "table_rules": {
            "clients": "rows of selected clients",
            "uplift_train": "rows of selected clients (labels preserved)",
            "uplift_test": "rows of selected clients; stays unlabeled (source columns only)",
            "purchases": "all product rows of all transactions of selected clients",
            "products": "products referenced by sampled purchases (products has no client_id)",
        },
        "outputs": outputs,
        "validation": validation,
        "source": provenance,
        "compression": settings.compression,
    }


def run_sampling(
    config: Mapping[str, Any],
    settings: SampleSettings,
    root: Path,
    input_dir: Path,
    *,
    check_orphans: bool = True,
    dry_run: bool = False,
    progress=lambda message: None,
) -> dict[str, Any]:
    tables = [t for t in configured_tables(config) if t in {*CLIENT_TABLES, "products"}]
    if settings.universe_table not in tables:
        raise SamplingError(f"universe table '{settings.universe_table}' is not a configured table")
    con = connect(config)
    try:
        sources = register_sources(con, tables, input_dir)
        started = time.monotonic()
        n_universe, k, missing = select_client_ids(
            con, settings.universe_table, settings.fraction, settings.seed
        )
        ids = selected_ids(con)
        progress(f"selected {k:,} of {n_universe:,} unique client IDs ({time.monotonic() - started:.1f}s)")
        if dry_run:
            return {
                "dry_run": True,
                "universe": n_universe,
                "selected": k,
                "ids_checksum": ids_checksum(ids),
                "would_write": [
                    str(settings.output_dir / _output_name(t, settings.compression)) for t in tables
                ],
            }
        tmp_dir = settings.output_dir.with_name(settings.output_dir.name + f".tmp-{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            progress("writing sampled tables (this scans purchases once)")
            files = write_sample(con, sources, settings, tmp_dir)
            progress("validating the written files")
            validation = validate_sample(
                con, sources, files, tmp_dir, expected_k=k, check_orphans=check_orphans
            )
            provenance = source_provenance(config, root, sources, input_dir)
            metadata = build_metadata(
                settings,
                n_universe=n_universe,
                k=k,
                rows_without_client_id=missing,
                ids=ids,
                files=files,
                validation=validation,
                provenance=provenance,
            )
            if not validation["ok"]:
                metadata_path = tmp_dir / METADATA_FILE
                atomic_write_json(metadata_path, metadata)
                failed_dir = settings.output_dir.with_name(settings.output_dir.name + ".failed")
                shutil.rmtree(failed_dir, ignore_errors=True)
                os.replace(tmp_dir, failed_dir)
                raise SamplingError(
                    "sample validation failed (nothing published; inspect "
                    + str(failed_dir)
                    + "): "
                    + "; ".join(validation["problems"])
                )
            atomic_write_json(tmp_dir / METADATA_FILE, metadata)
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            for stale in settings.output_dir.iterdir():
                if stale.is_file():
                    stale.unlink()
            for produced in tmp_dir.iterdir():
                os.replace(produced, settings.output_dir / produced.name)
            return metadata
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        con.close()


# --------------------------------------------------------------------------- upload


def upload_sample(
    settings: SampleSettings, store, *, overwrite: bool, progress=lambda message: None
) -> tuple[int, list[land_raw.ObjectResult]]:
    items = []
    for path in sorted(settings.output_dir.iterdir()):
        if not path.is_file():
            continue
        content_type = "application/json" if path.suffix == ".json" else land_raw._content_type(path)
        table = path.name.split(".")[0]
        items.append(
            land_raw.UploadItem(
                table,
                "sample",
                path,
                f"{settings.object_prefix}/{path.name}",
                content_type,
                local_checksums(path),
            )
        )
    # metadata last: its presence marks a complete sample
    items.sort(key=lambda item: item.local_path.name == METADATA_FILE)
    decisions = land_raw.preflight(items, store, overwrite=overwrite)
    conflicts = [d for d in decisions if d[2] == "conflict"]
    if conflicts:
        results = [
            land_raw._result_for(i, "conflict", r, "remote content differs; use --overwrite")
            for i, r, _ in conflicts
        ]
        return EXIT_REMOTE, results
    try:
        return EXIT_OK, land_raw.execute_upload(decisions, store, settings.upload_chunk_bytes, progress)
    except land_raw._LandingAbort as abort:
        return EXIT_REMOTE, abort.results


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reproducible 1% client sample of the X5 tables to raw_sample/. "
        "Precedence: CLI flag > env (X5_GCS_BUCKET, GOOGLE_CLOUD_PROJECT) > YAML > default.",
    )
    parser.add_argument("--config", default=str(resolve_path("configs/x5/sample.yaml")))
    parser.add_argument(
        "--data-root", help="override paths.data_root (inputs are read from <data-root>/downloads)"
    )
    parser.add_argument("--input-dir", help="directory with <table>.csv.gz or <table>.csv inputs")
    parser.add_argument("--output-dir", help="override sampling.output_dir")
    parser.add_argument(
        "--fraction", type=float, help="override sampling.fraction (fraction of unique client IDs)"
    )
    parser.add_argument("--seed", type=int, help="override sampling.seed")
    parser.add_argument(
        "--no-check-orphans",
        action="store_true",
        help="skip the extra full scans for IDs missing from clients",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="select client IDs and print the plan; write nothing"
    )
    parser.add_argument(
        "--upload", action="store_true", help="also upload raw_sample/ to gs://<bucket>/<object_prefix>/"
    )
    parser.add_argument("--bucket")
    parser.add_argument("--project")
    parser.add_argument(
        "--overwrite", action="store_true", help="with --upload: replace differing remote objects"
    )
    parser.add_argument(
        "--local-bucket-dir", help="rehearse --upload into <dir>/<bucket>/ (not a cloud check)"
    )
    return parser


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None, store=None) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        config = load_config(args.config)
        if args.fraction is not None:
            config.setdefault("sampling", {})["fraction"] = args.fraction
        if args.seed is not None:
            config.setdefault("sampling", {})["seed"] = args.seed
        if args.output_dir:
            config.setdefault("sampling", {})["output_dir"] = args.output_dir
        settings = SampleSettings.from_config(config)
        root = data_root(config, args.data_root)
        input_dir = resolve_path(args.input_dir) if args.input_dir else downloads_dir(config, root)
        bucket = land_raw.resolve_bucket(args.bucket, config, env)
        project = land_raw.resolve_project(args.project, config, env)
        if args.upload and args.dry_run:
            raise ConfigError("--upload and --dry-run are mutually exclusive")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = run_sampling(
            config,
            settings,
            root,
            input_dir,
            check_orphans=not args.no_check_orphans,
            dry_run=args.dry_run,
            progress=progress,
        )
    except SamplingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except duckdb.Error as exc:
        print(f"error: DuckDB could not read the inputs: {exc}", file=sys.stderr)
        return EXIT_VALIDATION

    if args.dry_run:
        print(
            f"dry run: {result['selected']:,} of {result['universe']:,} client IDs would be selected "
            f"(fraction={settings.fraction}, seed={settings.seed}); selected-ID checksum {result['ids_checksum']}"
        )
        for path in result["would_write"]:
            print(f"  would write {path}")
        return EXIT_OK

    print(f"sample written to {settings.output_dir}")
    print(
        f"  clients selected: {result['selected_client_count']:,} of {result['universe']['unique_client_ids']:,}"
    )
    print(f"  selected-ID checksum: {result['selected_ids_checksum']['value']}")
    for table, info in result["outputs"].items():
        print(f"  {table:<13} rows={info['rows']:>9,}  {info['file']}")
    for table, count in (
        result["validation"]["checks"].get("source_client_ids_absent_from_universe", {}).items()
    ):
        if count:
            print(
                f"  note: {count} distinct client_id(s) in source {table} are absent from clients and cannot be sampled"
            )

    if not args.upload:
        return EXIT_OK
    if bucket is None:
        print("error: --upload needs a bucket (gcp.bucket, X5_GCS_BUCKET or --bucket).", file=sys.stderr)
        return EXIT_CLOUD_CONFIG
    try:
        active = store or land_raw.make_store(bucket, project, args.local_bucket_dir)
        code, results = upload_sample(settings, active, overwrite=args.overwrite, progress=progress)
    except CloudConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CLOUD_CONFIG
    except ObjectStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REMOTE
    for r in results:
        print(f"  {r.status:<18} {r.object_name}")
    if code != EXIT_OK:
        print(
            "error: upload stopped; remote objects with different content are never replaced without --overwrite",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

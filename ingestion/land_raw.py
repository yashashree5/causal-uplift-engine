#!/usr/bin/env python3
"""Build manifests for the downloaded X5 artifacts and land them, unchanged, in Cloud Storage.

Target layout (work plan "Bucket layout")::

    gs://<bucket>/raw/batch/<logical_table>/ingest_date=YYYY-MM-DD/<artifact>
    gs://<bucket>/raw/batch/<logical_table>/ingest_date=YYYY-MM-DD/manifest.json

Modes
-----
``--manifest-only``  scan the local artifacts, write manifests and a validation report; no cloud access.
``--dry-run``        as above, then print the exact ``gs://`` objects an upload would write. With
                     ``--check-remote`` it also looks the objects up (needs a bucket and credentials) and
                     predicts upload / skip / conflict. Nothing is written remotely.
(default)            validate, then upload artifacts followed by manifests.

Exit codes: 0 ok, 1 validation failure (missing file/column, bad provenance, strict expectation miss),
2 usage/configuration error, 3 remote conflict or transfer-integrity failure, 4 cloud not configured
(bucket/credentials/permissions).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python ingestion/land_raw.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import manifest as mf  # noqa: E402
from ingestion.object_store import (  # noqa: E402
    CloudConfigError,
    DirectoryObjectStore,
    GcsObjectStore,
    LocalChecksums,
    ObjectStore,
    ObjectStoreError,
    PreconditionFailed,
    RemoteObject,
    TransferIntegrityError,
    local_checksums,
    same_content,
)
from ingestion.x5_common import (  # noqa: E402
    ConfigError,
    atomic_write_json,
    config_get,
    configured_tables,
    data_root,
    downloads_dir,
    load_config,
    resolve_path,
)

EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_USAGE = 2
EXIT_REMOTE = 3
EXIT_CLOUD_CONFIG = 4

LOG = logging.getLogger("x5.land_raw")

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLACEHOLDER_BUCKET = "<BUCKET-NOT-CONFIGURED>"


# --------------------------------------------------------------------------- settings


@dataclasses.dataclass
class LandingSettings:
    object_prefix: str
    manifests_subdir: str
    upload_chunk_bytes: int
    required_columns: dict[str, list[str]]
    expectations: dict[str, dict[str, Any]]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LandingSettings:
        section = config.get("raw_landing")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration has no 'raw_landing' section (use configs/x5/raw_landing.yaml)")
        prefix = str(section.get("object_prefix", "raw/batch")).strip("/")
        if not prefix or ".." in prefix.split("/"):
            raise ConfigError(f"invalid raw_landing.object_prefix: {prefix!r}")
        chunk = int(section.get("upload_chunk_bytes", 8 * 1024 * 1024))
        if chunk <= 0 or chunk % (256 * 1024):
            raise ConfigError("raw_landing.upload_chunk_bytes must be a positive multiple of 262144")
        return cls(
            object_prefix=prefix,
            manifests_subdir=str(section.get("manifests_subdir", "manifests")),
            upload_chunk_bytes=chunk,
            required_columns={k: list(v) for k, v in (section.get("required_columns") or {}).items()},
            expectations={k: dict(v) for k, v in (section.get("expectations") or {}).items()},
        )


def resolve_ingest_date(cli_value: str | None, config: Mapping[str, Any], today: date | None = None) -> str:
    value = cli_value or config_get(config, "raw_landing.ingest_date")
    if value is None:
        value = (today or datetime.now(UTC).date()).isoformat()
    value = str(value)
    if not _DATE_RE.match(value):
        raise ConfigError(f"ingest date must be YYYY-MM-DD, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"invalid ingest date {value!r}: {exc}") from exc
    return value


def normalise_bucket(value: str | None) -> str | None:
    if value is None:
        return None
    name = value.strip()
    if name.startswith("gs://"):
        name = name[5:]
    name = name.strip("/")
    if not name:
        return None
    if "/" in name or not _BUCKET_RE.match(name):
        raise ConfigError(f"invalid bucket name {value!r}: give the bare bucket name, e.g. my-project-x5")
    return name


def resolve_bucket(cli_value: str | None, config: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    """CLI flag > X5_GCS_BUCKET > gcp.bucket in the YAML."""
    return normalise_bucket(cli_value or env.get("X5_GCS_BUCKET") or config_get(config, "gcp.bucket"))


def resolve_project(cli_value: str | None, config: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    return cli_value or env.get("GOOGLE_CLOUD_PROJECT") or config_get(config, "gcp.project")


# --------------------------------------------------------------------------- manifests


@dataclasses.dataclass
class TableOutcome:
    table: str
    artifact_path: Path | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    manifest_bytes: bytes | None = None
    check: mf.TableCheck | None = None
    error: str | None = None
    scan_seconds: float | None = None
    reused: bool = False


def find_artifact(table: str, downloads: Path, provenance: Mapping[str, Any]) -> Path | None:
    record = (provenance.get("artifacts") or {}).get(table)
    names = [record["artifact_name"]] if record and record.get("artifact_name") else []
    names += [f"{table}.csv.gz", f"{table}.csv"]
    for name in names:
        candidate = downloads / name
        if candidate.is_file():
            return candidate
    return None


def _reusable_manifest(
    manifests_root: Path, table: str, ingest_date: str, artifact: Path
) -> dict[str, Any] | None:
    path = manifests_root / table / f"ingest_date={ingest_date}" / "manifest.json"
    if not path.is_file():
        return None
    try:
        existing = mf.read_json(path)
        if (
            existing.get("manifest_version") == mf.MANIFEST_VERSION
            and existing.get("row_count_method") == mf.ROW_COUNT_METHOD
            and existing.get("filename") == artifact.name
            and existing.get("bytes") == artifact.stat().st_size
            and existing.get("md5") == mf.file_digests(artifact, ("md5",))["md5"]
        ):
            return existing
    except (OSError, ValueError, KeyError):
        return None
    return None


def build_outcomes(
    config: Mapping[str, Any],
    settings: LandingSettings,
    root: Path,
    tables: tuple[str, ...],
    ingest_date: str,
    *,
    reuse: bool = False,
    progress: Callable[[str], None] = lambda message: None,
) -> list[TableOutcome]:
    downloads = downloads_dir(config, root)
    provenance = mf.load_download_provenance(
        downloads, str(config_get(config, "paths.provenance_file", "download_provenance.json"))
    )
    manifests_root = root / settings.manifests_subdir
    outcomes: list[TableOutcome] = []
    for table in tables:
        outcome = TableOutcome(table)
        artifact = find_artifact(table, downloads, provenance)
        if artifact is None:
            outcome.error = f"required file for table '{table}' not found in {downloads}"
            outcome.check = mf.validate_manifest(table, None, required_columns=None, expected_rows=None)
            outcomes.append(outcome)
            continue
        outcome.artifact_path = artifact
        try:
            manifest = _reusable_manifest(manifests_root, table, ingest_date, artifact) if reuse else None
            if manifest is not None:
                outcome.reused = True
                progress(f"[{table}] reusing existing manifest for unchanged {artifact.name}")
            else:
                progress(f"[{table}] scanning {artifact.name} ({artifact.stat().st_size:,} bytes)")
                started = time.monotonic()
                info = mf.inspect_artifact(artifact)
                outcome.scan_seconds = round(time.monotonic() - started, 2)
                manifest = mf.build_table_manifest(
                    table,
                    info,
                    ingest_date=ingest_date,
                    object_prefix=settings.object_prefix,
                    provenance_record=(provenance.get("artifacts") or {}).get(table),
                )
        except mf.ManifestError as exc:
            outcome.error = str(exc)
            outcome.check = mf.TableCheck(table, errors=[str(exc)])
            outcomes.append(outcome)
            continue
        outcome.manifest = manifest
        outcome.manifest_bytes = mf.manifest_bytes(manifest)
        outcome.manifest_path = mf.write_table_manifest(manifest, manifests_root)
        outcome.check = mf.validate_manifest(
            table,
            manifest,
            required_columns=settings.required_columns.get(table),
            expected_rows=settings.expectations.get(table),
        )
        outcomes.append(outcome)
    return outcomes


# --------------------------------------------------------------------------- upload plan


@dataclasses.dataclass
class UploadItem:
    table: str
    kind: str  # artifact | manifest
    local_path: Path
    object_name: str
    content_type: str
    local: LocalChecksums


@dataclasses.dataclass
class ObjectResult:
    object_name: str
    kind: str
    table: str
    # uploaded | overwritten | skipped_identical | would_upload | would_skip_identical
    # | would_overwrite | conflict | failed
    status: str
    local_size: int
    local_md5_base64: str
    local_crc32c_base64: str | None
    remote_size: int | None = None
    remote_md5_base64: str | None = None
    remote_crc32c_base64: str | None = None
    remote_generation: int | None = None
    transfer_verified: bool | None = None
    detail: str | None = None


def plan_uploads(
    outcomes: list[TableOutcome], settings: LandingSettings, ingest_date: str
) -> list[UploadItem]:
    """Artifacts first, then manifests (a manifest's presence signals a complete table landing)."""
    items: list[UploadItem] = []
    manifest_items: list[UploadItem] = []
    for outcome in outcomes:
        if outcome.manifest is None or outcome.artifact_path is None or outcome.manifest_path is None:
            continue
        table = outcome.table
        artifact_name = mf.raw_object_name(
            settings.object_prefix, table, ingest_date, outcome.artifact_path.name
        )
        items.append(
            UploadItem(
                table,
                "artifact",
                outcome.artifact_path,
                artifact_name,
                _content_type(outcome.artifact_path),
                local_checksums(outcome.artifact_path),
            )
        )
        manifest_name = mf.raw_object_name(settings.object_prefix, table, ingest_date, "manifest.json")
        manifest_items.append(
            UploadItem(
                table,
                "manifest",
                outcome.manifest_path,
                manifest_name,
                "application/json",
                local_checksums(outcome.manifest_path),
            )
        )
    return items + manifest_items


def _content_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        return "application/gzip"
    if name.endswith(".zip"):
        return "application/zip"
    if name.endswith(".csv"):
        return "text/csv"
    return "application/octet-stream"


def _result_for(
    item: UploadItem, status: str, remote: RemoteObject | None = None, detail: str | None = None
) -> ObjectResult:
    return ObjectResult(
        object_name=item.object_name,
        kind=item.kind,
        table=item.table,
        status=status,
        local_size=item.local.size,
        local_md5_base64=item.local.md5_base64,
        local_crc32c_base64=item.local.crc32c_base64,
        remote_size=remote.size if remote else None,
        remote_md5_base64=remote.md5_base64 if remote else None,
        remote_crc32c_base64=remote.crc32c_base64 if remote else None,
        remote_generation=remote.generation if remote else None,
        detail=detail,
    )


def preflight(
    items: list[UploadItem], store: ObjectStore, *, overwrite: bool
) -> list[tuple[UploadItem, RemoteObject | None, str]]:
    """Look every target up first. Returns (item, remote, decision) with decision in
    new | identical | overwrite | conflict. Nothing is written."""
    decisions = []
    for item in items:
        remote = store.stat(item.object_name)
        if remote is None:
            decisions.append((item, None, "new"))
        elif same_content(item.local, remote):
            decisions.append((item, remote, "identical"))
        elif overwrite:
            decisions.append((item, remote, "overwrite"))
        else:
            decisions.append((item, remote, "conflict"))
    return decisions


def execute_upload(
    decisions: list[tuple[UploadItem, RemoteObject | None, str]],
    store: ObjectStore,
    chunk_size: int,
    progress: Callable[[str], None] = lambda message: None,
) -> list[ObjectResult]:
    results: list[ObjectResult] = []
    for item, remote, decision in decisions:
        if decision == "identical":
            progress(f"skip   {item.object_name} (identical content already in the bucket)")
            results.append(_result_for(item, "skipped_identical", remote))
            continue
        generation = 0 if decision == "new" else int(remote.generation)  # type: ignore[union-attr]
        progress(f"upload {item.object_name} ({item.local.size:,} bytes)")
        try:
            stored = store.upload_file(
                item.local_path,
                item.object_name,
                content_type=item.content_type,
                if_generation_match=generation,
                chunk_size=chunk_size,
            )
        except PreconditionFailed as exc:
            result = _result_for(item, "conflict", remote, f"object changed while landing: {exc}")
            results.append(result)
            raise _LandingAbort(results, "conflict") from exc
        except TransferIntegrityError as exc:
            results.append(_result_for(item, "failed", remote, str(exc)))
            raise _LandingAbort(results, "integrity") from exc
        result = _result_for(item, "overwritten" if decision == "overwrite" else "uploaded", stored)
        result.transfer_verified = same_content(item.local, stored)
        if not result.transfer_verified:
            result.status = "failed"
            result.detail = "store-reported size/checksum differs from the local file after upload"
            results.append(result)
            raise _LandingAbort(results, "integrity")
        results.append(result)
    return results


class _LandingAbort(Exception):
    def __init__(self, results: list[ObjectResult], reason: str):
        super().__init__(reason)
        self.results = results
        self.reason = reason


# --------------------------------------------------------------------------- reporting


def build_report(
    *,
    ingest_date: str,
    mode: str,
    bucket: str | None,
    store_description: str | None,
    outcomes: list[TableOutcome],
    objects: list[ObjectResult],
    status: str,
    strict: bool,
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for outcome in outcomes:
        entry: dict[str, Any] = {
            "artifact": str(outcome.artifact_path) if outcome.artifact_path else None,
            "manifest_path": str(outcome.manifest_path) if outcome.manifest_path else None,
            "manifest_reused": outcome.reused,
            "scan_seconds": outcome.scan_seconds,
            "errors": list(outcome.check.errors) if outcome.check else [],
            "warnings": list(outcome.check.warnings) if outcome.check else [],
            "expectation_check": outcome.check.expectation if outcome.check else None,
        }
        if outcome.manifest:
            entry.update(
                row_count=outcome.manifest["row_count"],
                bytes=outcome.manifest["bytes"],
                md5=outcome.manifest["md5"],
                columns=outcome.manifest["columns"],
            )
        tables[outcome.table] = entry
    return {
        "report_version": 1,
        "ingest_date": ingest_date,
        "mode": mode,
        "bucket": bucket,
        "store": store_description,
        "status": status,
        "strict_expectations": strict,
        "tables": tables,
        "objects": [dataclasses.asdict(o) for o in objects],
        "note": (
            "transfer_verified compares the store-reported size/CRC32C/MD5 with the local file after upload. "
            "It is a transfer-integrity check and is separate from the content MD5 recorded in each manifest."
        ),
    }


def print_validation(outcomes: list[TableOutcome], out=None) -> None:
    out = out if out is not None else sys.stdout
    print("\nManifest summary (measured from the files):", file=out)
    for outcome in outcomes:
        if outcome.manifest:
            m = outcome.manifest
            print(
                f"  {outcome.table:<13} rows={m['row_count']:>11,}  bytes={m['bytes']:>13,}  md5={m['md5']}",
                file=out,
            )
        else:
            print(f"  {outcome.table:<13} NOT AVAILABLE: {outcome.error}", file=out)
        if outcome.check:
            exp = outcome.check.expectation
            if exp:
                verdict = "within" if exp["within_tolerance"] else "OUTSIDE"
                print(
                    f"      expectation (~{exp['expected_rows_approx']:,} per {exp['source']}): {verdict} "
                    f"{exp['tolerance_pct']}% tolerance (deviation {exp['deviation_pct']}%)",
                    file=out,
                )
            for message in outcome.check.errors:
                print(f"      ERROR: {message}", file=out)
            for message in outcome.check.warnings:
                print(f"      warning: {message}", file=out)


# --------------------------------------------------------------------------- orchestration


def validation_failed(outcomes: list[TableOutcome], strict: bool) -> bool:
    for outcome in outcomes:
        if outcome.check is None or outcome.check.errors:
            return True
        if strict and outcome.check.expectation and not outcome.check.expectation["within_tolerance"]:
            return True
    return False


def make_store(bucket: str, project: str | None, local_bucket_dir: str | None) -> ObjectStore:
    if local_bucket_dir:
        return DirectoryObjectStore(resolve_path(local_bucket_dir), bucket)
    return GcsObjectStore(bucket, project=project)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write manifests for the downloaded X5 artifacts and land them unchanged in GCS. "
        "Precedence: CLI flag > env (X5_GCS_BUCKET, GOOGLE_CLOUD_PROJECT) > YAML > default.",
    )
    parser.add_argument("--config", default=str(resolve_path("configs/x5/raw_landing.yaml")))
    parser.add_argument("--data-root", help="override paths.data_root")
    parser.add_argument("--tables", help="comma-separated subset of tables")
    parser.add_argument("--ingest-date", help="YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--bucket", help="bucket name (no gs://); overrides X5_GCS_BUCKET and gcp.bucket")
    parser.add_argument("--project", help="GCP project id; overrides GOOGLE_CLOUD_PROJECT and gcp.project")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="local manifests and validation only; never touch the cloud",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the objects an upload would write; write nothing remotely",
    )
    parser.add_argument(
        "--check-remote", action="store_true", help="with --dry-run: look the objects up in the bucket"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace remote objects whose content differs (default: refuse)",
    )
    parser.add_argument(
        "--strict-expectations",
        action="store_true",
        help="treat row counts outside the configured tolerance as errors",
    )
    parser.add_argument(
        "--reuse-manifests",
        action="store_true",
        help="skip re-scanning an artifact whose local manifest is up to date",
    )
    parser.add_argument(
        "--local-bucket-dir",
        help="rehearse the upload into <dir>/<bucket>/ instead of Cloud Storage (not a cloud check)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(
    argv: list[str] | None = None, *, env: Mapping[str, str] | None = None, store: ObjectStore | None = None
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        config = load_config(args.config)
        settings = LandingSettings.from_config(config)
        root = data_root(config, args.data_root)
        all_tables = configured_tables(config)
        tables = all_tables
        if args.tables:
            tables = tuple(t.strip() for t in args.tables.split(",") if t.strip())
            unknown = [t for t in tables if t not in all_tables]
            if unknown:
                raise ConfigError(f"unknown table(s) {unknown}; configured: {list(all_tables)}")
        ingest_date = resolve_ingest_date(args.ingest_date, config)
        bucket = resolve_bucket(args.bucket, config, env)
        project = resolve_project(args.project, config, env)
        if args.manifest_only and args.dry_run:
            raise ConfigError("--manifest-only and --dry-run are mutually exclusive")
        if args.check_remote and not args.dry_run:
            raise ConfigError("--check-remote only makes sense together with --dry-run")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    mode = "manifest-only" if args.manifest_only else "dry-run" if args.dry_run else "upload"
    outcomes = build_outcomes(
        config, settings, root, tables, ingest_date, reuse=args.reuse_manifests, progress=progress
    )
    print_validation(outcomes)
    report_path = root / settings.manifests_subdir / f"landing_report_ingest_date={ingest_date}.json"

    def finish(status: str, code: int, objects: list[ObjectResult], store_desc: str | None) -> int:
        report = build_report(
            ingest_date=ingest_date,
            mode=mode,
            bucket=bucket,
            store_description=store_desc,
            outcomes=outcomes,
            objects=objects,
            status=status,
            strict=args.strict_expectations,
        )
        atomic_write_json(report_path, report)
        print(f"\nstatus: {status}\nreport: {report_path}")
        return code

    if validation_failed(outcomes, args.strict_expectations):
        return finish("validation_failed", EXIT_VALIDATION, [], None)
    if args.manifest_only:
        return finish("manifests_written", EXIT_OK, [], None)

    items = plan_uploads(outcomes, settings, ingest_date)
    shown_bucket = bucket or _PLACEHOLDER_BUCKET
    print(f"\nPlanned objects ({len(items)}):")
    for item in items:
        print(
            f"  gs://{shown_bucket}/{item.object_name}  <- {item.local_path.name} ({item.local.size:,} bytes)"
        )

    if args.dry_run and not args.check_remote:
        results = [_result_for(item, "would_upload") for item in items]
        if bucket is None:
            print(
                "note: no bucket configured; paths use a placeholder. Set gcp.bucket, X5_GCS_BUCKET or --bucket."
            )
        return finish("dry_run_complete", EXIT_OK, results, None)

    if bucket is None:
        print(
            "error: no bucket configured. Set gcp.bucket in the YAML, X5_GCS_BUCKET, or pass --bucket "
            "(the bucket is created by the infrastructure owner; this script does not create it).",
            file=sys.stderr,
        )
        return finish("bucket_not_configured", EXIT_CLOUD_CONFIG, [], None)
    try:
        active_store = store or make_store(bucket, project, args.local_bucket_dir)
        decisions = preflight(items, active_store, overwrite=args.overwrite)
    except CloudConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return finish("cloud_not_configured", EXIT_CLOUD_CONFIG, [], None)
    except ObjectStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return finish("remote_error", EXIT_REMOTE, [], None)

    conflicts = [d for d in decisions if d[2] == "conflict"]
    if args.dry_run:
        label = {
            "new": "would_upload",
            "identical": "would_skip_identical",
            "overwrite": "would_overwrite",
            "conflict": "conflict",
        }
        results = [_result_for(item, label[decision], remote) for item, remote, decision in decisions]
        for r in results:
            print(f"  {r.status:<22} {r.object_name}")
        code = EXIT_REMOTE if conflicts else EXIT_OK
        return finish(
            "dry_run_conflicts" if conflicts else "dry_run_complete", code, results, active_store.describe()
        )
    if conflicts:
        results = [
            _result_for(item, "conflict", remote, "remote content differs; use --overwrite to replace")
            for item, remote, _ in conflicts
        ]
        for r in results:
            print(f"error: refusing to overwrite differing remote object {r.object_name}", file=sys.stderr)
        print(
            "nothing was uploaded. Re-run with --overwrite only if replacing the remote copy is intended.",
            file=sys.stderr,
        )
        return finish("remote_conflict", EXIT_REMOTE, results, active_store.describe())

    try:
        results = execute_upload(decisions, active_store, settings.upload_chunk_bytes, progress)
    except _LandingAbort as abort:
        print(f"error: landing stopped ({abort.reason}); see report", file=sys.stderr)
        return finish(f"aborted_{abort.reason}", EXIT_REMOTE, abort.results, active_store.describe())
    except CloudConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return finish("cloud_not_configured", EXIT_CLOUD_CONFIG, [], active_store.describe())
    except ObjectStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return finish("remote_error", EXIT_REMOTE, [], active_store.describe())
    return finish("landed", EXIT_OK, results, active_store.describe())


if __name__ == "__main__":
    raise SystemExit(main())

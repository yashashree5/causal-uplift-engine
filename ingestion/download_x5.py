#!/usr/bin/env python3
"""Download the X5 RetailHero uplift dataset (five logical tables) with provenance.

The downloaded *artifacts* (for example ``purchases.csv.gz``) are stored byte-for-byte as
served. They are never rewritten, recompressed or renamed. CSV *members* extracted from an
artifact are separate derived files under ``extracted/`` and are recorded separately.

Examples::

    # Show what would be downloaded, without touching the network or the disk:
    python ingestion/download_x5.py --dry-run

    # Same, plus small metadata requests (no dataset bytes) to check the sources:
    python ingestion/download_x5.py --dry-run --check-remote

    # Download everything (about 0.6 GB compressed) into the configured data root:
    python ingestion/download_x5.py

Exit codes: 0 success, 1 download or source failure, 2 usage/configuration error,
3 integrity verification failure.
"""

from __future__ import annotations

import argparse
import bz2
import dataclasses
import gzip
import hashlib
import logging
import os
import sys
import tarfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in (None, ""):  # allow `python ingestion/download_x5.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from ingestion.x5_common import (  # noqa: E402
    DEFAULT_CONFIG,
    TABLES,
    ConfigError,
    atomic_write_json,
    config_get,
    configured_tables,
    data_root,
    downloads_dir,
    extracted_dir,
    file_digests,
    load_config,
    read_json,
)

EXIT_OK = 0
EXIT_DOWNLOAD_FAILED = 1
EXIT_USAGE = 2
EXIT_VERIFY_FAILED = 3

PROVENANCE_SCHEMA_VERSION = 1

LOG = logging.getLogger("x5.download")

_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK\x03\x04"
_BZ2_MAGIC = b"BZh"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


# --------------------------------------------------------------------------- errors


class DownloadError(Exception):
    """Base class for download failures."""


class TransientSourceError(DownloadError):
    """Network or server problem that may succeed on retry (timeouts, 5xx, short reads)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentSourceError(DownloadError):
    """The source cannot deliver this file (404/403, bad URL). Retrying will not help."""


class VerificationError(DownloadError):
    """Downloaded bytes do not match the expected size/checksum, or are not the expected format."""


# --------------------------------------------------------------------------- plan model


@dataclasses.dataclass
class Candidate:
    """One place a logical table can be downloaded from."""

    table: str
    source_name: str
    kind: str
    url: str
    artifact_name: str
    repo_id: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    expected_size: int | None = None
    expected_sha256: str | None = None
    expected_md5: str | None = None

    def describe(self) -> str:
        return f"{self.source_name}: {self.url}"


@dataclasses.dataclass
class TableResult:
    table: str
    status: str  # downloaded | skipped | adopted | failed | planned
    artifact_path: Path | None = None
    source_name: str | None = None
    error: str | None = None
    error_kind: str | None = None  # download | verify


@dataclasses.dataclass
class DownloadSettings:
    connect_timeout_s: float = 15.0
    read_timeout_s: float = 60.0
    max_retries: int = 5
    backoff_initial_s: float = 2.0
    backoff_factor: float = 2.0
    backoff_max_s: float = 60.0
    chunk_bytes: int = 1024 * 1024
    resume_partial: bool = True
    extract: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DownloadSettings":
        section = config.get("download") or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ConfigError(f"unknown download setting(s): {sorted(unknown)}")
        settings = cls(**{k: v for k, v in section.items() if k in known})
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.max_retries < 0:
            raise ConfigError("download.max_retries must be >= 0")
        if self.connect_timeout_s <= 0 or self.read_timeout_s <= 0:
            raise ConfigError("download timeouts must be positive")
        if self.backoff_initial_s < 0 or self.backoff_factor < 1 or self.backoff_max_s < 0:
            raise ConfigError("download backoff settings are invalid")
        if self.chunk_bytes <= 0:
            raise ConfigError("download.chunk_bytes must be positive")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- plan building


def build_plan(
    config: Mapping[str, Any],
    tables: tuple[str, ...] | None = None,
    source_names: tuple[str, ...] | None = None,
) -> dict[str, list[Candidate]]:
    """Return, per logical table, the ordered download candidates. Pure; no network."""
    wanted = tables or configured_tables(config)
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError("configuration has no 'sources' list")
    if source_names:
        known = {s.get("name") for s in sources}
        missing = [n for n in source_names if n not in known]
        if missing:
            raise ConfigError(f"unknown source name(s) {missing}; configured: {sorted(map(str, known))}")
        sources = [s for s in sources if s.get("name") in source_names]

    plan: dict[str, list[Candidate]] = {table: [] for table in wanted}
    for source in sources:
        name = source.get("name")
        kind = source.get("kind")
        files = source.get("files") or {}
        if not name or kind not in {"huggingface", "http"}:
            raise ConfigError(f"source needs a name and kind in {{huggingface, http}}: {source!r}")
        for table in wanted:
            entry = files.get(table)
            if entry is None:
                continue
            if kind == "huggingface":
                endpoint = str(source.get("endpoint", "https://huggingface.co")).rstrip("/")
                repo_id = source.get("repo_id")
                revision = source.get("revision", "main")
                if not repo_id:
                    raise ConfigError(f"huggingface source {name!r} needs repo_id")
                path_in_repo = str(entry)
                url = f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{path_in_repo}"
                candidate = Candidate(
                    table=table,
                    source_name=name,
                    kind=kind,
                    url=url,
                    artifact_name=Path(path_in_repo).name,
                    repo_id=repo_id,
                    revision=revision,
                    endpoint=endpoint,
                )
                token_env = source.get("auth_token_env")
                if token_env and os.environ.get(token_env):
                    candidate.headers["Authorization"] = f"Bearer {os.environ[token_env]}"
            else:
                spec = entry if isinstance(entry, Mapping) else {"url": entry}
                url = spec.get("url")
                if not url:
                    raise ConfigError(f"source {name!r} table {table!r} needs a url")
                candidate = Candidate(
                    table=table,
                    source_name=name,
                    kind=kind,
                    url=str(url),
                    artifact_name=Path(str(url).split("?", 1)[0]).name,
                    expected_md5=(str(spec["md5"]).lower() if spec.get("md5") else None),
                    expected_sha256=(str(spec["sha256"]).lower() if spec.get("sha256") else None),
                    expected_size=spec.get("size"),
                )
            plan[table].append(candidate)
    return plan


def detect_compression(path: Path) -> str:
    """Identify a container/compression format from magic bytes (not from the file name)."""
    with open(path, "rb") as handle:
        head = handle.read(512)
    if head.startswith(_GZIP_MAGIC):
        return "gzip"
    if head.startswith(_ZIP_MAGIC):
        return "zip"
    if head.startswith(_BZ2_MAGIC):
        return "bzip2"
    if head.startswith(_ZSTD_MAGIC):
        return "zstd"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html", b"<?xml", b"{")):
        return "html-or-error-document"
    return "none"


def _expected_compression(artifact_name: str) -> str:
    lower = artifact_name.lower()
    for suffixes, label in (
        ((".csv.gz", ".gz"), "gzip"),
        ((".zip",), "zip"),
        ((".tar", ".tar.gz", ".tgz"), "tar-or-gzip"),
        ((".bz2",), "bzip2"),
        ((".zst",), "zstd"),
        ((".csv",), "none"),
    ):
        if lower.endswith(suffixes):
            return label
    return "unknown"


# --------------------------------------------------------------------------- downloader


class Downloader:
    """Downloads, verifies and records provenance for logical tables."""

    def __init__(
        self,
        config: Mapping[str, Any],
        root: Path,
        *,
        settings: DownloadSettings | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = _utc_now,
    ):
        self.config = config
        self.root = root
        self.settings = settings or DownloadSettings.from_config(config)
        self.session = session or requests.Session()
        self.sleep = sleep
        self.now = now
        self.downloads = downloads_dir(config, root)
        self.extracted = extracted_dir(config, root)
        self.provenance_path = self.downloads / str(
            config_get(config, "paths.provenance_file", "download_provenance.json")
        )
        self._hf_metadata_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    # ---- remote metadata -------------------------------------------------------------

    def hf_metadata(self, cand: Candidate) -> dict[str, Any] | None:
        """Hub metadata for the candidate's repo revision (resolved ids/sha, file sizes, LFS sha256).

        Returns ``None`` when the metadata endpoint is unavailable; callers then fall back to
        response headers and record that no independent checksum was available.
        """
        key = (cand.endpoint or "", cand.repo_id or "", cand.revision or "")
        if key in self._hf_metadata_cache:
            return self._hf_metadata_cache[key]
        url = f"{cand.endpoint}/api/datasets/{cand.repo_id}/revision/{cand.revision}?blobs=true"
        meta: dict[str, Any] | None = None
        try:
            resp = self.session.get(
                url,
                headers=cand.headers,
                timeout=(self.settings.connect_timeout_s, self.settings.read_timeout_s),
            )
            if resp.status_code == 200:
                payload = resp.json()
                files = {}
                for sibling in payload.get("siblings", []):
                    lfs = sibling.get("lfs") or {}
                    files[sibling.get("rfilename")] = {
                        "size": sibling.get("size"),
                        "sha256": (lfs.get("sha256") or "").lower() or None,
                    }
                meta = {"id": payload.get("id"), "sha": payload.get("sha"), "files": files}
            else:
                LOG.warning("metadata request %s returned HTTP %s", url, resp.status_code)
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("metadata request %s failed: %s", url, exc)
        self._hf_metadata_cache[key] = meta
        return meta

    def apply_hf_expectations(self, cand: Candidate) -> dict[str, Any]:
        """Fill expected size/sha256 from Hub metadata. Returns resolved repo info."""
        info: dict[str, Any] = {"repo_id_resolved": None, "revision_resolved": None}
        if cand.kind != "huggingface":
            return info
        meta = self.hf_metadata(cand)
        if not meta:
            return info
        info["repo_id_resolved"] = meta.get("id")
        info["revision_resolved"] = meta.get("sha")
        prefix = f"{cand.endpoint}/datasets/{cand.repo_id}/resolve/{cand.revision}/"
        path_in_repo = cand.url[len(prefix) :] if cand.url.startswith(prefix) else None
        file_meta = (meta.get("files") or {}).get(path_in_repo or "")
        if file_meta:
            if file_meta.get("size") is not None and cand.expected_size is None:
                cand.expected_size = int(file_meta["size"])
            if file_meta.get("sha256") and cand.expected_sha256 is None:
                cand.expected_sha256 = file_meta["sha256"]
        return info

    # ---- low-level transfer ----------------------------------------------------------

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        delay = min(
            self.settings.backoff_initial_s * (self.settings.backoff_factor**attempt), self.settings.backoff_max_s
        )
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.settings.backoff_max_s))
        return delay

    def _single_attempt(self, cand: Candidate, part: Path, allow_resume: bool) -> dict[str, Any]:
        offset = part.stat().st_size if (allow_resume and part.exists()) else 0
        headers = dict(cand.headers)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        timeout = (self.settings.connect_timeout_s, self.settings.read_timeout_s)
        try:
            resp = self.session.get(cand.url, headers=headers, stream=True, timeout=timeout, allow_redirects=True)
        except (
            requests.exceptions.InvalidURL,
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema,
        ) as exc:
            raise PermanentSourceError(f"invalid URL {cand.url}: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientSourceError(f"{type(exc).__name__}: {exc}") from exc

        with resp:
            status = resp.status_code
            if status == 416:
                # Our partial is longer than/inconsistent with the remote object: start over.
                part.unlink(missing_ok=True)
                raise TransientSourceError("HTTP 416 for resume request; discarded partial file")
            if status in (429, 500, 502, 503, 504) or 500 <= status < 600:
                retry_after = None
                raw = resp.headers.get("Retry-After")
                if raw and raw.isdigit():
                    retry_after = float(raw)
                raise TransientSourceError(f"HTTP {status} from server", retry_after=retry_after)
            if status not in (200, 206):
                raise PermanentSourceError(f"HTTP {status} for {cand.url}")

            mode = "ab"
            if status == 200:
                if offset:
                    LOG.info("server ignored the Range request; restarting %s from byte 0", cand.artifact_name)
                mode = "wb"
                offset = 0
                declared_total = _int_or_none(resp.headers.get("Content-Length"))
            else:
                content_range = resp.headers.get("Content-Range", "")
                start, total = _parse_content_range(content_range)
                if start != offset:
                    part.unlink(missing_ok=True)
                    raise TransientSourceError(f"unexpected Content-Range {content_range!r}; discarded partial file")
                declared_total = total

            part.parent.mkdir(parents=True, exist_ok=True)
            written = offset
            try:
                with open(part, mode) as handle:
                    for chunk in resp.iter_content(chunk_size=self.settings.chunk_bytes):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            except requests.RequestException as exc:
                raise TransientSourceError(
                    f"connection lost after {written} bytes: {type(exc).__name__}: {exc}"
                ) from exc
            except OSError as exc:
                # Local disk problems (disk full, permissions) are not retryable network errors.
                raise PermanentSourceError(f"cannot write {part}: {exc}") from exc

            if declared_total is not None and written != declared_total:
                raise TransientSourceError(f"incomplete transfer: received {written} of {declared_total} bytes")

            return self._response_metadata(resp)

    @staticmethod
    def _response_metadata(resp: requests.Response) -> dict[str, Any]:
        """Collect Hub-style integrity headers from the whole redirect chain."""
        merged: dict[str, str] = {}
        for hop in list(resp.history) + [resp]:
            for name in ("x-repo-commit", "x-linked-etag", "x-linked-size", "etag"):
                if name in hop.headers:
                    merged[name] = hop.headers[name]
        return {"final_url": _strip_query(resp.url), "headers": merged}

    def fetch_with_retries(self, cand: Candidate, part: Path, allow_resume: bool) -> dict[str, Any]:
        attempts = self.settings.max_retries + 1
        last: TransientSourceError | None = None
        for attempt in range(attempts):
            try:
                return self._single_attempt(cand, part, allow_resume)
            except TransientSourceError as exc:
                last = exc
                if attempt + 1 >= attempts:
                    break
                delay = self._backoff_delay(attempt, exc.retry_after)
                LOG.warning(
                    "%s: attempt %d/%d failed (%s); retrying in %.1fs",
                    cand.artifact_name,
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                self.sleep(delay)
        raise TransientSourceError(f"gave up after {attempts} attempt(s): {last}") from last

    # ---- verification ----------------------------------------------------------------

    def verify_file(
        self, cand: Candidate, path: Path, response_meta: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Check size, format and checksums. Returns the verification record or raises."""
        size = path.stat().st_size
        if size == 0:
            raise VerificationError(f"{cand.artifact_name}: downloaded file is empty")
        compression = detect_compression(path)
        if compression == "html-or-error-document":
            raise VerificationError(
                f"{cand.artifact_name}: server returned an HTML/XML/JSON document instead of data; refusing to store it"
            )
        expected_label = _expected_compression(cand.artifact_name)
        if expected_label in {"gzip", "zip", "bzip2", "zstd"} and compression != expected_label:
            raise VerificationError(
                f"{cand.artifact_name}: expected {expected_label} data but the bytes look like {compression!r}"
            )
        digests = file_digests(path, ("md5", "sha256"))

        expected_size = cand.expected_size
        expected_sha = cand.expected_sha256
        sha_origin = "hub_metadata" if cand.kind == "huggingface" else "config"
        if response_meta and cand.kind == "huggingface":
            headers = response_meta.get("headers", {})
            if expected_sha is None and headers.get("x-linked-etag"):
                expected_sha = headers["x-linked-etag"].strip('"').lower()
                sha_origin = "response_header_x-linked-etag"
            if expected_size is None and headers.get("x-linked-size"):
                expected_size = _int_or_none(headers["x-linked-size"])

        checks: list[dict[str, Any]] = []
        failures: list[str] = []
        if expected_size is not None:
            ok = size == int(expected_size)
            checks.append({"check": "size", "expected": int(expected_size), "actual": size, "matched": ok})
            if not ok:
                failures.append(f"size {size} != expected {expected_size}")
        if expected_sha:
            ok = digests["sha256"] == expected_sha
            checks.append(
                {
                    "check": "sha256",
                    "origin": sha_origin,
                    "expected": expected_sha,
                    "actual": digests["sha256"],
                    "matched": ok,
                }
            )
            if not ok:
                failures.append(f"sha256 {digests['sha256']} != expected {expected_sha}")
        if cand.expected_md5:
            ok = digests["md5"] == cand.expected_md5
            checks.append(
                {
                    "check": "md5",
                    "origin": "config",
                    "expected": cand.expected_md5,
                    "actual": digests["md5"],
                    "matched": ok,
                }
            )
            if not ok:
                failures.append(f"md5 {digests['md5']} != expected {cand.expected_md5}")
        if failures:
            raise VerificationError(f"{cand.artifact_name}: integrity check failed: " + "; ".join(failures))

        return {
            "bytes": size,
            "md5": digests["md5"],
            "sha256": digests["sha256"],
            "compression": compression,
            "checks": checks,
            "independently_verified": bool(checks),
        }

    # ---- provenance ------------------------------------------------------------------

    def load_provenance(self) -> dict[str, Any]:
        if self.provenance_path.exists():
            try:
                data = read_json(self.provenance_path)
            except (ValueError, OSError) as exc:
                raise DownloadError(f"provenance file {self.provenance_path} is unreadable: {exc}") from exc
            if isinstance(data, dict) and isinstance(data.get("artifacts"), dict):
                return data
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "dataset": config_get(self.config, "dataset.name", "x5_retailhero_uplift"),
            "artifacts": {},
        }

    def save_provenance(self, provenance: dict[str, Any]) -> None:
        provenance["schema_version"] = PROVENANCE_SCHEMA_VERSION
        provenance["updated_at"] = _iso(self.now())
        atomic_write_json(self.provenance_path, provenance)

    # ---- per-table orchestration -----------------------------------------------------

    def download_table(
        self,
        table: str,
        candidates: list[Candidate],
        provenance: dict[str, Any],
        *,
        force: bool = False,
    ) -> TableResult:
        if not candidates:
            return TableResult(
                table,
                "failed",
                error=f"no configured source provides table '{table}'; nothing was substituted",
                error_kind="download",
            )
        final_name = candidates[0].artifact_name
        existing_record = provenance["artifacts"].get(table)
        final_path = self.downloads / final_name
        if existing_record and existing_record.get("artifact_name") not in (None, final_name):
            final_path = self.downloads / existing_record["artifact_name"]

        if final_path.exists() and not force:
            outcome = self._check_existing(table, final_path, existing_record, candidates)
            if outcome is not None:
                return outcome

        errors: list[str] = []
        verification_failed = False
        for cand in candidates:
            target = self.downloads / cand.artifact_name
            part = target.with_name(target.name + ".part")
            if not self.settings.resume_partial:
                part.unlink(missing_ok=True)
            LOG.info("[%s] downloading from %s", table, cand.describe())
            resolved = self.apply_hf_expectations(cand)
            try:
                record = self._download_candidate(cand, target, part, resolved)
            except VerificationError as exc:
                verification_failed = True
                errors.append(f"{cand.source_name}: {exc}")
                LOG.error("[%s] %s", table, exc)
                continue
            except (PermanentSourceError, TransientSourceError) as exc:
                errors.append(f"{cand.source_name}: {exc}")
                LOG.error("[%s] source %s failed: %s", table, cand.source_name, exc)
                if isinstance(exc, PermanentSourceError):
                    part.unlink(missing_ok=True)
                elif not self.settings.resume_partial:
                    part.unlink(missing_ok=True)
                continue
            provenance["artifacts"][table] = record
            self.save_provenance(provenance)
            return TableResult(table, "downloaded", artifact_path=target, source_name=cand.source_name)

        return TableResult(
            table,
            "failed",
            error="; ".join(errors) or "no source succeeded",
            error_kind="verify" if verification_failed else "download",
        )

    def _download_candidate(
        self, cand: Candidate, target: Path, part: Path, resolved: Mapping[str, Any]
    ) -> dict[str, Any]:
        resumed = False
        for round_index in range(2):
            allow_resume = self.settings.resume_partial and round_index == 0
            if part.exists() and allow_resume and part.stat().st_size > 0:
                resumed = True
            if not allow_resume:
                part.unlink(missing_ok=True)
            if (
                allow_resume
                and cand.expected_size is not None
                and part.exists()
                and part.stat().st_size == cand.expected_size
            ):
                LOG.info(
                    "leftover partial file for %s is already full size; verifying instead of downloading",
                    cand.artifact_name,
                )
                response_meta = {"headers": {}}
            else:
                response_meta = self.fetch_with_retries(cand, part, allow_resume)
            try:
                verification = self.verify_file(cand, part, response_meta)
                break
            except VerificationError:
                part.unlink(missing_ok=True)
                if resumed and round_index == 0:
                    LOG.warning("verification failed after resuming a partial file; retrying once from scratch")
                    resumed = False
                    continue
                raise
        os.replace(part, target)
        revision_resolved = resolved.get("revision_resolved")
        if not revision_resolved and response_meta.get("headers", {}).get("x-repo-commit"):
            revision_resolved = response_meta["headers"]["x-repo-commit"]
        return {
            "logical_table": cand.table,
            "artifact_name": cand.artifact_name,
            "artifact_path": _relpath(target, self.root),
            "retrieval": "downloaded",
            "retrieved_at": _iso(self.now()),
            "source": {
                "name": cand.source_name,
                "kind": cand.kind,
                "url": cand.url,
                "final_url": response_meta.get("final_url"),
                "repo_id_requested": cand.repo_id,
                "repo_id_resolved": resolved.get("repo_id_resolved"),
                "revision_requested": cand.revision,
                "revision_resolved": revision_resolved,
            },
            "bytes": verification["bytes"],
            "md5": verification["md5"],
            "sha256": verification["sha256"],
            "compression": verification["compression"],
            "verification": {
                "independently_verified": verification["independently_verified"],
                "checks": verification["checks"],
            },
            "extracted_members": [],
            "resumed_partial": resumed,
        }

    def _check_existing(
        self,
        table: str,
        final_path: Path,
        record: Mapping[str, Any] | None,
        candidates: list[Candidate],
    ) -> TableResult | None:
        """Decide what to do with an artifact that is already on disk.

        Returns a TableResult to skip the download, or raises/returns a failure when the file
        cannot be trusted. Never silently overwrites.
        """
        size = final_path.stat().st_size
        digests = file_digests(final_path, ("md5", "sha256"))
        if record and record.get("sha256") == digests["sha256"] and record.get("bytes") == size:
            LOG.info("[%s] %s already complete and matches its provenance record; skipping", table, final_path.name)
            return TableResult(
                table, "skipped", artifact_path=final_path, source_name=(record.get("source") or {}).get("name")
            )
        if not record:
            # A file without a provenance record: adopt it only if an independent checksum agrees.
            for cand in candidates:
                self.apply_hf_expectations(cand)
                try:
                    verification = self.verify_file(cand, final_path)
                except VerificationError:
                    continue
                if verification["independently_verified"]:
                    new_record = self._adopted_record(cand, final_path, verification)
                    provenance = self.load_provenance()
                    provenance["artifacts"][table] = new_record
                    self.save_provenance(provenance)
                    LOG.info("[%s] adopted existing %s (independent checksum matched)", table, final_path.name)
                    return TableResult(table, "adopted", artifact_path=final_path, source_name=cand.source_name)
        return TableResult(
            table,
            "failed",
            error=(
                f"{final_path} exists but does not match its provenance record or any expected checksum; "
                "refusing to overwrite it. Re-run with --force to replace it."
            ),
            error_kind="verify",
        )

    def _adopted_record(self, cand: Candidate, path: Path, verification: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "logical_table": cand.table,
            "artifact_name": path.name,
            "artifact_path": _relpath(path, self.root),
            "retrieval": "adopted_existing_file",
            "retrieved_at": None,
            "source": {
                "name": cand.source_name,
                "kind": cand.kind,
                "url": cand.url,
                "repo_id_requested": cand.repo_id,
                "revision_requested": cand.revision,
            },
            "bytes": verification["bytes"],
            "md5": verification["md5"],
            "sha256": verification["sha256"],
            "compression": verification["compression"],
            "verification": {"independently_verified": True, "checks": verification["checks"]},
            "extracted_members": [],
            "resumed_partial": False,
        }

    # ---- extraction ------------------------------------------------------------------

    def extract_table(self, table: str, provenance: dict[str, Any]) -> list[dict[str, Any]]:
        """Write CSV member(s) of a stored artifact under ``extracted/``; never modifies the artifact."""
        record = provenance["artifacts"].get(table)
        if not record:
            raise DownloadError(f"cannot extract '{table}': no artifact recorded")
        artifact = self.downloads / record["artifact_name"]
        compression = record.get("compression", "none")
        self.extracted.mkdir(parents=True, exist_ok=True)

        members: list[dict[str, Any]] = []
        if compression == "none":
            # The artifact already is a CSV; there is nothing to extract.
            record["extracted_members"] = []
            self.save_provenance(provenance)
            return members
        if compression == "gzip" and not tarfile.is_tarfile(artifact):
            name = (
                record["artifact_name"][:-3]
                if record["artifact_name"].lower().endswith(".gz")
                else record["artifact_name"] + ".csv"
            )
            members.append(self._extract_stream(artifact, lambda: gzip.open(artifact, "rb"), name, record))
        elif compression == "bzip2" and not tarfile.is_tarfile(artifact):
            name = (
                record["artifact_name"][:-4]
                if record["artifact_name"].lower().endswith(".bz2")
                else record["artifact_name"] + ".csv"
            )
            members.append(self._extract_stream(artifact, lambda: bz2.open(artifact, "rb"), name, record))
        elif compression == "zip":
            with zipfile.ZipFile(artifact) as archive:
                csv_names = [
                    i.filename for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(".csv")
                ]
                if len(csv_names) != 1:
                    raise DownloadError(
                        f"zip artifact {artifact.name} has {len(csv_names)} CSV members; expected exactly 1"
                    )
                member = csv_names[0]
                members.append(
                    self._extract_stream(
                        artifact, lambda: archive.open(member, "r"), Path(member).name, record, member=member
                    )
                )
        elif tarfile.is_tarfile(artifact):
            with tarfile.open(artifact) as archive:
                csv_infos = [i for i in archive.getmembers() if i.isfile() and i.name.lower().endswith(".csv")]
                if len(csv_infos) != 1:
                    raise DownloadError(
                        f"tar artifact {artifact.name} has {len(csv_infos)} CSV members; expected exactly 1"
                    )
                info = csv_infos[0]
                members.append(
                    self._extract_stream(
                        artifact, lambda: archive.extractfile(info), Path(info.name).name, record, member=info.name
                    )
                )
        else:
            raise DownloadError(f"don't know how to extract {compression!r} artifact {artifact.name}")
        record["extracted_members"] = members
        self.save_provenance(provenance)
        return members

    def _extract_stream(
        self,
        artifact: Path,
        opener: Callable[[], Any],
        out_name: str,
        record: Mapping[str, Any],
        member: str | None = None,
    ) -> dict[str, Any]:
        safe_name = Path(out_name).name  # never trust archive paths (zip-slip)
        target = self.extracted / safe_name
        previous = next(
            (m for m in record.get("extracted_members", []) if m.get("path") == _relpath(target, self.root)), None
        )
        if target.exists() and previous and previous.get("source_artifact_sha256") == record["sha256"]:
            if target.stat().st_size == previous.get("bytes") and file_digests(target, ("md5",))["md5"] == previous.get(
                "md5"
            ):
                LOG.info("extracted member %s already present and matches provenance; skipping", safe_name)
                return previous
        part = target.with_name(target.name + ".part")
        md5 = hashlib.md5()
        total = 0
        try:
            with opener() as source, open(part, "wb") as out:
                for chunk in iter(lambda: source.read(self.settings.chunk_bytes), b""):
                    out.write(chunk)
                    md5.update(chunk)
                    total += len(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(part, target)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return {
            "member": member or safe_name,
            "path": _relpath(target, self.root),
            "kind": "extracted_csv_member",
            "bytes": total,
            "md5": md5.hexdigest(),
            "source_artifact": record["artifact_name"],
            "source_artifact_sha256": record["sha256"],
        }

    # ---- whole run -------------------------------------------------------------------

    def run(
        self,
        plan: Mapping[str, list[Candidate]],
        *,
        force: bool = False,
        extract: bool | None = None,
    ) -> list[TableResult]:
        do_extract = self.settings.extract if extract is None else extract
        self.downloads.mkdir(parents=True, exist_ok=True)
        provenance = self.load_provenance()
        results: list[TableResult] = []
        for table, candidates in plan.items():
            result = self.download_table(table, candidates, provenance, force=force)
            provenance = self.load_provenance()  # adoption/download paths persist their own changes
            if result.status in {"downloaded", "skipped", "adopted"} and do_extract:
                try:
                    self.extract_table(table, provenance)
                except (DownloadError, OSError, zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
                    result = TableResult(table, "failed", error=f"extraction failed: {exc}", error_kind="download")
            results.append(result)
        return results


# --------------------------------------------------------------------------- helpers


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _parse_content_range(value: str) -> tuple[int | None, int | None]:
    # "bytes 100-199/1000"
    try:
        unit, rest = value.split(" ", 1)
        span, total = rest.split("/", 1)
        start = int(span.split("-", 1)[0])
        return start, (int(total) if total != "*" else None)
    except (ValueError, AttributeError):
        return None, None


def _strip_query(url: str) -> str:
    """Drop query strings: redirect targets can carry short-lived signed credentials."""
    return url.split("?", 1)[0]


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- dry run / listing


def describe_plan(
    downloader: Downloader,
    plan: Mapping[str, list[Candidate]],
    *,
    check_remote: bool,
    out=None,
) -> int:
    """Print the download plan. With ``check_remote`` also probes sources using metadata/HEAD only."""
    out = out if out is not None else sys.stdout
    provenance = downloader.load_provenance() if downloader.provenance_path.exists() else {"artifacts": {}}
    unreachable: list[str] = []
    print(f"data root:  {downloader.root}", file=out)
    print(f"artifacts:  {downloader.downloads}", file=out)
    print(f"extracted:  {downloader.extracted} (only with --extract)", file=out)
    print("mode:       dry run; nothing is downloaded or written", file=out)
    for table, candidates in plan.items():
        local = downloader.downloads / (candidates[0].artifact_name if candidates else f"{table}.csv.gz")
        status = f"present ({local.stat().st_size} bytes)" if local.exists() else "absent"
        recorded = "recorded in provenance" if table in provenance["artifacts"] else "no provenance record"
        print(f"\n[{table}] local artifact {local.name}: {status}; {recorded}", file=out)
        if not candidates:
            print("  NO CONFIGURED SOURCE provides this table (a real run would fail)", file=out)
            unreachable.append(table)
            continue
        reachable_any = not check_remote
        for rank, cand in enumerate(candidates, start=1):
            line = f"  {rank}. {cand.describe()}"
            if check_remote:
                note, ok = _probe(downloader, cand)
                line += f"  -> {note}"
                reachable_any = reachable_any or ok
            print(line, file=out)
        if not reachable_any:
            unreachable.append(table)
    if unreachable:
        print(f"\nNo reachable source for: {', '.join(unreachable)}", file=out)
        return EXIT_DOWNLOAD_FAILED
    return EXIT_OK


def _probe(downloader: Downloader, cand: Candidate) -> tuple[str, bool]:
    timeout = (downloader.settings.connect_timeout_s, downloader.settings.read_timeout_s)
    try:
        if cand.kind == "huggingface":
            resolved = downloader.apply_hf_expectations(cand)
            if resolved["revision_resolved"] is None:
                return "metadata unavailable", False
            size = cand.expected_size
            size_text = size if size is not None else "unknown"
            sha_text = "known" if cand.expected_sha256 else "unknown"
            return (
                f"OK revision {resolved['revision_resolved']} (repo {resolved['repo_id_resolved']}), "
                f"size {size_text}, sha256 {sha_text}",
                size is not None,
            )
        resp = downloader.session.get(
            cand.url, headers={**cand.headers, "Range": "bytes=0-0"}, stream=True, timeout=timeout, allow_redirects=True
        )
        with resp:
            ok = resp.status_code in (200, 206)
            return f"HTTP {resp.status_code}", ok
    except requests.RequestException as exc:
        return f"unreachable ({type(exc).__name__})", False


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the X5 RetailHero uplift dataset with provenance. "
        "Precedence: CLI flag > YAML config > built-in default.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (default: configs/x5/x5.yaml)")
    parser.add_argument("--data-root", help="override paths.data_root")
    parser.add_argument("--tables", help=f"comma-separated subset of: {', '.join(TABLES)}")
    parser.add_argument(
        "--source", action="append", dest="sources", help="restrict to a configured source name (repeatable)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; download and write nothing")
    parser.add_argument(
        "--check-remote", action="store_true", help="with --dry-run: probe sources with metadata/range-0 requests only"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing artifact that fails verification")
    extract = parser.add_mutually_exclusive_group()
    extract.add_argument(
        "--extract", dest="extract", action="store_true", default=None, help="also write extracted CSV members"
    )
    extract.add_argument(
        "--no-extract", dest="extract", action="store_false", help="do not extract (default from config)"
    )
    parser.add_argument("--no-resume", action="store_true", help="delete leftover .part files instead of resuming them")
    parser.add_argument("--max-retries", type=int, help="override download.max_retries")
    parser.add_argument("--connect-timeout", type=float, help="override download.connect_timeout_s")
    parser.add_argument("--read-timeout", type=float, help="override download.read_timeout_s")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    try:
        config = load_config(args.config)
        tables = None
        if args.tables:
            tables = tuple(t.strip() for t in args.tables.split(",") if t.strip())
            unknown = [t for t in tables if t not in configured_tables(config)]
            if unknown:
                raise ConfigError(f"unknown table(s) {unknown}; configured: {list(configured_tables(config))}")
        root = data_root(config, args.data_root)
        settings = DownloadSettings.from_config(config)
        if args.no_resume:
            settings.resume_partial = False
        if args.max_retries is not None:
            settings.max_retries = args.max_retries
        if args.connect_timeout is not None:
            settings.connect_timeout_s = args.connect_timeout
        if args.read_timeout is not None:
            settings.read_timeout_s = args.read_timeout
        settings.validate()
        plan = build_plan(config, tables, tuple(args.sources) if args.sources else None)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.check_remote and not args.dry_run:
        print("--check-remote only makes sense together with --dry-run", file=sys.stderr)
        return EXIT_USAGE

    downloader = Downloader(config, root, settings=settings, session=session, sleep=sleep)
    if args.dry_run:
        return describe_plan(downloader, plan, check_remote=args.check_remote)

    try:
        results = downloader.run(plan, force=args.force, extract=args.extract)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DOWNLOAD_FAILED

    failed = [r for r in results if r.status == "failed"]
    for r in results:
        detail = f" from {r.source_name}" if r.source_name else ""
        location = f" -> {r.artifact_path}" if r.artifact_path else ""
        print(f"{r.table}: {r.status}{detail}{location}")
        if r.error:
            print(f"  error: {r.error}", file=sys.stderr)
    used = {r.source_name for r in results if r.source_name}
    first_choice = {t: c[0].source_name for t, c in plan.items() if c}
    if len(used) > 1 or any(r.source_name and r.source_name != first_choice.get(r.table) for r in results):
        print(
            "warning: artifacts come from more than one source (see provenance); the mirrors are not "
            "guaranteed to contain identical rows",
            file=sys.stderr,
        )
    if failed:
        print(
            f"{len(failed)} table(s) failed: {', '.join(r.table for r in failed)}. No substitute data was created.",
            file=sys.stderr,
        )
        return EXIT_VERIFY_FAILED if any(r.error_kind == "verify" for r in failed) else EXIT_DOWNLOAD_FAILED
    print(f"provenance: {downloader.provenance_path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""Manifest generation for raw X5 files: size, exact CSV row count and MD5, with bounded memory.

Definitions
-----------
* **artifact**: the file that is stored and uploaded unchanged, e.g. ``purchases.csv.gz``.
  Its ``md5``/``sha256`` are computed over the artifact bytes (*content hashes*).
* **member**: the CSV inside an artifact (the gzip payload, or the single ``.csv`` member of a zip
  or tar). Its byte size, MD5 and row count describe the decompressed CSV.
* **row count**: the number of CSV *data records*, excluding the header and excluding blank
  lines. It is computed by scanning for record terminators outside quoted fields, so quoted
  newlines do not inflate it (``wc -l`` would). It is not read from any other source.
* **transfer integrity** (CRC32C / GCS-reported MD5 after an upload) is a different check and is
  recorded by ``land_raw.py`` in its upload report, never in this manifest.

Memory is bounded by the read chunk (1 MiB) regardless of file size.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import re
import tarfile
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from ingestion.x5_common import atomic_write_json, file_digests, read_json

MANIFEST_VERSION = 1
ROW_COUNT_METHOD = "csv_records_outside_quotes_excluding_header_and_blank_lines"
CHUNK_BYTES = 1024 * 1024

_BLANK_LINE_RE = re.compile(rb"(?<=\n)\r?\n")
_BARE_CR_RE = re.compile(rb"\r(?!\n)")


class ManifestError(Exception):
    """The file cannot be inspected (unsupported format, malformed CSV, unreadable)."""


# --------------------------------------------------------------------------- CSV scanning


@dataclass
class CsvScan:
    """Result of one streaming pass over decompressed CSV bytes."""

    bytes: int
    md5: str
    sha256: str
    row_count: int
    blank_lines: int
    columns: list[str]
    method: str = ROW_COUNT_METHOD
    warnings: list[str] = field(default_factory=list)


class _RecordCounter:
    """Counts CSV records in a byte stream chunk by chunk (RFC 4180 quoting, ``""`` escapes).

    Only characters *outside* quoted fields are inspected for record terminators, so a quoted
    newline never starts a new record. Doubled quotes (``""``) toggle the state twice and need no
    special handling.
    """

    def __init__(self) -> None:
        self.inside_quotes = False
        self.terminators = 0  # '\n' outside quotes
        self.blank_lines = 0
        self.bare_cr = 0
        self.total_bytes = 0
        self.header_bytes = b""
        self._header_buf = bytearray()
        self._header_done = False
        self._prev = b"\n"  # virtual newline before byte 0 so a leading blank line is detected
        self._pending_cr = False
        self._last_byte = b""

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._last_byte = chunk[-1:]
        self._capture_header(chunk)
        if not self.inside_quotes and b'"' not in chunk:
            self._outside(chunk)
            return
        segments = chunk.split(b'"')
        for index, segment in enumerate(segments):
            if index:  # a quote character preceded this segment: toggle the quoting state
                self.inside_quotes = not self.inside_quotes
                self._boundary()
            if not self.inside_quotes:
                self._outside(segment)

    def _boundary(self) -> None:
        if self._pending_cr:
            self.bare_cr += 1
            self._pending_cr = False
        self._prev = b'"'

    def _outside(self, segment: bytes) -> None:
        if self._pending_cr:
            if not segment.startswith(b"\n"):
                self.bare_cr += 1
            self._pending_cr = False
        if not segment:
            return
        text = self._prev + segment
        self.terminators += segment.count(b"\n")
        offset = len(self._prev)
        self.blank_lines += sum(1 for m in _BLANK_LINE_RE.finditer(text) if m.end() > offset)
        body = segment
        if segment.endswith(b"\r"):
            self._pending_cr = True
            body = segment[:-1]
        self.bare_cr += len(_BARE_CR_RE.findall(body))
        self._prev = text[-2:]

    def _capture_header(self, chunk: bytes) -> None:
        if self._header_done:
            return
        self._header_buf.extend(chunk)
        while True:
            newline = self._header_buf.find(b"\n")
            if newline == -1:
                if len(self._header_buf) > 64 * 1024:
                    raise ManifestError("no header line found in the first 64 KiB")
                return
            header = bytes(self._header_buf[:newline])
            if header.strip(b"\r"):
                break
            del self._header_buf[: newline + 1]  # skip leading blank lines
        if header.count(b'"') % 2:
            raise ManifestError("header contains a quoted newline; unsupported")
        self.header_bytes = header
        self._header_done = True
        self._header_buf = bytearray()

    def finish(self) -> tuple[int, int]:
        """Return (data_row_count, blank_lines). Raises on structurally broken CSV."""
        if self.total_bytes == 0:
            raise ManifestError("file is empty")
        if self.inside_quotes:
            raise ManifestError("unterminated quoted field at end of file")
        if self._pending_cr:
            self.bare_cr += 1
            self._pending_cr = False
        if self.bare_cr:
            raise ManifestError(
                f"{self.bare_cr} bare carriage return(s) outside quotes; unsupported line endings"
            )
        if not self._header_done:  # single-line file without a trailing newline
            self.header_bytes = bytes(self._header_buf)
        records = self.terminators + (0 if self._last_byte == b"\n" else 1) - self.blank_lines
        if records < 1:
            raise ManifestError("no header record found")
        return records - 1, self.blank_lines


def _header_columns(header: bytes) -> list[str]:
    text = header.decode("utf-8-sig").rstrip("\r")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or not rows[0]:
        raise ManifestError("empty header")
    return rows[0]


def scan_csv_stream(stream: BinaryIO, chunk_bytes: int = CHUNK_BYTES) -> CsvScan:
    """Stream decompressed CSV bytes once: size, MD5, SHA-256, exact data row count, header columns."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    counter = _RecordCounter()
    for chunk in iter(lambda: stream.read(chunk_bytes), b""):
        md5.update(chunk)
        sha.update(chunk)
        counter.feed(chunk)
    rows, blank = counter.finish()
    warnings = []
    if blank:
        warnings.append(f"{blank} blank line(s) ignored in the row count")
    return CsvScan(
        bytes=counter.total_bytes,
        md5=md5.hexdigest(),
        sha256=sha.hexdigest(),
        row_count=rows,
        blank_lines=blank,
        columns=_header_columns(counter.header_bytes),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- artifacts


def detect_compression(path: Path) -> str:
    with open(path, "rb") as handle:
        head = handle.read(512)
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"BZh"):
        return "bzip2"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    return "none"


@contextmanager
def open_member(path: Path, compression: str) -> Iterator[tuple[BinaryIO, str | None]]:
    """Yield ``(stream, member_name)``; ``member_name`` is None when the artifact itself is the CSV."""
    if compression == "none":
        with open(path, "rb") as handle:
            yield handle, None
    elif compression == "gzip":
        member = path.name[:-3] if path.name.lower().endswith(".gz") else path.name + ".csv"
        with gzip.open(path, "rb") as handle:
            yield handle, member
    elif compression == "bzip2":
        member = path.name[:-4] if path.name.lower().endswith(".bz2") else path.name + ".csv"
        with bz2.open(path, "rb") as handle:
            yield handle, member
    elif compression == "zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                i.filename
                for i in archive.infolist()
                if not i.is_dir() and i.filename.lower().endswith(".csv")
            ]
            if len(names) != 1:
                raise ManifestError(
                    f"{path.name}: expected exactly one .csv member in the zip, found {len(names)}"
                )
            with archive.open(names[0]) as handle:
                yield handle, names[0]
    elif compression == "tar":
        with tarfile.open(path) as archive:
            infos = [i for i in archive.getmembers() if i.isfile() and i.name.lower().endswith(".csv")]
            if len(infos) != 1:
                raise ManifestError(
                    f"{path.name}: expected exactly one .csv member in the tar, found {len(infos)}"
                )
            handle = archive.extractfile(infos[0])
            if handle is None:
                raise ManifestError(f"{path.name}: cannot read member {infos[0].name}")
            with handle:
                yield handle, infos[0].name
    else:
        raise ManifestError(f"{path.name}: unsupported compression {compression!r}")


@dataclass
class ArtifactInfo:
    path: Path
    size: int
    md5: str
    sha256: str
    compression: str
    member_name: str | None
    scan: CsvScan


def inspect_artifact(path: Path, chunk_bytes: int = CHUNK_BYTES) -> ArtifactInfo:
    """Compute artifact hashes and the member's row count, using bounded memory."""
    if not path.is_file():
        raise ManifestError(f"artifact not found: {path}")
    digests = file_digests(path, ("md5", "sha256"))
    compression = detect_compression(path)
    try:
        with open_member(path, compression) as (stream, member_name):
            scan = scan_csv_stream(stream, chunk_bytes)
    except (OSError, EOFError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise ManifestError(f"{path.name}: cannot read {compression} data: {exc}") from exc
    return ArtifactInfo(
        path=path,
        size=path.stat().st_size,
        md5=digests["md5"],
        sha256=digests["sha256"],
        compression=compression,
        member_name=member_name,
        scan=scan,
    )


# --------------------------------------------------------------------------- manifest assembly


def raw_object_prefix(prefix: str, table: str, ingest_date: str) -> str:
    """``raw/batch/<table>/ingest_date=YYYY-MM-DD`` (no bucket, no trailing slash)."""
    return f"{prefix.strip('/')}/{table}/ingest_date={ingest_date}"


def raw_object_name(prefix: str, table: str, ingest_date: str, filename: str) -> str:
    if "/" in filename or filename in {"", ".", ".."}:
        raise ValueError(f"unsafe object file name: {filename!r}")
    return f"{raw_object_prefix(prefix, table, ingest_date)}/{filename}"


def load_download_provenance(
    downloads_dir: Path, filename: str = "download_provenance.json"
) -> dict[str, Any]:
    path = downloads_dir / filename
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict):
            return data
    return {"artifacts": {}}


def build_table_manifest(
    table: str,
    info: ArtifactInfo,
    *,
    ingest_date: str,
    object_prefix: str,
    provenance_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic manifest for one logical table (no timestamps of generation)."""
    source: dict[str, Any] | None = None
    provenance_check: dict[str, Any] = {"status": "no_download_provenance"}
    if provenance_record:
        source = dict(provenance_record.get("source") or {})
        source["retrieved_at"] = provenance_record.get("retrieved_at")
        source["retrieval"] = provenance_record.get("retrieval")
        matches = (
            provenance_record.get("sha256") == info.sha256 and provenance_record.get("bytes") == info.size
        )
        provenance_check = {
            "status": "matches_artifact" if matches else "MISMATCH_with_artifact",
            "recorded_sha256": provenance_record.get("sha256"),
            "recorded_bytes": provenance_record.get("bytes"),
        }
    object_name = raw_object_name(object_prefix, table, ingest_date, info.path.name)
    artifact = {
        "filename": info.path.name,
        "role": "uploaded_artifact",
        "bytes": info.size,
        "md5": info.md5,
        "sha256": info.sha256,
        "compression": info.compression,
        "object_name": object_name,
    }
    member = None
    if info.member_name is not None:
        member = {
            "filename": info.member_name,
            "role": "csv_member_of_artifact",
            "artifact": info.path.name,
            "bytes": info.scan.bytes,
            "md5": info.scan.md5,
            "sha256": info.scan.sha256,
            "row_count": info.scan.row_count,
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": "x5_retailhero_uplift",
        "logical_table": table,
        "ingest_date": ingest_date,
        "filename": info.path.name,
        "bytes": info.size,
        "md5": info.md5,
        "compression": info.compression,
        "row_count": info.scan.row_count,
        "row_count_method": info.scan.method,
        "row_count_excludes_header": True,
        "columns": info.scan.columns,
        "artifact": artifact,
        "member": member,
        "source": source,
        "download_provenance_check": provenance_check,
        "scan_warnings": info.scan.warnings,
        "notes": (
            "md5/sha256 are content hashes of the stored artifact bytes. Storage transfer integrity "
            "(CRC32C/MD5 reported by GCS) is verified separately and not part of this file."
        ),
    }


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Canonical serialisation: identical manifests give identical bytes (idempotent uploads)."""
    import json

    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- validation


@dataclass
class TableCheck:
    table: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expectation: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_manifest(
    table: str,
    manifest: Mapping[str, Any] | None,
    *,
    required_columns: list[str] | None,
    expected_rows: Mapping[str, Any] | None,
) -> TableCheck:
    """Check structure against configured requirements; report row-count expectation discrepancies.

    Expectations (e.g. the work plan's "about 45.8 million purchases") are *checked against* the
    measured count. They are never copied into a manifest and a miss is a warning, not an error,
    unless the caller asks for strict handling.
    """
    check = TableCheck(table)
    if manifest is None:
        check.errors.append("required file is missing")
        return check
    if manifest["row_count"] == 0:
        check.errors.append("table has zero data rows")
    if manifest["download_provenance_check"]["status"] == "MISMATCH_with_artifact":
        check.errors.append("artifact bytes differ from the download provenance record")
    if required_columns is not None:
        missing = [c for c in required_columns if c not in manifest["columns"]]
        extra = [c for c in manifest["columns"] if c not in required_columns]
        if missing:
            check.errors.append(f"missing required column(s): {missing}")
        if extra:
            check.warnings.append(f"unexpected extra column(s): {extra}")
        if not missing and manifest["columns"] != required_columns:
            check.warnings.append("column order differs from the configured order")
        if len(set(manifest["columns"])) != len(manifest["columns"]):
            check.errors.append("duplicate column names in header")
    if expected_rows:
        expected = int(expected_rows["rows"])
        tolerance = float(expected_rows.get("tolerance_pct", 1.0))
        actual = int(manifest["row_count"])
        deviation_pct = abs(actual - expected) / expected * 100 if expected else float("inf")
        within = deviation_pct <= tolerance
        check.expectation = {
            "expected_rows_approx": expected,
            "tolerance_pct": tolerance,
            "measured_rows": actual,
            "deviation_pct": round(deviation_pct, 4),
            "within_tolerance": within,
            "source": expected_rows.get("source"),
        }
        if not within:
            check.warnings.append(
                f"measured {actual} rows deviates {deviation_pct:.2f}% from the approximate expectation "
                f"{expected} (tolerance {tolerance}%)"
            )
    for warning in manifest.get("scan_warnings", []):
        check.warnings.append(warning)
    return check


def write_table_manifest(manifest: Mapping[str, Any], out_dir: Path) -> Path:
    path = out_dir / manifest["logical_table"] / f"ingest_date={manifest['ingest_date']}" / "manifest.json"
    atomic_write_json(path, manifest)
    return path


ProgressFn = Callable[[str], None]

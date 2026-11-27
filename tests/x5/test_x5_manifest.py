"""Manifest tests: CSV row counting, hashes, compressed artifacts, bounded memory, validation."""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import random
import tarfile
import tracemalloc
import zipfile
from pathlib import Path

import pytest

from ingestion import manifest as mf


def write_csv_bytes(rows, header=("a", "b"), term="\n", trailing=True) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator=term)
    writer.writerow(header)
    writer.writerows(rows)
    text = buf.getvalue()
    if not trailing:
        text = text[: -len(term)]
    return text.encode()


def count(data: bytes, chunk: int = mf.CHUNK_BYTES) -> int:
    return mf.scan_csv_stream(io.BytesIO(data), chunk_bytes=chunk).row_count


# ------------------------------------------------------------------ row counting


def test_simple_counts_exclude_header():
    assert count(b"a,b\n1,2\n3,4\n") == 2
    assert count(b"a,b\n") == 0
    assert count(b"a,b") == 0  # header only, no newline


def test_missing_trailing_newline_and_crlf():
    assert count(b"a,b\n1,2\n3,4") == 2
    assert count(b"a,b\r\n1,2\r\n3,4\r\n") == 2


def test_quoted_newlines_do_not_inflate_the_count():
    data = b'a,b\n"line1\nline2",x\n"q""uote\n",y\n'
    assert data.count(b"\n") == 5  # wc -l would say 5 lines (4 records + ...); real rows = 2
    assert count(data) == 2


def test_blank_lines_are_ignored_and_reported():
    scan = mf.scan_csv_stream(io.BytesIO(b"a,b\n1,2\n\n3,4\n\n"))
    assert scan.row_count == 2 and scan.blank_lines == 2
    assert any("blank" in w for w in scan.warnings)


def test_header_columns_are_parsed_with_quotes_and_bom():
    scan = mf.scan_csv_stream(io.BytesIO('\ufeffclient_id,"a,b",c\n1,2,3\n'.encode()))
    assert scan.columns == ["client_id", "a,b", "c"]


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b'a,b\n"unterminated,1\n', "unterminated"),
        (b"a,b\r1,2\r", "bare carriage return"),
    ],
)
def test_malformed_input_is_rejected(data, message):
    with pytest.raises(mf.ManifestError, match=message):
        mf.scan_csv_stream(io.BytesIO(data))


def test_random_csvs_match_the_csv_module_for_any_chunk_size():
    rng = random.Random(11)
    alphabet = ["a", "1", ",", " ", '"', "\n", "\r\n", '""', "é"]
    for _ in range(600):
        ncols = rng.randint(1, 4)
        header = [f"c{i}" for i in range(ncols)]
        rows = [
            ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6))) for _ in range(ncols)]
            for _ in range(rng.randint(0, 6))
        ]
        term = rng.choice(["\n", "\r\n"])
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator=term)
        writer.writerow(header)
        for row in rows:
            if rng.random() < 0.2:
                buf.write(term)  # blank line
            writer.writerow(row)
        text = buf.getvalue()
        if rng.random() < 0.3:
            text = text[: -len(term)]
        data = text.encode()
        expected = len([r for r in csv.reader(io.StringIO(text, newline="")) if r]) - 1
        assert count(data, chunk=rng.choice([1, 2, 3, 7, 64, 1 << 20])) == expected, repr(text)


def test_chunk_boundaries_inside_quotes_and_crlf():
    data = b'a,b\r\n"x\r\ny",1\r\n2,3\r\n'
    for chunk in range(1, len(data) + 1):
        assert count(data, chunk) == 2, chunk


# ------------------------------------------------------------------ artifacts and hashes


CSV = write_csv_bytes([[str(i), f"v{i}"] for i in range(1, 51)], header=("client_id", "value"))


def make_artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    paths["gzip"] = tmp_path / "t.csv.gz"
    paths["gzip"].write_bytes(gzip.compress(CSV, mtime=0))
    paths["bzip2"] = tmp_path / "t.csv.bz2"
    paths["bzip2"].write_bytes(bz2.compress(CSV))
    paths["zip"] = tmp_path / "t.zip"
    with zipfile.ZipFile(paths["zip"], "w") as archive:
        archive.writestr("inner/t.csv", CSV)
    paths["tar"] = tmp_path / "t.tar"
    with tarfile.open(paths["tar"], "w") as archive:
        info = tarfile.TarInfo("t.csv")
        info.size = len(CSV)
        archive.addfile(info, io.BytesIO(CSV))
    paths["none"] = tmp_path / "t.csv"
    paths["none"].write_bytes(CSV)
    return paths


@pytest.mark.parametrize("kind", ["gzip", "bzip2", "zip", "tar", "none"])
def test_counts_and_hashes_for_every_container(tmp_path, kind):
    path = make_artifacts(tmp_path)[kind]
    info = mf.inspect_artifact(path)
    assert info.compression == kind
    assert info.scan.row_count == 50
    assert info.scan.columns == ["client_id", "value"]
    raw = path.read_bytes()
    assert info.md5 == hashlib.md5(raw).hexdigest()  # artifact content hash
    assert info.sha256 == hashlib.sha256(raw).hexdigest()
    assert info.size == len(raw)
    assert info.scan.md5 == hashlib.md5(CSV).hexdigest()  # member (decompressed) hash
    assert info.scan.bytes == len(CSV)
    if kind == "none":
        assert info.member_name is None
    else:
        assert info.member_name is not None


def test_artifact_and_member_hashes_differ_for_compressed_files(tmp_path):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["gzip"])
    assert info.md5 != info.scan.md5


def test_zip_with_two_csv_members_is_rejected(tmp_path):
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", CSV)
        archive.writestr("b.csv", CSV)
    with pytest.raises(mf.ManifestError, match="exactly one"):
        mf.inspect_artifact(path)


def test_corrupt_gzip_and_missing_file_raise_manifest_error(tmp_path):
    bad = tmp_path / "bad.csv.gz"
    bad.write_bytes(gzip.compress(CSV)[:-20])
    with pytest.raises(mf.ManifestError):
        mf.inspect_artifact(bad)
    with pytest.raises(mf.ManifestError, match="not found"):
        mf.inspect_artifact(tmp_path / "nope.csv.gz")


def test_large_file_is_scanned_with_bounded_memory(tmp_path):
    path = tmp_path / "big.csv.gz"
    line = (
        b"0000abcdef,7e3e2e3984,2018-12-01 07:12:45,10.0,0.0,0.0,0.0,1007.0,54a4a11a29,9a80204f78,2.0,80.0,\n"
    )
    rows = 400_000  # about 38 MB uncompressed
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write(b"client_id,transaction_id,transaction_datetime,a,b,c,d,e,f,g,h,i,j\n")
        for _ in range(rows // 1000):
            handle.write(line * 1000)
    tracemalloc.start()
    info = mf.inspect_artifact(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert info.scan.row_count == rows
    assert info.scan.bytes > 35_000_000
    assert peak < 12 * 1024 * 1024  # far below the ~38 MB decompressed size


# ------------------------------------------------------------------ manifest document


def provenance_for(info: mf.ArtifactInfo, **overrides):
    record = {
        "sha256": info.sha256,
        "bytes": info.size,
        "retrieved_at": "2026-11-26T00:56:41Z",
        "retrieval": "downloaded",
        "source": {
            "name": "huggingface",
            "url": "https://example.test/x.csv.gz",
            "revision_resolved": "abc123",
        },
    }
    record.update(overrides)
    return record


def test_manifest_contains_required_fields_and_relationships(tmp_path):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["gzip"])
    manifest = mf.build_table_manifest(
        "clients",
        info,
        ingest_date="2026-11-25",
        object_prefix="raw/batch",
        provenance_record=provenance_for(info),
    )
    assert manifest["logical_table"] == "clients"
    assert manifest["ingest_date"] == "2026-11-25"
    assert manifest["filename"] == "t.csv.gz"
    assert manifest["bytes"] == info.size
    assert manifest["md5"] == info.md5
    assert manifest["row_count"] == 50 and manifest["row_count_excludes_header"] is True
    assert manifest["compression"] == "gzip"
    assert manifest["source"]["name"] == "huggingface" and manifest["source"]["revision_resolved"] == "abc123"
    assert manifest["artifact"]["object_name"] == "raw/batch/clients/ingest_date=2026-11-25/t.csv.gz"
    assert manifest["member"]["role"] == "csv_member_of_artifact"
    assert manifest["member"]["artifact"] == "t.csv.gz"
    assert manifest["member"]["row_count"] == 50
    assert manifest["download_provenance_check"]["status"] == "matches_artifact"
    assert "generated_at" not in manifest  # deterministic: same inputs -> identical bytes


def test_manifest_bytes_are_deterministic(tmp_path):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["gzip"])
    kwargs = dict(ingest_date="2026-11-25", object_prefix="raw/batch", provenance_record=provenance_for(info))
    a = mf.manifest_bytes(mf.build_table_manifest("clients", info, **kwargs))
    b = mf.manifest_bytes(mf.build_table_manifest("clients", info, **kwargs))
    assert a == b


def test_provenance_mismatch_is_flagged(tmp_path):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["gzip"])
    manifest = mf.build_table_manifest(
        "clients",
        info,
        ingest_date="2026-11-25",
        object_prefix="raw/batch",
        provenance_record=provenance_for(info, sha256="0" * 64),
    )
    assert manifest["download_provenance_check"]["status"] == "MISMATCH_with_artifact"
    check = mf.validate_manifest("clients", manifest, required_columns=None, expected_rows=None)
    assert any("differ from the download provenance" in e for e in check.errors)


def test_uncompressed_artifact_has_no_member(tmp_path):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["none"])
    manifest = mf.build_table_manifest("clients", info, ingest_date="2026-11-25", object_prefix="raw/batch")
    assert manifest["member"] is None and manifest["row_count"] == 50
    assert manifest["download_provenance_check"]["status"] == "no_download_provenance"


# ------------------------------------------------------------------ object names


def test_object_names_follow_the_bucket_layout():
    assert (
        mf.raw_object_prefix("raw/batch/", "purchases", "2026-11-25")
        == "raw/batch/purchases/ingest_date=2026-11-25"
    )
    assert (
        mf.raw_object_name("raw/batch", "purchases", "2026-11-25", "purchases.csv.gz")
        == "raw/batch/purchases/ingest_date=2026-11-25/purchases.csv.gz"
    )
    for bad in ("../x", "a/b", "", ".."):
        with pytest.raises(ValueError):
            mf.raw_object_name("raw/batch", "t", "2026-11-25", bad)


# ------------------------------------------------------------------ validation


def make_manifest(tmp_path, columns=None, rows=50):
    info = mf.inspect_artifact(make_artifacts(tmp_path)["gzip"])
    manifest = mf.build_table_manifest("t", info, ingest_date="2026-11-25", object_prefix="raw/batch")
    if columns is not None:
        manifest["columns"] = columns
    manifest["row_count"] = rows
    return manifest


def test_missing_file_and_zero_rows_are_errors(tmp_path):
    assert mf.validate_manifest("t", None, required_columns=None, expected_rows=None).errors == [
        "required file is missing"
    ]
    check = mf.validate_manifest(
        "t", make_manifest(tmp_path, rows=0), required_columns=None, expected_rows=None
    )
    assert "table has zero data rows" in check.errors


def test_missing_and_extra_columns(tmp_path):
    manifest = make_manifest(tmp_path, columns=["client_id", "oops"])
    check = mf.validate_manifest("t", manifest, required_columns=["client_id", "value"], expected_rows=None)
    assert any("missing required column" in e and "value" in e for e in check.errors)
    assert any("extra column" in w for w in check.warnings)


def test_expectation_is_checked_not_copied(tmp_path):
    manifest = make_manifest(tmp_path, rows=1000)
    ok = mf.validate_manifest(
        "t", manifest, required_columns=None, expected_rows={"rows": 1005, "tolerance_pct": 1.0}
    )
    assert (
        ok.expectation["within_tolerance"] is True
        and ok.expectation["measured_rows"] == 1000
        and not ok.warnings
    )
    bad = mf.validate_manifest(
        "t", manifest, required_columns=None, expected_rows={"rows": 2000, "tolerance_pct": 1.0}
    )
    assert bad.expectation["within_tolerance"] is False and bad.ok  # a discrepancy is a warning
    assert any("deviates" in w for w in bad.warnings)
    assert manifest["row_count"] == 1000  # measured value untouched by the expectation

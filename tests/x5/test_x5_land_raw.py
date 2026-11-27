"""Raw landing tests: path generation, validation gates, idempotent uploads, mocked GCS."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from x5_fixtures import build_tiny_x5, write_landing_config

from ingestion import land_raw
from ingestion import object_store as osx
from ingestion.x5_common import ConfigError

BUCKET = "test-proj-x5"
DATE = "2026-11-25"


@pytest.fixture()
def landed_inputs(tmp_path):
    """Tiny gzip'd X5-shaped tables placed where the downloader would have put them."""
    paths = build_tiny_x5(tmp_path / "data" / "downloads", gzip_files=True)
    config = write_landing_config(tmp_path, bucket=BUCKET)
    return tmp_path, config, paths


def run(config, *extra, env=None, store=None):
    argv = ["--config", str(config), "--ingest-date", DATE, *extra]
    return land_raw.main(argv, env={} if env is None else env, store=store)


def object_names(table: str) -> list[str]:
    return [
        f"raw/batch/{table}/ingest_date={DATE}/{table}.csv.gz",
        f"raw/batch/{table}/ingest_date={DATE}/manifest.json",
    ]


# ------------------------------------------------------------------ settings helpers


def test_ingest_date_default_and_validation():
    from datetime import date

    assert land_raw.resolve_ingest_date(None, {}, today=date(2026, 11, 25)) == "2026-11-25"
    assert land_raw.resolve_ingest_date("2026-01-02", {"raw_landing": {"ingest_date": "2020-01-01"}}) == "2026-01-02"
    assert land_raw.resolve_ingest_date(None, {"raw_landing": {"ingest_date": "2020-01-01"}}) == "2020-01-01"
    for bad in ("2026-13-01", "26-1-1", "tomorrow"):
        with pytest.raises(ConfigError):
            land_raw.resolve_ingest_date(bad, {})


def test_bucket_normalisation_and_precedence():
    assert land_raw.normalise_bucket("gs://my-proj-x5/") == "my-proj-x5"
    assert land_raw.normalise_bucket(None) is None and land_raw.normalise_bucket("  ") is None
    for bad in ("UPPER", "has/slash", "x"):
        with pytest.raises(ConfigError):
            land_raw.normalise_bucket(bad)
    config = {"gcp": {"bucket": "from-yaml-x5"}}
    assert land_raw.resolve_bucket(None, config, {}) == "from-yaml-x5"
    assert land_raw.resolve_bucket(None, config, {"X5_GCS_BUCKET": "from-env-x5"}) == "from-env-x5"
    assert land_raw.resolve_bucket("cli-bucket-x5", config, {"X5_GCS_BUCKET": "from-env-x5"}) == "cli-bucket-x5"
    assert land_raw.resolve_project(None, {"gcp": {"project": "yaml"}}, {"GOOGLE_CLOUD_PROJECT": "env"}) == "env"


def test_shipped_raw_landing_config_loads():
    config = land_raw.load_config(land_raw.resolve_path("configs/x5/raw_landing.yaml"))
    settings = land_raw.LandingSettings.from_config(config)
    assert settings.object_prefix == "raw/batch"
    assert set(settings.required_columns) == {"uplift_train", "uplift_test", "clients", "products", "purchases"}
    assert settings.expectations["purchases"]["rows"] == 45_800_000  # an expectation, not a measurement
    assert config["gcp"]["bucket"] is None  # not provisioned by this repo


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ConfigError):
        land_raw.LandingSettings.from_config({"raw_landing": {"upload_chunk_bytes": 1000}})


# ------------------------------------------------------------------ manifest-only / dry-run


def test_manifest_only_writes_manifests_and_never_touches_cloud(landed_inputs, capsys):
    tmp_path, config, paths = landed_inputs
    with mock.patch.object(land_raw, "make_store", side_effect=AssertionError("cloud touched")):
        code = run(config, "--manifest-only")
    assert code == land_raw.EXIT_OK
    manifest = json.loads((tmp_path / "data/manifests/clients/ingest_date=2026-11-25/manifest.json").read_text())
    clients_gz = paths["clients"]
    assert manifest["filename"] == "clients.csv.gz"
    assert manifest["row_count"] == 200
    assert manifest["md5"] == hashlib.md5(clients_gz.read_bytes()).hexdigest()
    with gzip.open(clients_gz, "rb") as handle:
        assert manifest["member"]["md5"] == hashlib.md5(handle.read()).hexdigest()
    report = json.loads((tmp_path / "data/manifests/landing_report_ingest_date=2026-11-25.json").read_text())
    assert report["status"] == "manifests_written" and report["objects"] == []
    assert report["tables"]["purchases"]["row_count"] == manifest_row_count(tmp_path, "purchases")
    assert "Manifest summary" in capsys.readouterr().out


def manifest_row_count(tmp_path, table):
    return json.loads((tmp_path / f"data/manifests/{table}/ingest_date={DATE}/manifest.json").read_text())["row_count"]


def test_missing_required_file_fails_validation(landed_inputs, capsys):
    tmp_path, config, paths = landed_inputs
    paths["products"].unlink()
    assert run(config, "--manifest-only") == land_raw.EXIT_VALIDATION
    assert "required file for table 'products' not found" in capsys.readouterr().out
    report = json.loads((tmp_path / "data/manifests/landing_report_ingest_date=2026-11-25.json").read_text())
    assert report["status"] == "validation_failed"


def test_missing_required_column_fails_validation(landed_inputs, capsys):
    tmp_path, config, paths = landed_inputs
    with gzip.open(paths["uplift_train"], "wt") as handle:
        handle.write("client_id,treatment_flg\n000012768d,1\n")
    assert run(config, "--manifest-only") == land_raw.EXIT_VALIDATION
    assert "missing required column(s): ['target']" in capsys.readouterr().out


def test_strict_expectations_turn_a_count_miss_into_failure(tmp_path, capsys):
    build_tiny_x5(tmp_path / "data" / "downloads", gzip_files=True)
    config = write_landing_config(
        tmp_path, expectations={"clients": {"rows": 400000, "tolerance_pct": 1.0, "source": "plan"}}
    )
    assert run(config, "--manifest-only") == land_raw.EXIT_OK  # discrepancy is only reported
    out = capsys.readouterr().out
    assert "OUTSIDE" in out and "deviates" in out
    assert run(config, "--manifest-only", "--strict-expectations") == land_raw.EXIT_VALIDATION


def test_dry_run_without_bucket_prints_placeholder_paths_and_uploads_nothing(tmp_path, capsys):
    build_tiny_x5(tmp_path / "data" / "downloads", gzip_files=True)
    config = write_landing_config(tmp_path, bucket=None)
    with mock.patch.object(land_raw, "make_store", side_effect=AssertionError("cloud touched")):
        code = run(config, "--dry-run")
    out = capsys.readouterr().out
    assert code == land_raw.EXIT_OK
    assert f"gs://<BUCKET-NOT-CONFIGURED>/raw/batch/purchases/ingest_date={DATE}/purchases.csv.gz" in out
    assert f"gs://<BUCKET-NOT-CONFIGURED>/raw/batch/purchases/ingest_date={DATE}/manifest.json" in out
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    assert {o["status"] for o in report["objects"]} == {"would_upload"}


def test_dry_run_and_manifest_only_are_exclusive(landed_inputs):
    _, config, _ = landed_inputs
    assert run(config, "--dry-run", "--manifest-only") == land_raw.EXIT_USAGE
    assert run(config, "--check-remote") == land_raw.EXIT_USAGE


def test_reuse_manifests_skips_the_rescan_for_unchanged_artifacts(landed_inputs):
    _, config, _ = landed_inputs
    assert run(config, "--manifest-only") == land_raw.EXIT_OK
    with mock.patch.object(land_raw.mf, "inspect_artifact", side_effect=AssertionError("rescanned")):
        assert run(config, "--manifest-only", "--reuse-manifests") == land_raw.EXIT_OK


# ------------------------------------------------------------------ upload through a directory store


def test_upload_lands_exact_bytes_at_the_contract_paths(landed_inputs, tmp_path_factory):
    tmp_path, config, paths = landed_inputs
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    assert run(config, store=store) == land_raw.EXIT_OK
    for table, local in paths.items():
        artifact, manifest = object_names(table)
        assert (store.base / artifact).read_bytes() == local.read_bytes()  # unchanged bytes
        landed_manifest = json.loads((store.base / manifest).read_text())
        assert landed_manifest["artifact"]["object_name"] == artifact
        assert landed_manifest["logical_table"] == table
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    assert report["status"] == "landed"
    assert {o["status"] for o in report["objects"]} == {"uploaded"}
    assert all(o["transfer_verified"] for o in report["objects"])


def test_rerun_is_idempotent(landed_inputs):
    tmp_path, config, _ = landed_inputs
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    assert run(config, store=store) == land_raw.EXIT_OK
    before = {p: p.stat().st_mtime_ns for p in store.base.rglob("*") if p.is_file()}
    assert run(config, store=store) == land_raw.EXIT_OK
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    assert {o["status"] for o in report["objects"]} == {"skipped_identical"}
    assert before == {p: p.stat().st_mtime_ns for p in store.base.rglob("*") if p.is_file()}


def test_differing_remote_content_is_not_overwritten_silently(landed_inputs, capsys):
    tmp_path, config, paths = landed_inputs
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    existing = store.base / object_names("clients")[0]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"someone else's bytes")
    assert run(config, store=store) == land_raw.EXIT_REMOTE
    assert existing.read_bytes() == b"someone else's bytes"
    # preflight aborts before anything is written: no partial landing
    assert not (store.base / object_names("purchases")[0]).exists()
    assert "refusing to overwrite" in capsys.readouterr().err
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    assert report["status"] == "remote_conflict"


def test_overwrite_flag_replaces_differing_content(landed_inputs):
    tmp_path, config, paths = landed_inputs
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    existing = store.base / object_names("clients")[0]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"stale")
    assert run(config, "--overwrite", store=store) == land_raw.EXIT_OK
    assert existing.read_bytes() == paths["clients"].read_bytes()
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    statuses = {o["object_name"]: o["status"] for o in report["objects"]}
    assert statuses[object_names("clients")[0]] == "overwritten"


def test_dry_run_check_remote_predicts_without_writing(landed_inputs):
    tmp_path, config, paths = landed_inputs
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    run(config, "--tables", "clients", store=store)
    store.base.joinpath(object_names("clients")[0]).write_bytes(b"changed")
    code = run(config, "--dry-run", "--check-remote", store=store)
    assert code == land_raw.EXIT_REMOTE  # conflict predicted
    assert not store.base.joinpath(object_names("purchases")[0]).exists()
    assert store.base.joinpath(object_names("clients")[0]).read_bytes() == b"changed"


class CorruptingStore(osx.DirectoryObjectStore):
    """Pretends the store holds different bytes than were sent."""

    def upload_file(self, local_path, name, **kwargs):
        stored = super().upload_file(local_path, name, **kwargs)
        return osx.RemoteObject(
            stored.name, stored.size, base64.b64encode(b"x" * 16).decode(), "AAAAAA==", stored.generation
        )


def test_transfer_integrity_failure_is_reported_separately_from_content_md5(landed_inputs, capsys):
    tmp_path, config, _ = landed_inputs
    store = CorruptingStore(tmp_path / "bucket_root", BUCKET)
    assert run(config, store=store) == land_raw.EXIT_REMOTE
    assert "aborted_integrity" in capsys.readouterr().out
    report = json.loads((tmp_path / f"data/manifests/landing_report_ingest_date={DATE}.json").read_text())
    failed = [o for o in report["objects"] if o["status"] == "failed"]
    assert failed and failed[0]["transfer_verified"] is False
    # the content MD5 in the manifest is unaffected by the transfer failure
    manifest = json.loads((tmp_path / f"data/manifests/uplift_train/ingest_date={DATE}/manifest.json").read_text())
    assert manifest["md5"] == hashlib.md5((tmp_path / "data/downloads/uplift_train.csv.gz").read_bytes()).hexdigest()


def test_upload_without_bucket_is_a_clear_cloud_config_error(tmp_path, capsys):
    build_tiny_x5(tmp_path / "data" / "downloads", gzip_files=True)
    config = write_landing_config(tmp_path, bucket=None)
    assert run(config) == land_raw.EXIT_CLOUD_CONFIG
    assert "no bucket configured" in capsys.readouterr().err


def test_missing_credentials_is_a_clear_cloud_config_error(landed_inputs, capsys):
    _, config, _ = landed_inputs
    from google.auth import exceptions as auth_exc

    with mock.patch("google.cloud.storage.Client", side_effect=auth_exc.DefaultCredentialsError("none")):
        code = run(config)
    assert code == land_raw.EXIT_CLOUD_CONFIG
    assert "Application Default Credentials" in capsys.readouterr().err


def test_local_bucket_dir_rehearsal(landed_inputs, tmp_path):
    root, config, paths = landed_inputs
    assert run(config, "--local-bucket-dir", str(root / "rehearsal")) == land_raw.EXIT_OK
    assert (root / "rehearsal" / BUCKET / object_names("clients")[0]).read_bytes() == paths["clients"].read_bytes()


# ------------------------------------------------------------------ mocked Cloud Storage client


def fake_blob(name="n", size=3, md5="bWQ1", crc="Y3Jj", generation=5):
    return SimpleNamespace(name=name, size=size, md5_hash=md5, crc32c=crc, generation=generation)


def make_gcs_store(client):
    return osx.GcsObjectStore("some-bucket", client=client)


def test_gcs_stat_returns_none_for_missing_object_and_metadata_for_existing():
    client = mock.MagicMock()
    client.bucket.return_value.get_blob.return_value = None
    assert make_gcs_store(client).stat("a/b") is None
    client.bucket.return_value.get_blob.return_value = fake_blob("a/b", size=10, generation=7)
    remote = make_gcs_store(client).stat("a/b")
    assert remote == osx.RemoteObject("a/b", 10, "bWQ1", "Y3Jj", 7)


def test_gcs_upload_uses_generation_precondition_crc32c_and_no_content_encoding(tmp_path):
    local = tmp_path / "f.csv.gz"
    local.write_bytes(b"abc")
    client = mock.MagicMock()
    bucket = client.bucket.return_value
    bucket.blob.return_value = fake_blob("o/f.csv.gz", size=3)
    bucket.blob.return_value.upload_from_filename = mock.MagicMock()
    store = make_gcs_store(client)
    remote = store.upload_file(
        local, "o/f.csv.gz", content_type="application/gzip", if_generation_match=0, chunk_size=262144
    )
    bucket.blob.assert_called_once_with("o/f.csv.gz", chunk_size=262144)
    kwargs = bucket.blob.return_value.upload_from_filename.call_args.kwargs
    assert kwargs == {"content_type": "application/gzip", "if_generation_match": 0, "checksum": "crc32c"}
    assert "content_encoding" not in kwargs  # bytes must be stored as-is, no transcoding
    assert remote.size == 3


def test_gcs_error_translation(tmp_path):
    from google.api_core import exceptions as gexc

    local = tmp_path / "f"
    local.write_bytes(b"abc")
    client = mock.MagicMock()
    blob = client.bucket.return_value.blob.return_value
    store = make_gcs_store(client)
    blob.upload_from_filename.side_effect = gexc.PreconditionFailed("generation mismatch")
    with pytest.raises(osx.PreconditionFailed):
        store.upload_file(local, "o", content_type="x", if_generation_match=0, chunk_size=262144)
    blob.upload_from_filename.side_effect = gexc.Forbidden("no permission")
    with pytest.raises(osx.CloudConfigError, match="access denied"):
        store.upload_file(local, "o", content_type="x", if_generation_match=0, chunk_size=262144)
    client.bucket.return_value.get_blob.side_effect = gexc.NotFound("bucket gone")
    with pytest.raises(osx.CloudConfigError, match="not found"):
        store.stat("o")

    class DataCorruption(Exception):
        pass

    blob.upload_from_filename.side_effect = DataCorruption("crc mismatch")
    with pytest.raises(osx.TransferIntegrityError):
        store.upload_file(local, "o", content_type="x", if_generation_match=0, chunk_size=262144)


def test_end_to_end_with_mocked_gcs_client_skips_identical_and_uploads_new(landed_inputs):
    tmp_path, config, paths = landed_inputs
    uploaded: dict[str, Path] = {}

    class FakeBlob(SimpleNamespace):
        def upload_from_filename(self, filename, **kwargs):
            uploaded[self.name] = Path(filename)
            checks = osx.local_checksums(Path(filename))
            self.size, self.md5_hash, self.crc32c, self.generation = (
                checks.size,
                checks.md5_base64,
                checks.crc32c_base64,
                1,
            )

    class FakeBucket:
        def get_blob(self, name):
            return None

        def blob(self, name, chunk_size=None):
            return FakeBlob(name=name)

    client = mock.MagicMock()
    client.bucket.return_value = FakeBucket()
    assert run(config, store=make_gcs_store(client)) == land_raw.EXIT_OK
    assert set(uploaded) == {n for t in paths for n in object_names(t)}
    assert uploaded["raw/batch/purchases/ingest_date=2026-11-25/purchases.csv.gz"] == paths["purchases"]

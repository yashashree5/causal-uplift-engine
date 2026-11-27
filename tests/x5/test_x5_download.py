"""Download tests against a local HTTP fixture (no real network, no real dataset)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import zipfile

import pytest
import yaml

from ingestion import download_x5 as dl
from ingestion.x5_common import ConfigError, load_config

CSV = {
    "uplift_train": b"client_id,treatment_flg,target\n000012768d,0,1\n000036f903,1,1\n",
    "uplift_test": b"client_id\n000048b7a6\n000073194a\n",
    "clients": b"client_id,first_issue_date,first_redeem_date,age,gender\n000012768d,2017-08-05 15:40:48,,45,U\n",
    "products": b"product_id,level_1\n0003020d3c,c3d3a8e8c6\n",
    "purchases": b"client_id,transaction_id\n000012768d,7e3e2e3984\n" * 500,
}


def gz(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as handle:
        handle.write(data)
    return buf.getvalue()


ARTIFACTS = {table: gz(body) for table, body in CSV.items()}


def make_config(server, *, tables=None, with_fallback=True, retries=3, extra_download=None, hub_files=None):
    tables = tables or list(CSV)
    hub = {
        "name": "huggingface",
        "kind": "huggingface",
        "endpoint": server.base_url,
        "repo_id": "org/retail",
        "revision": "rev1",
        "files": {t: f"data/{t}.csv.gz" for t in tables},
    }
    if hub_files is not None:
        hub["files"] = hub_files
    sources = [hub]
    if with_fallback:
        sources.append(
            {
                "name": "fallback",
                "kind": "http",
                "files": {
                    t: {"url": f"{server.base_url}/fallback/{t}.csv.gz", "md5": hashlib.md5(ARTIFACTS[t]).hexdigest()}
                    for t in ("uplift_train", "clients", "purchases")
                },
            }
        )
    return {
        "dataset": {"name": "x5_test", "tables": tables},
        "paths": {"data_root": "unused", "downloads_subdir": "downloads", "extracted_subdir": "extracted"},
        "download": {
            "connect_timeout_s": 2,
            "read_timeout_s": 5,
            "max_retries": retries,
            "backoff_initial_s": 1.0,
            "backoff_factor": 2.0,
            "backoff_max_s": 10.0,
            "chunk_bytes": 1024,
            "resume_partial": True,
            "extract": False,
            **(extra_download or {}),
        },
        "sources": sources,
    }


@pytest.fixture()
def hub(server):
    server.add_hub_repo("org/retail", "rev1", {f"data/{t}.csv.gz": body for t, body in ARTIFACTS.items()})
    return server


def make_downloader(config, root, session, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return dl.Downloader(config, root, session=session, sleep=sleeps.append)


# ------------------------------------------------------------------ plan / dry run


def test_plan_orders_sources_and_resolves_urls(server):
    config = make_config(server)
    plan = dl.build_plan(config)
    assert list(plan) == list(CSV)
    purchases = plan["purchases"]
    assert [c.source_name for c in purchases] == ["huggingface", "fallback"]
    assert purchases[0].url == f"{server.base_url}/datasets/org/retail/resolve/rev1/data/purchases.csv.gz"
    assert purchases[1].expected_md5 == hashlib.md5(ARTIFACTS["purchases"]).hexdigest()
    # fallback has no products/uplift_test, like the scikit-uplift loader
    assert [c.source_name for c in plan["products"]] == ["huggingface"]


def test_plan_rejects_unknown_source_name(server):
    with pytest.raises(ConfigError, match="unknown source"):
        dl.build_plan(make_config(server), source_names=("nope",))


def test_shipped_config_declares_all_five_tables_and_verified_sources():
    config = load_config()
    plan = dl.build_plan(config)
    assert set(plan) == {"uplift_train", "uplift_test", "clients", "products", "purchases"}
    for table, candidates in plan.items():
        assert candidates[0].source_name == "huggingface", table
        assert candidates[0].artifact_name == f"{table}.csv.gz"
        assert "/resolve/7744156a4f89f4828607921e8c9668f04801f4de/data/" in candidates[0].url
    # the S3 fallback only covers what sklift's fetch_x5 loads
    assert {t for t, c in plan.items() if len(c) > 1} == {"uplift_train", "clients", "purchases"}
    assert all(c.expected_md5 for t, cs in plan.items() for c in cs[1:])


def test_dry_run_makes_no_requests_and_writes_nothing(server, tmp_path, session, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(make_config(server)))
    root = tmp_path / "data"
    code = dl.main(["--config", str(config_path), "--data-root", str(root), "--dry-run"], session=session)
    out = capsys.readouterr().out
    assert code == dl.EXIT_OK
    assert "dry run" in out and "purchases" in out
    assert server.requests == []
    assert not root.exists()


def test_dry_run_check_remote_uses_metadata_only(hub, tmp_path, session, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(make_config(hub, with_fallback=False)))
    code = dl.main(
        ["--config", str(config_path), "--data-root", str(tmp_path / "d"), "--dry-run", "--check-remote"],
        session=session,
    )
    out = capsys.readouterr().out
    assert code == dl.EXIT_OK
    assert "revision" in out and "sha256 known" in out
    assert not any(p.startswith("/datasets/") for _, p, _ in hub.requests)  # no dataset bytes requested


def test_check_remote_requires_dry_run(server, tmp_path, session):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(make_config(server)))
    assert (
        dl.main(["--config", str(config_path), "--data-root", str(tmp_path), "--check-remote"], session=session)
        == dl.EXIT_USAGE
    )


# ------------------------------------------------------------------ successful download


def test_download_preserves_bytes_and_records_provenance(hub, tmp_path, session):
    config = make_config(hub, with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    results = dler.run(dl.build_plan(config))
    assert [r.status for r in results] == ["downloaded"] * 5
    for table, body in ARTIFACTS.items():
        assert (tmp_path / "downloads" / f"{table}.csv.gz").read_bytes() == body  # byte-for-byte
    prov = json.loads(dler.provenance_path.read_text())
    rec = prov["artifacts"]["purchases"]
    assert rec["artifact_name"] == "purchases.csv.gz"
    assert rec["compression"] == "gzip"
    assert rec["bytes"] == len(ARTIFACTS["purchases"])
    assert rec["md5"] == hashlib.md5(ARTIFACTS["purchases"]).hexdigest()
    assert rec["sha256"] == hashlib.sha256(ARTIFACTS["purchases"]).hexdigest()
    assert rec["source"]["url"].endswith("/datasets/org/retail/resolve/rev1/data/purchases.csv.gz")
    assert rec["source"]["revision_requested"] == "rev1"
    assert rec["source"]["revision_resolved"] == "c0ffee" * 6 + "abcd"
    assert rec["retrieved_at"].endswith("Z")
    assert rec["verification"]["independently_verified"] is True
    assert rec["extracted_members"] == []
    assert not list((tmp_path / "downloads").glob("*.part"))


def test_hub_id_redirect_is_recorded(server, tmp_path, session):
    server.add_hub_repo(
        "org/retail", "rev1", {"data/uplift_train.csv.gz": ARTIFACTS["uplift_train"]}, resolved_id="new-org/retail"
    )
    config = make_config(server, tables=["uplift_train"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    dler.run(dl.build_plan(config))
    rec = json.loads(dler.provenance_path.read_text())["artifacts"]["uplift_train"]["source"]
    assert rec["repo_id_requested"] == "org/retail"
    assert rec["repo_id_resolved"] == "new-org/retail"


def test_query_string_of_redirect_target_is_not_persisted(server, tmp_path, session):
    server.add_hub_repo("org/retail", "rev1", {"data/uplift_train.csv.gz": ARTIFACTS["uplift_train"]})
    real = "/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz"
    body = server.routes.pop(real)
    server.routes["/cdn/blob"] = body
    server.add(real, redirect_to=f"{server.base_url}/cdn/blob?Signature=SECRET&Expires=1")
    config = make_config(server, tables=["uplift_train"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    dler.run(dl.build_plan(config))
    text = dler.provenance_path.read_text()
    assert "SECRET" not in text
    assert json.loads(text)["artifacts"]["uplift_train"]["source"]["final_url"].endswith("/cdn/blob")


# ------------------------------------------------------------------ retries / failure handling


def test_transient_errors_are_retried_with_exponential_backoff(hub, tmp_path, session):
    route = hub.routes["/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz"]
    route.status_sequence = [503, 500]
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    sleeps: list[float] = []
    results = make_downloader(config, tmp_path, session, sleeps).run(dl.build_plan(config))
    assert results[0].status == "downloaded"
    assert sleeps == [1.0, 2.0]


def test_retry_after_header_is_honoured_and_capped(hub, tmp_path, session):
    route = hub.routes["/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz"]
    route.status_sequence = [429]
    route.retry_after = "7"
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    sleeps: list[float] = []
    make_downloader(config, tmp_path, session, sleeps).run(dl.build_plan(config))
    assert sleeps == [7.0]


def test_retries_exhausted_fails_with_nonzero_exit_and_no_artifact(hub, tmp_path, session, capsys):
    hub.routes["/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz"].status_sequence = [503] * 10
    config = make_config(hub, tables=["uplift_train"], with_fallback=False, retries=2)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    sleeps: list[float] = []
    code = dl.main(
        ["--config", str(config_path), "--data-root", str(tmp_path / "d")], session=session, sleep=sleeps.append
    )
    err = capsys.readouterr().err
    assert code == dl.EXIT_DOWNLOAD_FAILED
    assert "gave up after 3 attempt(s)" in err
    assert "No substitute data was created" in err
    assert not (tmp_path / "d" / "downloads" / "uplift_train.csv.gz").exists()
    assert len(sleeps) == 2


def test_permanent_error_is_not_retried(server, tmp_path, session):
    config = make_config(server, tables=["uplift_train"], with_fallback=False)  # hub has no routes -> 404
    sleeps: list[float] = []
    results = make_downloader(config, tmp_path, session, sleeps).run(dl.build_plan(config))
    assert results[0].status == "failed"
    assert "HTTP 404" in results[0].error
    assert sleeps == []
    assert server.count("/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz") == 1


def test_connection_refused_is_transient_and_reported(tmp_path, session):
    config = {
        "dataset": {"tables": ["uplift_train"]},
        "paths": {"data_root": "x"},
        "download": {"max_retries": 1, "backoff_initial_s": 0.0, "connect_timeout_s": 1, "read_timeout_s": 1},
        "sources": [
            {
                "name": "dead",
                "kind": "http",
                "files": {"uplift_train": {"url": "http://127.0.0.1:9/uplift_train.csv.gz"}},
            }
        ],
    }
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "failed"
    assert "gave up after 2 attempt(s)" in results[0].error


# ------------------------------------------------------------------ source fallback


def test_falls_back_to_second_source_when_first_fails(server, tmp_path, session):
    server.add("/fallback/uplift_train.csv.gz", ARTIFACTS["uplift_train"])  # hub route absent -> 404
    config = make_config(server, tables=["uplift_train"])
    dler = make_downloader(config, tmp_path, session)
    results = dler.run(dl.build_plan(config))
    assert results[0].status == "downloaded" and results[0].source_name == "fallback"
    assert (tmp_path / "downloads" / "uplift_train.csv.gz").read_bytes() == ARTIFACTS["uplift_train"]
    rec = json.loads(dler.provenance_path.read_text())["artifacts"]["uplift_train"]
    assert rec["source"]["name"] == "fallback"
    assert rec["verification"]["checks"][0]["check"] == "md5"


def test_table_missing_from_every_source_fails_without_substitution(server, tmp_path, session, capsys):
    server.add("/fallback/uplift_train.csv.gz", ARTIFACTS["uplift_train"])
    config = make_config(server, tables=["products"])
    config["sources"] = [config["sources"][1]]  # fallback only; it has no products table
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    code = dl.main(["--config", str(config_path), "--data-root", str(tmp_path / "d")], session=session)
    assert code == dl.EXIT_DOWNLOAD_FAILED
    assert "no configured source provides table 'products'" in capsys.readouterr().err
    assert not list((tmp_path / "d").rglob("*.csv*"))


def test_fallback_md5_mismatch_is_rejected(server, tmp_path, session):
    server.add("/fallback/uplift_train.csv.gz", gz(b"client_id,treatment_flg,target\nwrong,0,0\n"))
    config = make_config(server, tables=["uplift_train"])
    config["sources"] = [config["sources"][1]]
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "failed" and results[0].error_kind == "verify"
    assert not (tmp_path / "downloads" / "uplift_train.csv.gz").exists()


# ------------------------------------------------------------------ partial downloads


def test_truncated_transfer_is_resumed_with_range_request(server, tmp_path, session):
    import random

    rng = random.Random(1)
    big = gz("\n".join(rng.randbytes(16).hex() for _ in range(6000)).encode())  # >> chunk_bytes
    assert len(big) > 20_000
    path = "/datasets/org/retail/resolve/rev1/data/purchases.csv.gz"
    server.add_hub_repo("org/retail", "rev1", {"data/purchases.csv.gz": big})
    body = big
    hub = server
    hub.routes[path].truncate_first = 1
    config = make_config(hub, tables=["purchases"], with_fallback=False)
    sleeps: list[float] = []
    results = make_downloader(config, tmp_path, session, sleeps).run(dl.build_plan(config))
    assert results[0].status == "downloaded"
    assert (tmp_path / "downloads" / "purchases.csv.gz").read_bytes() == body
    ranges = [r for m, p, r in hub.requests if p == path and m == "GET"]
    assert ranges[0] is None and ranges[1] and ranges[1].startswith("bytes=") and not ranges[1].startswith("bytes=0-")
    assert len(sleeps) == 1


def test_server_without_range_support_restarts_cleanly(server, tmp_path, session):
    import random

    big = gz("\n".join(random.Random(2).randbytes(16).hex() for _ in range(6000)).encode())
    server.add_hub_repo("org/retail", "rev1", {"data/purchases.csv.gz": big})
    hub = server
    path = "/datasets/org/retail/resolve/rev1/data/purchases.csv.gz"
    hub.routes[path].truncate_first = 1
    hub.routes[path].supports_range = False
    config = make_config(hub, tables=["purchases"], with_fallback=False)
    make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert (tmp_path / "downloads" / "purchases.csv.gz").read_bytes() == big


def test_leftover_part_file_is_resumed_on_rerun(hub, tmp_path, session):
    body = ARTIFACTS["purchases"]
    part = tmp_path / "downloads" / "purchases.csv.gz.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(body[:100])
    config = make_config(hub, tables=["purchases"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    dler.run(dl.build_plan(config))
    assert (tmp_path / "downloads" / "purchases.csv.gz").read_bytes() == body
    assert not part.exists()
    assert json.loads(dler.provenance_path.read_text())["artifacts"]["purchases"]["resumed_partial"] is True


def test_corrupt_leftover_part_is_discarded_and_download_restarts(hub, tmp_path, session):
    body = ARTIFACTS["purchases"]
    part = tmp_path / "downloads" / "purchases.csv.gz.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\x1f\x8b" + b"garbage" * 5)  # wrong prefix: the resumed file will fail its checksum
    config = make_config(hub, tables=["purchases"], with_fallback=False)
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "downloaded"
    assert (tmp_path / "downloads" / "purchases.csv.gz").read_bytes() == body
    assert not part.exists()


def test_no_resume_flag_removes_stale_partial(hub, tmp_path, session):
    part = tmp_path / "downloads" / "purchases.csv.gz.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"stale")
    config = make_config(hub, tables=["purchases"], with_fallback=False, extra_download={"resume_partial": False})
    make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert (tmp_path / "downloads" / "purchases.csv.gz").read_bytes() == ARTIFACTS["purchases"]
    assert not part.exists()
    path = "/datasets/org/retail/resolve/rev1/data/purchases.csv.gz"
    assert [r for m, p, r in hub.requests if p == path and m == "GET"] == [None]


def test_failed_download_never_leaves_a_final_artifact(hub, tmp_path, session):
    path = "/datasets/org/retail/resolve/rev1/data/purchases.csv.gz"
    hub.routes[path].truncate_first = 99  # every response is cut off
    config = make_config(hub, tables=["purchases"], with_fallback=False, retries=2)
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "failed"
    assert not (tmp_path / "downloads" / "purchases.csv.gz").exists()
    prov_path = tmp_path / "downloads" / "download_provenance.json"
    assert not prov_path.exists() or "purchases" not in json.loads(prov_path.read_text())["artifacts"]


# ------------------------------------------------------------------ integrity


def test_checksum_mismatch_fails_with_verification_exit_code(server, tmp_path, session, capsys):
    server.add_hub_repo(
        "org/retail",
        "rev1",
        {"data/uplift_train.csv.gz": ARTIFACTS["uplift_train"]},
        lfs_sha256={"data/uplift_train.csv.gz": "0" * 64},
    )
    config = make_config(server, tables=["uplift_train"], with_fallback=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    code = dl.main(["--config", str(config_path), "--data-root", str(tmp_path / "d")], session=session)
    assert code == dl.EXIT_VERIFY_FAILED
    assert "integrity check failed" in capsys.readouterr().err
    assert not list((tmp_path / "d" / "downloads").glob("uplift_train.csv.gz*"))


def test_html_error_page_served_with_200_is_rejected(server, tmp_path, session):
    server.add("/fallback/uplift_train.csv.gz", b"<html><body>Access denied</body></html>")
    config = make_config(server, tables=["uplift_train"])
    config["sources"] = [config["sources"][1]]
    config["sources"][0]["files"]["uplift_train"].pop("md5")
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "failed" and "HTML" in results[0].error


def test_expected_size_mismatch_is_rejected(server, tmp_path, session):
    body = ARTIFACTS["uplift_train"]
    server.add_hub_repo("org/retail", "rev1", {"data/uplift_train.csv.gz": body})
    meta_path = "/api/datasets/org/retail/revision/rev1"
    payload = json.loads(server.routes[meta_path].body)
    payload["siblings"][0]["size"] = len(body) + 1
    server.routes[meta_path].body = json.dumps(payload).encode()
    config = make_config(server, tables=["uplift_train"], with_fallback=False)
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].error_kind == "verify" and "size" in results[0].error


def test_metadata_unavailable_falls_back_to_response_headers(hub, tmp_path, session):
    hub.routes.pop("/api/datasets/org/retail/revision/rev1")
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    assert dler.run(dl.build_plan(config))[0].status == "downloaded"
    rec = json.loads(dler.provenance_path.read_text())["artifacts"]["uplift_train"]
    sha_check = next(c for c in rec["verification"]["checks"] if c["check"] == "sha256")
    assert sha_check["origin"] == "response_header_x-linked-etag" and sha_check["matched"]
    assert rec["source"]["revision_resolved"] == "c0ffee" * 6 + "abcd"


# ------------------------------------------------------------------ reruns


def test_rerun_skips_complete_artifacts_without_fetching(hub, tmp_path, session):
    config = make_config(hub, with_fallback=False)
    make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    before = len(hub.requests)
    artifact = tmp_path / "downloads" / "purchases.csv.gz"
    mtime = artifact.stat().st_mtime_ns
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert [r.status for r in results] == ["skipped"] * 5
    assert len(hub.requests) == before  # not even metadata requests
    assert artifact.stat().st_mtime_ns == mtime


def test_existing_file_without_provenance_is_adopted_only_if_checksum_matches(hub, tmp_path, session):
    target = tmp_path / "downloads" / "uplift_train.csv.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(ARTIFACTS["uplift_train"])
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    assert dler.run(dl.build_plan(config))[0].status == "adopted"
    rec = json.loads(dler.provenance_path.read_text())["artifacts"]["uplift_train"]
    assert rec["retrieval"] == "adopted_existing_file" and rec["retrieved_at"] is None


def test_corrupt_existing_artifact_is_not_overwritten_without_force(hub, tmp_path, session):
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    target = tmp_path / "downloads" / "uplift_train.csv.gz"
    target.write_bytes(b"corrupted")
    results = make_downloader(config, tmp_path, session).run(dl.build_plan(config))
    assert results[0].status == "failed" and "--force" in results[0].error
    assert target.read_bytes() == b"corrupted"  # untouched
    forced = make_downloader(config, tmp_path, session).run(dl.build_plan(config), force=True)
    assert forced[0].status == "downloaded"
    assert target.read_bytes() == ARTIFACTS["uplift_train"]


# ------------------------------------------------------------------ extraction


def test_extract_gzip_keeps_artifact_and_records_member_separately(hub, tmp_path, session):
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    dler = make_downloader(config, tmp_path, session)
    dler.run(dl.build_plan(config), extract=True)
    artifact = tmp_path / "downloads" / "uplift_train.csv.gz"
    assert artifact.read_bytes() == ARTIFACTS["uplift_train"]
    extracted = tmp_path / "extracted" / "uplift_train.csv"
    assert extracted.read_bytes() == CSV["uplift_train"]
    rec = json.loads(dler.provenance_path.read_text())["artifacts"]["uplift_train"]
    (member,) = rec["extracted_members"]
    assert member["kind"] == "extracted_csv_member"
    assert member["source_artifact"] == "uplift_train.csv.gz"
    assert member["source_artifact_sha256"] == rec["sha256"]
    assert member["md5"] == hashlib.md5(CSV["uplift_train"]).hexdigest()
    assert member["md5"] != rec["md5"]  # artifact hash and member hash are different things


def test_extract_is_idempotent(hub, tmp_path, session):
    config = make_config(hub, tables=["uplift_train"], with_fallback=False)
    make_downloader(config, tmp_path, session).run(dl.build_plan(config), extract=True)
    extracted = tmp_path / "extracted" / "uplift_train.csv"
    mtime = extracted.stat().st_mtime_ns
    make_downloader(config, tmp_path, session).run(dl.build_plan(config), extract=True)
    assert extracted.stat().st_mtime_ns == mtime


def test_extract_zip_and_tar_members_and_path_sanitising(server, tmp_path, session):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as archive:
        archive.writestr("../evil/clients.csv", CSV["clients"])
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as archive:
        info = tarfile.TarInfo("nested/dir/products.csv")
        info.size = len(CSV["products"])
        archive.addfile(info, io.BytesIO(CSV["products"]))
    server.add("/a/clients.zip", zbuf.getvalue())
    server.add("/a/products.tar", tbuf.getvalue())
    config = {
        "dataset": {"tables": ["clients", "products"]},
        "paths": {"data_root": "x"},
        "download": {"max_retries": 0},
        "sources": [
            {
                "name": "arch",
                "kind": "http",
                "files": {
                    "clients": {"url": f"{server.base_url}/a/clients.zip"},
                    "products": {"url": f"{server.base_url}/a/products.tar"},
                },
            }
        ],
    }
    dler = make_downloader(config, tmp_path, session)
    results = dler.run(dl.build_plan(config), extract=True)
    assert [r.status for r in results] == ["downloaded", "downloaded"]
    prov = json.loads(dler.provenance_path.read_text())["artifacts"]
    assert prov["clients"]["compression"] == "zip" and prov["products"]["compression"] == "tar"
    assert (tmp_path / "extracted" / "clients.csv").read_bytes() == CSV["clients"]  # not ../evil/
    assert not (tmp_path.parent / "evil").exists()
    assert (tmp_path / "extracted" / "products.csv").read_bytes() == CSV["products"]
    assert prov["clients"]["extracted_members"][0]["member"] == "../evil/clients.csv"


# ------------------------------------------------------------------ CLI / config


def test_cli_rejects_unknown_table_and_bad_config(server, tmp_path, session, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(make_config(server)))
    assert dl.main(["--config", str(config_path), "--tables", "bogus", "--dry-run"], session=session) == dl.EXIT_USAGE
    assert dl.main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"], session=session) == dl.EXIT_USAGE
    assert "configuration error" in capsys.readouterr().err


def test_cli_overrides_beat_config_values(server, tmp_path, session):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(make_config(server, retries=5)))
    server.add("/datasets/org/retail/resolve/rev1/data/uplift_train.csv.gz", status_sequence=[503] * 20)
    sleeps: list[float] = []
    code = dl.main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(tmp_path / "cli_root"),
            "--tables",
            "uplift_train",
            "--source",
            "huggingface",
            "--max-retries",
            "1",
        ],
        session=session,
        sleep=sleeps.append,
    )
    assert code == dl.EXIT_DOWNLOAD_FAILED
    assert len(sleeps) == 1  # CLI --max-retries 1 overrides max_retries: 5
    assert (tmp_path / "cli_root" / "downloads").exists()  # CLI --data-root overrides paths.data_root


def test_invalid_download_settings_are_rejected(server):
    config = make_config(server, extra_download={"max_retries": -1})
    with pytest.raises(ConfigError):
        dl.DownloadSettings.from_config(config)
    config = make_config(server, extra_download={"bogus_key": 1})
    with pytest.raises(ConfigError, match="unknown download setting"):
        dl.DownloadSettings.from_config(config)


def test_mixed_source_run_prints_warning(server, tmp_path, session, capsys):
    server.add_hub_repo("org/retail", "rev1", {"data/clients.csv.gz": ARTIFACTS["clients"]})
    server.add("/fallback/uplift_train.csv.gz", ARTIFACTS["uplift_train"])  # hub has no uplift_train -> fallback
    config = make_config(server, tables=["uplift_train", "clients"])
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert dl.main(["--config", str(config_path), "--data-root", str(tmp_path / "d")], session=session) == dl.EXIT_OK
    assert "more than one source" in capsys.readouterr().err

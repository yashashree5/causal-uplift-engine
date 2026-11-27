"""Client-sample tests: reproducibility, order independence, joins, products, boundaries, upload."""

from __future__ import annotations

import csv
import gzip
import json
import random
from pathlib import Path

import pytest
import yaml
from x5_fixtures import build_tiny_x5

from ingestion import land_raw
from ingestion import object_store as osx
from ingestion import sample_clients as sc

BUCKET = "test-proj-x5"


def read_gz_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_config(tmp_path: Path, **sampling) -> Path:
    config = {
        "dataset": {"tables": ["uplift_train", "uplift_test", "clients", "products", "purchases"]},
        "paths": {
            "data_root": str(tmp_path / "data"),
            "downloads_subdir": "downloads",
            "provenance_file": "download_provenance.json",
        },
        "gcp": {"bucket": None, "project": None},
        "sampling": {
            "fraction": 0.1,
            "seed": 82718,
            "universe_table": "clients",
            "output_dir": str(tmp_path / "raw_sample"),
            "compression": "gzip",
            "object_prefix": "raw_sample",
            "upload_chunk_bytes": 262144,
            **sampling,
        },
        "duckdb": {"memory_limit": "1GB", "threads": 2},
    }
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture()
def tiny(tmp_path):
    paths = build_tiny_x5(tmp_path / "data" / "downloads", n_clients=400, gzip_files=True)
    return tmp_path, paths


def run_sample(tmp_path, *extra, store=None, env=None):
    return sc.main(
        ["--config", str(write_config(tmp_path)), *extra], env={} if env is None else env, store=store
    )


def load_all(tmp_path):
    out = tmp_path / "raw_sample"
    return {
        name: read_gz_csv(out / f"{name}.csv.gz")
        for name in ("clients", "uplift_train", "uplift_test", "purchases", "products")
    }


def selected_from_disk(tmp_path) -> list[str]:
    return [row["client_id"] for row in read_gz_csv(tmp_path / "raw_sample" / sc.IDS_FILE)[1]]


# ------------------------------------------------------------------ pure selection rule


def test_sample_size_rounding_and_boundaries():
    assert sc.sample_size(400_162, 0.01) == 4002
    assert sc.sample_size(100, 1.0) == 100
    assert sc.sample_size(1, 1.0) == 1
    assert sc.sample_size(150, 0.01) == 2  # 1.5 rounds half up
    for bad in (0, -0.1, 1.5):
        with pytest.raises(sc.SamplingError, match="fraction"):
            sc.sample_size(10, bad)
    with pytest.raises(sc.SamplingError, match="rounds to an empty"):
        sc.sample_size(10, 0.01)
    with pytest.raises(sc.SamplingError, match="no client IDs"):
        sc.sample_size(0, 0.5)


def test_reference_selection_is_deterministic_and_order_independent():
    ids = [f"{i:010x}" for i in range(1000)]
    a = sc.reference_select(ids, 0.01, 82718)
    shuffled = ids[:]
    random.Random(3).shuffle(shuffled)
    assert sc.reference_select(shuffled + ids[:50], 0.01, 82718) == a  # duplicates and order do not matter
    assert len(a) == 10 and len(set(a)) == 10
    assert sc.reference_select(ids, 0.01, 1) != a  # the seed matters
    assert a == sorted(a)


def test_ids_checksum_is_stable_and_order_sensitive_only_through_sorting():
    assert sc.ids_checksum(["a", "b"]) == sc.ids_checksum(["a", "b"])
    assert sc.ids_checksum(["a", "b"]) != sc.ids_checksum(["a", "c"])
    import hashlib

    assert sc.ids_checksum(["0000abcdef"]) == hashlib.sha256(b"0000abcdef\n").hexdigest()


def test_duckdb_selection_equals_the_python_reference(tiny):
    tmp_path, paths = tiny
    con = sc.connect({"duckdb": {"threads": 3}})
    sc.register_sources(con, ["clients"], tmp_path / "data" / "downloads")
    n, k, missing = sc.select_client_ids(con, "clients", 0.05, 82718)
    all_ids = [r["client_id"] for r in read_gz_csv(paths["clients"])[1]]
    assert (n, k, missing) == (400, 20, 0)
    assert sc.selected_ids(con) == sc.reference_select(all_ids, 0.05, 82718)


# ------------------------------------------------------------------ end-to-end sample


def test_sample_is_by_client_and_all_rows_of_selected_clients_are_kept(tiny):
    tmp_path, paths = tiny
    assert run_sample(tmp_path) == sc.EXIT_OK
    selected = set(selected_from_disk(tmp_path))
    assert len(selected) == 40  # round(0.1 * 400)
    data = load_all(tmp_path)

    _, source_purchases = read_gz_csv(paths["purchases"])
    expected_purchases = [r for r in source_purchases if r["client_id"] in selected]
    assert data["purchases"][1] == expected_purchases  # same rows, same order, same text
    # whole transactions only: per-transaction row counts equal the source
    from collections import Counter

    source_counts = Counter(r["transaction_id"] for r in source_purchases)
    sample_counts = Counter(r["transaction_id"] for r in data["purchases"][1])
    assert all(source_counts[t] == c for t, c in sample_counts.items())
    # selection is by client: every selected client with purchases keeps all of its transactions
    source_tx_by_client = {}
    for r in source_purchases:
        source_tx_by_client.setdefault(r["client_id"], set()).add(r["transaction_id"])
    sample_tx_by_client = {}
    for r in data["purchases"][1]:
        sample_tx_by_client.setdefault(r["client_id"], set()).add(r["transaction_id"])
    assert sample_tx_by_client == {c: t for c, t in source_tx_by_client.items() if c in selected}

    for table in ("clients", "uplift_train", "uplift_test"):
        _, source_rows = read_gz_csv(paths[table])
        assert data[table][1] == [r for r in source_rows if r["client_id"] in selected], table


def test_schema_and_string_ids_are_preserved(tiny):
    tmp_path, paths = tiny
    run_sample(tmp_path)
    data = load_all(tmp_path)
    for table, (columns, _rows) in data.items():
        assert columns == read_gz_csv(paths[table])[0], table
    ids = selected_from_disk(tmp_path)
    assert any(cid.startswith("0000") for cid in ids)  # leading zeros survive
    assert all(len(cid) == 10 for cid in ids)
    assert all(len(r["client_id"]) == 10 for r in data["purchases"][1])
    # numeric text is untouched ("10.0" stays "10.0", never "10")
    assert {r["regular_points_received"] for r in data["purchases"][1]} <= {"0.0", "2.0", "5.1", "10.0"}


def test_uplift_test_stays_unlabeled_and_train_keeps_labels(tiny):
    tmp_path, _ = tiny
    run_sample(tmp_path)
    data = load_all(tmp_path)
    assert data["uplift_test"][0] == ["client_id"]
    assert data["uplift_train"][0] == ["client_id", "treatment_flg", "target"]
    train_ids = {r["client_id"] for r in data["uplift_train"][1]}
    test_ids = {r["client_id"] for r in data["uplift_test"][1]}
    assert train_ids and test_ids and not (train_ids & test_ids)
    assert train_ids | test_ids == {r["client_id"] for r in data["clients"][1]}


def test_tables_join_through_the_same_selected_ids(tiny):
    tmp_path, _ = tiny
    run_sample(tmp_path)
    data = load_all(tmp_path)
    selected = set(selected_from_disk(tmp_path))
    client_ids = {r["client_id"] for r in data["clients"][1]}
    assert client_ids == selected
    for table in ("uplift_train", "uplift_test", "purchases"):
        assert {r["client_id"] for r in data[table][1]} <= client_ids
    # clients with no purchase history are legitimate and stay in clients/labels
    purchase_clients = {r["client_id"] for r in data["purchases"][1]}
    assert purchase_clients < client_ids


def test_products_are_those_referenced_by_sampled_purchases(tiny):
    tmp_path, paths = tiny
    assert (
        sc.main(["--config", str(write_config(tmp_path, fraction=0.01))], env={}) == sc.EXIT_OK
    )  # 4 clients
    data = load_all(tmp_path)
    referenced = {r["product_id"] for r in data["purchases"][1]}
    sampled_products = {r["product_id"] for r in data["products"][1]}
    source_products = {r["product_id"] for r in read_gz_csv(paths["products"])[1]}
    assert sampled_products == referenced & source_products
    assert sampled_products < source_products  # unreferenced products are left out
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    products = metadata["validation"]["checks"]["products"]
    assert products["distinct_products_referenced_by_sampled_purchases"] == len(referenced)
    assert products["products_rows_not_referenced_by_sample"] == 0


def test_referenced_product_missing_from_products_is_reported_not_invented(tmp_path):
    paths = build_tiny_x5(tmp_path / "data" / "downloads", n_clients=100, gzip_files=True)
    columns, rows = read_gz_csv(paths["products"])
    dropped = rows[0]["product_id"]
    with gzip.open(paths["products"], "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[1:])
    assert run_sample(tmp_path, "--fraction", "1.0") == sc.EXIT_OK
    data = load_all(tmp_path)
    assert dropped not in {r["product_id"] for r in data["products"][1]}
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    assert metadata["validation"]["checks"]["products"]["referenced_products_missing_from_products"] >= 1


def test_ids_absent_from_clients_are_reported_and_never_sampled(tmp_path):
    paths = build_tiny_x5(tmp_path / "data" / "downloads", n_clients=100, gzip_files=True)
    with gzip.open(paths["uplift_train"], "at", newline="") as handle:
        handle.write("ffffffffff,1,1\n")  # label row for a client that is not in clients
    assert run_sample(tmp_path, "--fraction", "1.0") == sc.EXIT_OK
    data = load_all(tmp_path)
    assert "ffffffffff" not in {r["client_id"] for r in data["uplift_train"][1]}
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    assert metadata["validation"]["checks"]["source_client_ids_absent_from_universe"]["uplift_train"] == 1


def test_rows_without_client_id_are_excluded_from_the_universe(tmp_path):
    paths = build_tiny_x5(tmp_path / "data" / "downloads", n_clients=100, gzip_files=True)
    with gzip.open(paths["clients"], "at", newline="") as handle:
        handle.write(",2017-01-01 00:00:00,,30,F\n")
    assert run_sample(tmp_path, "--fraction", "1.0") == sc.EXIT_OK
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    assert metadata["universe"] == {"table": "clients", "unique_client_ids": 100, "rows_without_client_id": 1}


# ------------------------------------------------------------------ reproducibility


def test_rerun_reproduces_identical_ids_and_data(tiny):
    tmp_path, _ = tiny
    run_sample(tmp_path)
    first = {
        p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        for p in (tmp_path / "raw_sample").iterdir()
    }
    run_sample(tmp_path)
    second = {
        p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        for p in (tmp_path / "raw_sample").iterdir()
    }
    assert first == second  # includes sample_metadata.json: deterministic, no timestamps


def test_selection_is_independent_of_input_row_order_and_duplicates(tiny):
    tmp_path, paths = tiny
    run_sample(tmp_path)
    baseline = selected_from_disk(tmp_path)
    checksum = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())["selected_ids_checksum"][
        "value"
    ]

    columns, rows = read_gz_csv(paths["clients"])
    rng = random.Random(9)
    rng.shuffle(rows)
    rows = rows + rows[:25]  # duplicate source rows must not change the selection
    with gzip.open(paths["clients"], "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    run_sample(tmp_path)
    assert selected_from_disk(tmp_path) == baseline
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    assert metadata["selected_ids_checksum"]["value"] == checksum
    assert metadata["universe"]["unique_client_ids"] == 400


def test_seed_and_fraction_overrides_change_the_selection_reproducibly(tiny):
    tmp_path, _ = tiny
    run_sample(tmp_path)
    base = selected_from_disk(tmp_path)
    run_sample(tmp_path, "--seed", "1")
    other = selected_from_disk(tmp_path)
    assert len(other) == len(base) and other != base
    run_sample(tmp_path, "--seed", "1")
    assert selected_from_disk(tmp_path) == other
    run_sample(tmp_path, "--fraction", "0.25")
    assert len(selected_from_disk(tmp_path)) == 100


def test_fraction_boundaries_through_the_cli(tiny, capsys):
    tmp_path, _ = tiny
    assert run_sample(tmp_path, "--fraction", "0") == sc.EXIT_USAGE
    assert run_sample(tmp_path, "--fraction", "1.01") == sc.EXIT_USAGE
    assert run_sample(tmp_path, "--fraction", "0.001") == sc.EXIT_VALIDATION  # 0.4 clients: empty sample
    assert "rounds to an empty sample" in capsys.readouterr().err
    assert not (tmp_path / "raw_sample").exists()
    assert run_sample(tmp_path, "--fraction", "1.0") == sc.EXIT_OK
    assert len(selected_from_disk(tmp_path)) == 400


def test_metadata_records_seed_method_checksum_provenance_and_counts(tiny):
    tmp_path, paths = tiny
    (tmp_path / "data" / "downloads" / "download_provenance.json").write_text(json.dumps({"artifacts": {}}))
    run_sample(tmp_path)
    metadata = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())
    assert metadata["seed"] == 82718 and metadata["fraction"] == 0.1
    assert metadata["method"] == sc.SAMPLE_METHOD
    ids = selected_from_disk(tmp_path)
    assert metadata["selected_ids_checksum"]["value"] == sc.ids_checksum(sorted(ids))
    assert metadata["selected_client_count"] == len(ids) == 40
    data = load_all(tmp_path)
    for table, (_, rows) in data.items():
        assert metadata["outputs"][table]["rows"] == len(rows)
    import hashlib

    assert metadata["source"]["purchases"]["md5"] == hashlib.md5(paths["purchases"].read_bytes()).hexdigest()
    assert metadata["validation"]["ok"] is True
    assert "fraction of rows" not in json.dumps(metadata["table_rules"])


def test_source_provenance_is_linked_when_a_download_record_exists(tiny):
    tmp_path, paths = tiny
    import hashlib

    record = {
        "sha256": hashlib.sha256(paths["purchases"].read_bytes()).hexdigest(),
        "retrieved_at": "2026-11-26T00:00:00Z",
        "source": {"name": "huggingface", "url": "u", "revision_resolved": "abc"},
    }
    (tmp_path / "data" / "downloads" / "download_provenance.json").write_text(
        json.dumps({"artifacts": {"purchases": record}})
    )
    run_sample(tmp_path)
    source = json.loads((tmp_path / "raw_sample" / sc.METADATA_FILE).read_text())["source"]["purchases"]
    assert source["provenance_matches_artifact"] is True and source["source"]["revision_resolved"] == "abc"


# ------------------------------------------------------------------ validation


def test_validation_detects_foreign_rows_and_label_leakage(tiny, tmp_path):
    base, _ = tiny
    config = sc.load_config(write_config(base))
    settings = sc.SampleSettings.from_config(config)
    con = sc.connect(config)
    sources = sc.register_sources(con, list(config["dataset"]["tables"]), base / "data" / "downloads")
    n, k, _ = sc.select_client_ids(con, "clients", settings.fraction, settings.seed)
    out = base / "work"
    files = sc.write_sample(con, sources, settings, out)
    assert sc.validate_sample(con, sources, files, out, expected_k=k, check_orphans=False)["ok"]
    # a purchase row of a client that was not selected
    outsider = con.execute(
        "SELECT client_id FROM universe WHERE client_id NOT IN (SELECT client_id FROM selected) LIMIT 1"
    ).fetchone()[0]
    path = files["purchases"]["path"]
    with gzip.open(path, "at", newline="") as handle:
        handle.write(f"{outsider},t,2018-01-01 00:00:00,0.0,0.0,0.0,0.0,1.0,s,p,1.0,1.0,\n")
    report = sc.validate_sample(con, sources, files, out, expected_k=k, check_orphans=False)
    assert not report["ok"] and any("not selected" in p for p in report["problems"])


def test_failed_validation_publishes_nothing(tiny, monkeypatch, capsys):
    tmp_path, _ = tiny
    monkeypatch.setattr(
        sc, "validate_sample", lambda *a, **k: {"ok": False, "problems": ["forced"], "checks": {}}
    )
    assert run_sample(tmp_path) == sc.EXIT_VALIDATION
    assert not (tmp_path / "raw_sample").exists()
    assert (tmp_path / "raw_sample.failed").exists()
    assert "sample validation failed" in capsys.readouterr().err


def test_missing_input_table_is_a_clear_error(tmp_path, capsys):
    build_tiny_x5(tmp_path / "data" / "downloads", n_clients=50, gzip_files=True)
    (tmp_path / "data" / "downloads" / "purchases.csv.gz").unlink()
    assert run_sample(tmp_path) == sc.EXIT_VALIDATION
    assert "no input file for table 'purchases'" in capsys.readouterr().err


def test_dry_run_selects_but_writes_nothing(tiny, capsys):
    tmp_path, _ = tiny
    assert run_sample(tmp_path, "--dry-run") == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "40 of 400" in out and "checksum" in out
    assert not (tmp_path / "raw_sample").exists()


# ------------------------------------------------------------------ optional GCS upload


def test_upload_to_a_directory_store_is_idempotent_and_never_overwrites_silently(tiny):
    tmp_path, _ = tiny
    store = osx.DirectoryObjectStore(tmp_path / "bucket_root", BUCKET)
    assert run_sample(tmp_path, "--upload", "--bucket", BUCKET, store=store) == sc.EXIT_OK
    landed = sorted(p.name for p in (store.base / "raw_sample").iterdir())
    assert landed == sorted(p.name for p in (tmp_path / "raw_sample").iterdir())
    for p in (tmp_path / "raw_sample").iterdir():
        assert (store.base / "raw_sample" / p.name).read_bytes() == p.read_bytes()
    mtimes = {p: p.stat().st_mtime_ns for p in store.base.rglob("*") if p.is_file()}
    assert run_sample(tmp_path, "--upload", "--bucket", BUCKET, store=store) == sc.EXIT_OK
    assert mtimes == {p: p.stat().st_mtime_ns for p in store.base.rglob("*") if p.is_file()}
    # a different sample (new seed) now conflicts with what is in the bucket
    code = run_sample(tmp_path, "--seed", "5", "--upload", "--bucket", BUCKET, store=store)
    assert code == sc.EXIT_REMOTE
    assert (
        run_sample(tmp_path, "--seed", "5", "--upload", "--overwrite", "--bucket", BUCKET, store=store)
        == sc.EXIT_OK
    )


def test_upload_without_bucket_and_conflicting_flags(tiny, capsys):
    tmp_path, _ = tiny
    assert run_sample(tmp_path, "--upload") == sc.EXIT_CLOUD_CONFIG
    assert "needs a bucket" in capsys.readouterr().err
    assert run_sample(tmp_path, "--upload", "--dry-run") == sc.EXIT_USAGE


def test_shipped_sample_config_documents_seed_and_fraction():
    config = land_raw.load_config(land_raw.resolve_path("configs/x5/sample.yaml"))
    settings = sc.SampleSettings.from_config(config)
    assert settings.fraction == 0.01 and settings.seed == 82718 and settings.universe_table == "clients"
    assert settings.output_dir.name == "raw_sample"

"""Tests for the shared X5 config loader and helpers."""

from __future__ import annotations

import hashlib
import os

import pytest
import yaml

from ingestion import x5_common as common


def write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def test_extends_merges_nested_mappings_and_child_wins(tmp_path):
    write(tmp_path / "base.yaml", {"a": {"x": 1, "y": 2}, "keep": "base"})
    child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "a": {"y": 20, "z": 30}})
    config = common.load_config(child)
    assert config == {"a": {"x": 1, "y": 20, "z": 30}, "keep": "base"}


def test_extends_cycle_is_rejected(tmp_path):
    write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(common.ConfigError, match="circular"):
        common.load_config(tmp_path / "a.yaml")


def test_missing_and_malformed_files_raise_config_error(tmp_path):
    with pytest.raises(common.ConfigError, match="not found"):
        common.load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(common.ConfigError, match="not valid YAML"):
        common.load_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(common.ConfigError, match="mapping"):
        common.load_config(lst)


def test_default_config_is_valid_and_has_no_secrets_or_invented_values():
    config = common.load_config()
    assert common.configured_tables(config) == common.TABLES
    assert common.config_get(config, "campaign.cutoff") is None  # never guessed
    assert common.config_get(config, "gcp.bucket") is None
    text = open(common.DEFAULT_CONFIG, encoding="utf-8").read().lower()
    for forbidden in ("private_key", "password", "secret", "token:"):
        assert forbidden not in text


def test_relative_paths_resolve_against_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = common.resolve_path("data/x5")
    assert resolved == common.REPO_ROOT / "data" / "x5"
    assert common.resolve_path(tmp_path) == tmp_path


def test_data_root_cli_override_wins():
    config = {"paths": {"data_root": "from_yaml"}}
    assert common.data_root(config).name == "from_yaml"
    assert common.data_root(config, "/tmp/override") == common.resolve_path("/tmp/override")
    with pytest.raises(common.ConfigError):
        common.data_root({})


def test_configured_tables_validation():
    with pytest.raises(common.ConfigError):
        common.configured_tables({"dataset": {"tables": []}})
    with pytest.raises(common.ConfigError):
        common.configured_tables({"dataset": {"tables": ["ok", "ok"]}})
    with pytest.raises(common.ConfigError):
        common.configured_tables({"dataset": {"tables": ["Bad-Name"]}})


def test_file_digests_stream_large_files(tmp_path):
    path = tmp_path / "big.bin"
    block = os.urandom(1024)
    with open(path, "wb") as handle:
        for _ in range(3000):  # ~3 MB, several hash chunks
            handle.write(block)
    digests = common.file_digests(path)
    expected = hashlib.md5(block * 3000).hexdigest()
    assert digests["md5"] == expected
    assert digests["sha256"] == hashlib.sha256(block * 3000).hexdigest()


def test_atomic_write_json_leaves_no_temp_files_and_replaces_atomically(tmp_path):
    target = tmp_path / "sub" / "out.json"
    common.atomic_write_json(target, {"b": 1, "a": [1, 2]})
    common.atomic_write_json(target, {"b": 2})
    assert common.read_json(target) == {"b": 2}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]

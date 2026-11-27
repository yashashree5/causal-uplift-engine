"""Shared helpers for the X5 RetailHero ingestion code.

Everything here is dependency-light (standard library + PyYAML) so that the
download, manifest, sampling, EDA and replay code share one configuration
loader, one hashing routine and one atomic-write helper.

Configuration precedence (highest first)::

    CLI flag  >  environment variable (only where documented)  >  YAML file  >  built-in default

Relative paths in a config are resolved against the repository root, not the
current working directory, so commands behave the same from any directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "x5" / "x5.yaml"

#: Logical tables of the X5 RetailHero uplift dataset, in dependency order.
TABLES = ("uplift_train", "uplift_test", "clients", "products", "purchases")

HASH_CHUNK_BYTES = 1024 * 1024

_EXTENDS_KEY = "extends"
_TABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ConfigError(ValueError):
    """The configuration is missing, malformed or inconsistent."""


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated with ``overlay``; nested mappings merge, other values replace."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"configuration file is not valid YAML: {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return data


def load_config(path: str | os.PathLike[str] | None = None, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``extends: <file>`` chain.

    ``extends`` is resolved relative to the extending file. The extending file wins
    on conflicts. Cycles are rejected.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if config_path in _seen:
        raise ConfigError(f"circular 'extends' chain involving {config_path}")
    data = _read_yaml(config_path)
    parent_ref = data.pop(_EXTENDS_KEY, None)
    if parent_ref:
        parent = load_config(config_path.parent / str(parent_ref), _seen=_seen + (config_path,))
        data = deep_merge(parent, data)
    return data


def resolve_path(value: str | os.PathLike[str], *, root: Path = REPO_ROOT) -> Path:
    """Resolve ``value`` against ``root`` unless it is absolute. ``~`` is expanded."""
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (root / path)


def config_get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from nested mappings; return ``default`` when any level is missing/None."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def require_config(config: Mapping[str, Any], dotted: str) -> Any:
    """Like :func:`config_get` but raise :class:`ConfigError` if the key is unset."""
    value = config_get(config, dotted)
    if value is None:
        raise ConfigError(f"required configuration key is not set: {dotted}")
    return value


def configured_tables(config: Mapping[str, Any]) -> tuple[str, ...]:
    tables = config_get(config, "dataset.tables", list(TABLES))
    if not isinstance(tables, (list, tuple)) or not tables:
        raise ConfigError("dataset.tables must be a non-empty list")
    for name in tables:
        if not isinstance(name, str) or not _TABLE_NAME_RE.match(name):
            raise ConfigError(f"invalid logical table name: {name!r}")
    if len(set(tables)) != len(tables):
        raise ConfigError("dataset.tables contains duplicates")
    return tuple(tables)


def data_root(config: Mapping[str, Any], override: str | os.PathLike[str] | None = None) -> Path:
    """Local data root: CLI override, else ``paths.data_root`` from the config."""
    value = override if override is not None else require_config(config, "paths.data_root")
    return resolve_path(value)


def downloads_dir(config: Mapping[str, Any], root: Path) -> Path:
    return root / str(config_get(config, "paths.downloads_subdir", "downloads"))


def extracted_dir(config: Mapping[str, Any], root: Path) -> Path:
    return root / str(config_get(config, "paths.extracted_subdir", "extracted"))


# --------------------------------------------------------------------------- hashing / IO


def file_digests(path: str | os.PathLike[str], algorithms: Iterable[str] = ("md5", "sha256")) -> dict[str, str]:
    """Stream ``path`` once and return ``{algorithm: hexdigest}`` with bounded memory."""
    hashers = {name: hashlib.new(name) for name in algorithms}
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file in the same directory + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.chmod(tmp_name, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

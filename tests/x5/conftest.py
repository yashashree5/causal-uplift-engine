"""Shared pytest fixtures for the X5 pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from http_fixture import FixtureServer  # noqa: E402


@pytest.fixture()
def server():
    with FixtureServer() as srv:
        yield srv


@pytest.fixture()
def session():
    """A requests session that ignores proxy environment variables (localhost fixture only)."""
    s = requests.Session()
    s.trust_env = False
    yield s
    s.close()

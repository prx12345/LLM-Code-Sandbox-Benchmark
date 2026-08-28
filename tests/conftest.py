"""Shared pytest configuration and fixtures for the harness's own test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def write_candidate(tmp_path: Path):
    """Return a helper that writes candidate source code to a temp .py file."""

    def _write(source: str, name: str = "candidate.py") -> Path:
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        return path

    return _write

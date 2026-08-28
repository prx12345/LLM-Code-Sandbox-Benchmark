"""Tests for :mod:`main` — CLI flag behaviour and exit-code semantics."""

from __future__ import annotations

import json
from pathlib import Path

import main


def _write_fixture(tmp_path: Path, *, correct: bool) -> tuple[Path, Path]:
    """Write a tiny candidate + two-case suite; return their paths."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_name": "adder",
                "function": "add",
                "cases": [
                    {"id": "ok", "category": "baseline", "args": [1, 2], "expected": 3},
                    {"id": "edge", "category": "edge", "args": [0, 0], "expected": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    body = "a + b" if correct else "a - b"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"def add(a, b):\n    return {body}\n", encoding="utf-8")
    return candidate, suite_path


def _argv(candidate: Path, suite: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--code", str(candidate),
        "--tests", str(suite),
        "--output-dir", str(tmp_path / "results"),
        *extra,
    ]


def test_show_scorecard_prints_even_with_format_none(tmp_path, capsys) -> None:
    """Regression: ``--show-scorecard`` must not depend on ``--format md``."""
    candidate, suite = _write_fixture(tmp_path, correct=True)
    exit_code = main.main(
        _argv(candidate, suite, tmp_path, "--format", "none", "--show-scorecard")
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation Scorecard" in captured.out
    # --format none must still suppress file output.
    assert not (tmp_path / "results").exists()


def test_exit_codes_pass_fail_and_fail_under(tmp_path) -> None:
    """Exit 0 on PASS, 1 on FAIL, and ``--fail-under`` overrides the verdict."""
    good, suite = _write_fixture(tmp_path, correct=True)
    assert main.main(_argv(good, suite, tmp_path, "--format", "none")) == 0

    bad, suite = _write_fixture(tmp_path, correct=False)
    assert main.main(_argv(bad, suite, tmp_path, "--format", "none")) == 1
    # Failing candidate scores 0 here; a 0.0 gate therefore passes it.
    assert (
        main.main(_argv(bad, suite, tmp_path, "--format", "none", "--fail-under", "0"))
        == 0
    )


def test_malformed_suite_is_a_config_error(tmp_path) -> None:
    """Exit 2 with a clear message when the suite file is invalid."""
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    bad_suite = tmp_path / "suite.json"
    bad_suite.write_text("{not json", encoding="utf-8")
    assert main.main(_argv(candidate, bad_suite, tmp_path, "--format", "none")) == 2

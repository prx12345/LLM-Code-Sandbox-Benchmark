"""Tests for :mod:`batch` — multi-candidate ranking and the leaderboard CLI."""

from __future__ import annotations

import json
from pathlib import Path

import batch
from evaluator import EvaluationResult


def _write_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_name": "adder",
                "function": "add",
                "cases": [
                    {"id": "ok", "category": "baseline", "args": [1, 2], "expected": 3},
                    {"id": "zero", "category": "edge", "args": [0, 0], "expected": 0},
                    {"id": "neg", "category": "edge", "args": [-1, 1], "expected": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return suite


def _write_candidates(tmp_path: Path) -> Path:
    code_dir = tmp_path / "cands"
    code_dir.mkdir()
    (code_dir / "good.py").write_text("def add(a, b):\n    return a + b\n")
    (code_dir / "half.py").write_text(
        "def add(a, b):\n    return a + b if a >= 0 and b >= 0 else 1\n"
    )
    (code_dir / "bad.py").write_text("def add(a, b):\n    return a - b\n")
    (code_dir / "conftest.py").write_text("# must be ignored\n")
    return code_dir


def _eval(name: str, score: float, mean_ms: float | None) -> EvaluationResult:
    return EvaluationResult(
        suite_name="s", function_name="f", code_path=name, score=score,
        timing_ms={"mean_ms": mean_ms} if mean_ms is not None else None,
        verdict="PASS" if score == 100 else "FAIL",
    )


def test_rank_orders_by_score_then_speed_then_name() -> None:
    entries = batch.rank(
        [
            ("slow_perfect.py", _eval("slow_perfect.py", 100, 5.0), None),
            ("fast_perfect.py", _eval("fast_perfect.py", 100, 1.0), None),
            ("broken.py", None, "file not found"),
            ("partial.py", _eval("partial.py", 60, 0.1), None),
            ("zz_perfect_notiming.py", _eval("zz_perfect_notiming.py", 100, None), None),
        ]
    )
    assert [e.candidate for e in entries] == [
        "fast_perfect.py",
        "slow_perfect.py",
        "zz_perfect_notiming.py",  # missing timing sorts after measured ones
        "partial.py",
        "broken.py",  # harness errors always last
    ]
    assert [e.rank for e in entries] == [1, 2, 3, 4, 5]
    assert entries[-1].error == "file not found"
    assert entries[-1].score is None


def test_discover_candidates_sorted_deduped_and_filtered(tmp_path) -> None:
    code_dir = _write_candidates(tmp_path)
    explicit = code_dir / "good.py"
    found = batch.discover_candidates(
        code_dir, [explicit, code_dir / "conftest.py"], exclude_names=("conftest",)
    )
    # conftest excluded from the glob but kept when named explicitly; good.py
    # is not duplicated by being passed twice.
    assert [p.name for p in found] == ["bad.py", "good.py", "half.py", "conftest.py"]


def test_cli_writes_leaderboard_and_ranks_real_candidates(tmp_path, capsys) -> None:
    suite = _write_suite(tmp_path)
    code_dir = _write_candidates(tmp_path)
    out = tmp_path / "out"
    code = batch.main(
        ["--code-dir", str(code_dir), "--tests", str(suite), "--output-dir", str(out)]
    )
    assert code == 0  # top candidate passes

    board = json.loads((out / "leaderboard.json").read_text())
    names = [e["candidate"] for e in board["entries"]]
    assert names == ["good.py", "half.py", "bad.py"]
    assert "conftest.py" not in names
    assert board["candidates_passed"] == 1
    assert board["entries"][0]["score"] == 100
    assert board["entries"][1]["cases_passed"] == 2  # fails the negative edge case
    assert board["entries"][2]["cases_passed"] == 1  # 0-0 happens to be right

    md = (out / "leaderboard.md").read_text()
    assert "| 1 | `good.py` | PASS |" in md
    assert "Leaderboard" in capsys.readouterr().out


def test_cli_exit_codes(tmp_path) -> None:
    suite = _write_suite(tmp_path)
    code_dir = _write_candidates(tmp_path)
    only_bad = tmp_path / "onlybad"
    only_bad.mkdir()
    (only_bad / "bad.py").write_text((code_dir / "bad.py").read_text())

    common = ["--tests", str(suite), "--format", "none", "-q"]
    assert batch.main(["--code-dir", str(only_bad), *common]) == 1
    assert batch.main(["--code-dir", str(code_dir), "--fail-under", "100", *common]) == 0
    assert batch.main(["--code-dir", str(only_bad), "--fail-under", "50", *common]) == 1
    # Usage errors.
    assert batch.main(common) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert batch.main(["--code-dir", str(empty), *common]) == 2
    assert batch.main(["--code-dir", str(tmp_path / "nope"), *common]) == 2


def test_cli_one_broken_candidate_does_not_abort_batch(tmp_path) -> None:
    suite = _write_suite(tmp_path)
    code_dir = _write_candidates(tmp_path)
    out = tmp_path / "out"
    code = batch.main(
        [
            "--code-dir", str(code_dir),
            "--code", str(tmp_path / "missing.py"),
            "--tests", str(suite), "--output-dir", str(out), "-q",
        ]
    )
    assert code == 0
    board = json.loads((out / "leaderboard.json").read_text())
    last = board["entries"][-1]
    assert last["candidate"] == "missing.py"
    assert last["error"] and last["score"] is None
    assert board["candidates_total"] == 4

"""Tests for :mod:`reporter` — JSON report structure and Markdown scorecard."""

from __future__ import annotations

import json
from pathlib import Path

import reporter
from evaluator import Evaluator, load_suite
from sandbox import Sandbox


def _run_small_evaluation(tmp_path: Path, write_candidate, *, correct: bool):
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
    candidate = write_candidate(f"def add(a, b):\n    return {body}\n")
    suite = load_suite(suite_path)
    evaluation = Evaluator(Sandbox(), suite).run(candidate)
    config = {"code": str(candidate), "tests": str(suite_path), "limits": {"wall_time_s": 5.0, "memory_mb": 512}}
    return reporter.build_report(evaluation, config=config)


def test_report_contains_expected_schema_keys(tmp_path, write_candidate) -> None:
    report = _run_small_evaluation(tmp_path, write_candidate, correct=True)
    assert report["schema_version"] == reporter.SCHEMA_VERSION
    assert {"tool", "generated_at", "environment", "config", "summary", "cases"} <= set(report)
    summary = report["summary"]
    assert summary["verdict"] == "PASS"
    assert summary["cases_total"] == 2
    assert isinstance(report["cases"], list) and len(report["cases"]) == 2
    # The whole report must be JSON-serializable end to end.
    json.dumps(report)


def test_json_and_markdown_files_are_written(tmp_path, write_candidate) -> None:
    report = _run_small_evaluation(tmp_path, write_candidate, correct=True)
    json_path = reporter.write_json(report, tmp_path / "out" / "report.json")
    md_path = reporter.write_markdown(
        reporter.render_scorecard(report), tmp_path / "out" / "scorecard.md"
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["verdict"] == "PASS"
    assert "Evaluation Scorecard" in md_path.read_text(encoding="utf-8")


def test_scorecard_shows_verdict_cases_and_failures(tmp_path, write_candidate) -> None:
    passing = reporter.render_scorecard(
        _run_small_evaluation(tmp_path, write_candidate, correct=True)
    )
    assert "PASS" in passing and "`ok`" in passing
    assert "all cases passed" in passing

    failing = reporter.render_scorecard(
        _run_small_evaluation(tmp_path, write_candidate, correct=False)
    )
    assert "FAIL" in failing
    assert "wrong_answer" in failing
    assert "expected:" in failing and "actual:" in failing

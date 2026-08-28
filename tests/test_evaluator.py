"""Tests for :mod:`evaluator` — suite validation, comparison, and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator import (
    Evaluator,
    SuiteFormatError,
    load_suite,
    values_equal,
)
from sandbox import Sandbox, SandboxLimits


def _write_suite(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _basic_suite(**overrides) -> dict:
    suite = {
        "suite_name": "adder",
        "function": "add",
        "cases": [
            {"id": "small", "category": "baseline", "args": [1, 2], "expected": 3},
            {"id": "zero", "category": "edge", "args": [0, 0], "expected": 0},
            {"id": "negative", "category": "edge", "args": [-5, 2], "expected": -3},
        ],
    }
    suite.update(overrides)
    return suite


def test_correct_candidate_gets_full_marks(tmp_path, write_candidate) -> None:
    suite = load_suite(_write_suite(tmp_path, _basic_suite()))
    candidate = write_candidate("def add(a, b):\n    return a + b\n")
    evaluation = Evaluator(Sandbox(SandboxLimits(wall_time_s=5.0)), suite).run(candidate)
    assert evaluation.verdict == "PASS"
    assert evaluation.passed_count == evaluation.total_count == 3
    assert evaluation.accuracy_pct == 100.0
    assert evaluation.score == 100.0


def test_wrong_answers_produce_fail_verdict_and_partial_score(
    tmp_path, write_candidate
) -> None:
    suite = load_suite(_write_suite(tmp_path, _basic_suite()))
    candidate = write_candidate("def add(a, b):\n    return a + b + 1\n")
    evaluation = Evaluator(Sandbox(), suite).run(candidate)
    assert evaluation.verdict == "FAIL"
    assert evaluation.passed_count == 0
    assert all(r.failure_reason == "wrong_answer" for r in evaluation.cases)
    assert evaluation.score == 0.0


def test_partial_pass_scores_between_bounds(tmp_path, write_candidate) -> None:
    # Fails only on negative inputs -> baseline passes, edge partially fails.
    suite = load_suite(_write_suite(tmp_path, _basic_suite()))
    candidate = write_candidate("def add(a, b):\n    return abs(a) + abs(b)\n")
    evaluation = Evaluator(Sandbox(), suite).run(candidate)
    assert evaluation.verdict == "FAIL"
    assert 0.0 < evaluation.score < 100.0
    stats = evaluation.category_stats
    assert stats["baseline"]["passed"] == 1
    assert stats["edge"]["passed"] == 1 and stats["edge"]["total"] == 2


def test_unordered_comparison_ignores_top_level_order(tmp_path, write_candidate) -> None:
    payload = {
        "function": "pair",
        "comparison": "unordered",
        "cases": [
            {"id": "swap", "args": [], "expected": [1, 2], "category": "baseline"}
        ],
    }
    suite = load_suite(_write_suite(tmp_path, payload))
    candidate = write_candidate("def pair():\n    return [2, 1]\n")
    evaluation = Evaluator(Sandbox(), suite).run(candidate)
    assert evaluation.verdict == "PASS"


def test_generators_materialize_inputs(tmp_path) -> None:
    payload = {
        "function": "noop",
        "cases": [
            {
                "id": "generated",
                "args": [{"$generate": {"type": "repeat", "pattern": "ab", "times": 3}}],
                "expected": None,
            },
            {
                "id": "unique",
                "args": [{"$generate": {"type": "unique_chars", "count": 5}}],
                "expected": None,
            },
            {
                "id": "range",
                "args": [{"$generate": {"type": "range", "stop": 4}}],
                "expected": None,
            },
        ],
    }
    suite = load_suite(_write_suite(tmp_path, payload))
    by_id = {case.id: case for case in suite.cases}
    assert by_id["generated"].args[0] == "ababab"
    assert len(set(by_id["unique"].args[0])) == 5
    assert by_id["range"].args[0] == [0, 1, 2, 3]
    assert by_id["generated"].size == 6  # inferred from generated value


def test_values_equal_float_mode() -> None:
    assert values_equal([1.0, 2.0], [1.0000000001, 2.0], "float", 1e-6)
    assert not values_equal([1.0], [1.1], "float", 1e-6)
    assert values_equal({"x": 0.5}, {"x": 0.5}, "float", 1e-9)


@pytest.mark.parametrize(
    "broken",
    [
        {"cases": [{"args": [], "expected": 1}]},  # missing function
        {"function": "f", "cases": []},  # empty cases
        {"function": "f", "cases": [{"args": []}]},  # missing expected
        {"function": "f", "comparison": "vibes", "cases": [{"args": [], "expected": 1}]},
        {
            "function": "f",
            "cases": [{"id": "dup", "args": [], "expected": 1},
                      {"id": "dup", "args": [], "expected": 1}],
        },
        {"function": "f", "cases": [{"args": [], "expected": 1, "repeats": 0}]},
    ],
)
def test_malformed_suites_raise(tmp_path, broken) -> None:
    with pytest.raises(SuiteFormatError):
        load_suite(_write_suite(tmp_path, broken))


def test_runtime_error_case_is_failure_not_crash_of_harness(
    tmp_path, write_candidate
) -> None:
    suite = load_suite(_write_suite(tmp_path, _basic_suite()))
    candidate = write_candidate("def add(a, b):\n    raise RuntimeError('nope')\n")
    evaluation = Evaluator(Sandbox(), suite).run(candidate)
    assert evaluation.verdict == "FAIL"
    assert all(r.failure_reason == "runtime_error" for r in evaluation.cases)

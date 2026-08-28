"""Tests for :mod:`sandbox` — process isolation, limits, and result mapping."""

from __future__ import annotations

import os
import time

import pytest

from sandbox import Sandbox, SandboxError, SandboxLimits

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="requires POSIX rlimits")


def test_successful_call_returns_value_and_timing(write_candidate) -> None:
    path = write_candidate("def add(a, b):\n    return a + b\n")
    result = Sandbox().run(path, "add", [2, 3])
    assert result.ok
    assert result.serializable
    assert result.return_value == 5
    assert result.call_time_ms is not None and result.call_time_ms >= 0.0
    assert result.exit_code == 0


def test_infinite_loop_is_killed_within_budget(write_candidate) -> None:
    path = write_candidate("def loop():\n    while True:\n        pass\n")
    limits = SandboxLimits(wall_time_s=1.0, memory_mb=256)
    started = time.perf_counter()
    result = Sandbox(limits).run(path, "loop")
    elapsed = time.perf_counter() - started
    assert result.status == "timeout"
    assert not result.ok
    assert elapsed < 6.0, "timeout enforcement took far too long"


def test_candidate_exception_is_reported(write_candidate) -> None:
    path = write_candidate(
        "def boom():\n    raise ValueError('deliberate failure')\n"
    )
    result = Sandbox().run(path, "boom")
    assert result.status == "runtime_error"
    assert result.error is not None
    assert result.error["type"] == "ValueError"
    assert "deliberate failure" in result.error["message"]


def test_missing_function_is_a_runtime_error(write_candidate) -> None:
    path = write_candidate("def something_else():\n    return 1\n")
    result = Sandbox().run(path, "does_not_exist")
    assert result.status == "runtime_error"
    assert result.error is not None and result.error["type"] == "AttributeError"


@POSIX_ONLY
def test_memory_hog_hits_the_limit(write_candidate) -> None:
    path = write_candidate("def hog():\n    return len([0] * (10 ** 9))\n")
    limits = SandboxLimits(wall_time_s=5.0, memory_mb=256)
    result = Sandbox(limits).run(path, "hog")
    assert result.status == "memory_exceeded"
    assert result.error is not None and result.error["type"] == "MemoryError"


def test_candidate_prints_are_captured_without_breaking_protocol(write_candidate) -> None:
    path = write_candidate(
        "def chatty():\n    print('hello from candidate')\n    return 42\n"
    )
    result = Sandbox().run(path, "chatty")
    assert result.ok
    assert result.return_value == 42
    assert "hello from candidate" in result.stdout


def test_non_serializable_return_is_flagged(write_candidate) -> None:
    path = write_candidate("def weird():\n    return {1, 2, 3}\n")
    result = Sandbox().run(path, "weird")
    assert result.ok
    assert not result.serializable
    assert result.note is not None


def test_repeated_calls_report_min_timing(write_candidate) -> None:
    path = write_candidate("def f():\n    return 7\n")
    result = Sandbox().run(path, "f", repeats=5)
    assert result.ok
    assert result.return_value == 7
    assert result.timing_repeats == 5
    assert result.call_time_ms is not None and result.call_time_ms >= 0.0


def test_missing_candidate_file_raises_sandbox_error(tmp_path) -> None:
    with pytest.raises(SandboxError):
        Sandbox().run(tmp_path / "nope.py", "anything")

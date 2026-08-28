"""Correctness and performance evaluation of sandboxed candidate code.

Responsibilities:

* Load and validate a JSON test suite (baseline / edge / performance cases,
  optional input generators, comparison modes, scoring weights).
* Execute every case through :class:`sandbox.Sandbox`.
* Compare actual vs. expected outputs (exact, order-insensitive, or
  tolerance-based float comparison).
* Aggregate accuracy, timing and memory statistics, fit an empirical
  complexity estimate, and produce a weighted 0-100 score plus a
  PASS / FAIL verdict.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from complexity import ComplexityEstimate, estimate_complexity
from sandbox import Sandbox, SandboxResult

VALID_CATEGORIES: tuple[str, ...] = ("baseline", "edge", "performance")
VALID_COMPARISONS: tuple[str, ...] = ("exact", "unordered", "float")
DEFAULT_WEIGHTS: dict[str, float] = {
    "baseline": 0.5,
    "edge": 0.3,
    "performance": 0.2,
}
_MAX_UNIQUE_CHARS: int = 20_000
_REPORT_VALUE_CAP: int = 2_000


class SuiteFormatError(ValueError):
    """Raised when a test-suite file is malformed."""


# --------------------------------------------------------------------------
# Suite model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestCase:
    """A single, fully materialized test case."""

    id: str
    category: str
    args: list[Any]
    kwargs: dict[str, Any]
    expected: Any
    comparison: str | None = None
    float_tolerance: float | None = None
    timeout_s: float | None = None
    size: int | None = None
    complexity_sample: bool = True
    repeats: int = 1
    description: str = ""


@dataclass(frozen=True)
class TestSuite:
    """A validated collection of test cases for one target function."""

    name: str
    function_name: str
    comparison: str
    float_tolerance: float
    weights: dict[str, float]
    cases: list[TestCase]


def _generate(spec: dict[str, Any]) -> Any:
    """Materialize a ``$generate`` input spec into a concrete value.

    Supported generators:

    * ``{"type": "repeat", "pattern": <str|list>, "times": N}`` -- pattern
      repeated ``N`` times (large inputs without bloating the JSON file).
    * ``{"type": "range", "stop": N, "start": 0, "step": 1}`` -- ``list(range(...))``.
    * ``{"type": "unique_chars", "count": N, "start_codepoint": 0x4E00}`` --
      a string of ``N`` distinct characters (worst case for many string
      algorithms; used for complexity sampling).
    """
    gtype = spec.get("type")
    try:
        if gtype == "repeat":
            times = int(spec["times"])
            if times < 0:
                raise SuiteFormatError("'repeat' generator: 'times' must be >= 0")
            return spec["pattern"] * times
        if gtype == "range":
            return list(
                range(int(spec.get("start", 0)), int(spec["stop"]), int(spec.get("step", 1)))
            )
        if gtype == "unique_chars":
            count = int(spec["count"])
            if not 0 < count <= _MAX_UNIQUE_CHARS:
                raise SuiteFormatError(
                    f"'unique_chars' generator: 'count' must be in 1..{_MAX_UNIQUE_CHARS}"
                )
            start = int(spec.get("start_codepoint", 0x4E00))
            return "".join(chr(start + i) for i in range(count))
    except KeyError as exc:
        raise SuiteFormatError(f"generator {gtype!r} is missing key {exc}") from exc
    raise SuiteFormatError(f"unknown generator type: {gtype!r}")


def _materialize(value: Any) -> Any:
    """Replace ``{"$generate": {...}}`` wrappers with generated values."""
    if isinstance(value, dict) and "$generate" in value:
        return _generate(value["$generate"])
    return value


def _infer_size(args: list[Any]) -> int | None:
    """Best-effort input size: length of the first sized positional argument."""
    for arg in args:
        try:
            return len(arg)
        except TypeError:
            continue
    return None


def load_suite(path: str | Path) -> TestSuite:
    """Load, validate and materialize a test suite from a JSON file.

    Raises:
        SuiteFormatError: If required fields are missing or invalid.
    """
    suite_path = Path(path)
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SuiteFormatError(f"test suite not found: {suite_path}") from exc
    except ValueError as exc:
        raise SuiteFormatError(f"test suite is not valid JSON: {exc}") from exc

    function_name = raw.get("function")
    if not isinstance(function_name, str) or not function_name:
        raise SuiteFormatError("test suite must declare a target 'function' name")

    comparison = raw.get("comparison", "exact")
    if comparison not in VALID_COMPARISONS:
        raise SuiteFormatError(
            f"unknown comparison mode {comparison!r}; expected one of {VALID_COMPARISONS}"
        )
    float_tolerance = float(raw.get("float_tolerance", 1e-9))

    weights_raw = raw.get("weights") or {}
    weights: dict[str, float] = {}
    for key, value in weights_raw.items():
        if key not in VALID_CATEGORIES:
            raise SuiteFormatError(f"unknown weight category {key!r}")
        weight = float(value)
        if weight < 0:
            raise SuiteFormatError(f"weight for {key!r} must be >= 0")
        weights[key] = weight

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteFormatError("test suite must contain a non-empty 'cases' list")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise SuiteFormatError(f"case #{index} must be a JSON object")
        case_id = str(item.get("id") or f"case_{index:02d}")
        if case_id in seen_ids:
            raise SuiteFormatError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)

        category = item.get("category", "baseline")
        if category not in VALID_CATEGORIES:
            raise SuiteFormatError(
                f"case {case_id!r}: unknown category {category!r}; "
                f"expected one of {VALID_CATEGORIES}"
            )
        if "expected" not in item:
            raise SuiteFormatError(f"case {case_id!r} is missing an 'expected' value")

        case_comparison = item.get("comparison")
        if case_comparison is not None and case_comparison not in VALID_COMPARISONS:
            raise SuiteFormatError(
                f"case {case_id!r}: unknown comparison {case_comparison!r}"
            )

        repeats = int(item.get("repeats", 1))
        if repeats < 1:
            raise SuiteFormatError(f"case {case_id!r}: 'repeats' must be >= 1")

        args = [_materialize(arg) for arg in item.get("args", [])]
        kwargs = {key: _materialize(val) for key, val in (item.get("kwargs") or {}).items()}
        size = item.get("size")
        cases.append(
            TestCase(
                id=case_id,
                category=category,
                args=args,
                kwargs=kwargs,
                expected=item["expected"],
                comparison=case_comparison,
                float_tolerance=item.get("float_tolerance"),
                timeout_s=item.get("timeout_s"),
                size=int(size) if size is not None else _infer_size(args),
                complexity_sample=bool(item.get("complexity_sample", True)),
                repeats=repeats,
                description=str(item.get("description", "")),
            )
        )

    return TestSuite(
        name=str(raw.get("suite_name", suite_path.stem)),
        function_name=function_name,
        comparison=comparison,
        float_tolerance=float_tolerance,
        weights=weights,
        cases=cases,
    )


# --------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Sort a top-level list into a deterministic order for unordered compare."""
    if isinstance(value, list):
        return sorted(value, key=lambda item: json.dumps(item, sort_keys=True, default=repr))
    return value


def _float_close(expected: Any, actual: Any, tolerance: float) -> bool:
    """Recursive comparison treating numbers with a relative/absolute tolerance."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _float_close(e, a, tolerance) for e, a in zip(expected, actual)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _float_close(val, actual[key], tolerance) for key, val in expected.items()
        )
    return expected == actual


def values_equal(expected: Any, actual: Any, mode: str, tolerance: float) -> bool:
    """Compare an actual return value against the expected one.

    Args:
        expected: Ground-truth value from the test suite.
        actual: JSON-decoded return value from the sandbox.
        mode: ``"exact"``, ``"unordered"`` (top-level list order ignored) or
            ``"float"`` (recursive ``math.isclose``).
        tolerance: Tolerance used by ``"float"`` mode.
    """
    if mode == "unordered":
        return _canonical(expected) == _canonical(actual)
    if mode == "float":
        return _float_close(expected, actual, tolerance)
    return expected == actual


def _compact(value: Any, cap: int = _REPORT_VALUE_CAP) -> Any:
    """Clip large values for inclusion in reports (comparison uses full values)."""
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        encoded = repr(value)
    if len(encoded) <= cap:
        return value
    return {"$truncated": encoded[:cap] + "...", "$full_length": len(encoded)}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Outcome of a single test case."""

    id: str
    category: str
    description: str
    size: int | None
    complexity_sample: bool
    passed: bool
    status: str
    failure_reason: str | None
    expected: Any
    actual: Any
    call_time_ms: float | None
    total_time_ms: float | None
    max_rss_mb: float | None
    tracemalloc_peak_kb: float | None
    stdout: str
    stderr: str
    error: dict[str, str] | None


@dataclass
class EvaluationResult:
    """Aggregated outcome of a full suite run against one candidate."""

    suite_name: str
    function_name: str
    code_path: str
    cases: list[CaseResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0
    accuracy_pct: float = 0.0
    category_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    timing_ms: dict[str, float] | None = None
    memory: dict[str, float] | None = None
    complexity: ComplexityEstimate | None = None
    score: float = 0.0
    verdict: str = "FAIL"


ProgressCallback = Callable[[int, int, TestCase, CaseResult], None]


class Evaluator:
    """Runs a :class:`TestSuite` against candidate code inside a sandbox."""

    def __init__(self, sandbox: Sandbox, suite: TestSuite) -> None:
        """Bind an evaluator to a sandbox configuration and a test suite."""
        self._sandbox = sandbox
        self._suite = suite

    # ------------------------------------------------------------------ API

    def run(
        self,
        code_path: str | Path,
        *,
        function_name: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """Evaluate *code_path* against every case in the suite.

        Args:
            code_path: Candidate ``.py`` file under evaluation.
            function_name: Optional override of the suite's target function.
            progress: Optional callback invoked after each case, e.g. for
                CLI progress output.
        """
        target = function_name or self._suite.function_name
        results: list[CaseResult] = []
        total = len(self._suite.cases)

        for index, case in enumerate(self._suite.cases, start=1):
            sandbox_result = self._sandbox.run(
                code_path,
                target,
                case.args,
                case.kwargs,
                wall_time_s=case.timeout_s,
                repeats=case.repeats,
            )
            case_result = self._judge(case, sandbox_result)
            results.append(case_result)
            if progress is not None:
                progress(index, total, case, case_result)

        return self._aggregate(str(code_path), target, results)

    # ------------------------------------------------------------- internals

    def _judge(self, case: TestCase, sr: SandboxResult) -> CaseResult:
        """Combine sandbox status and output comparison into a case verdict."""
        mode = case.comparison or self._suite.comparison
        tolerance = (
            case.float_tolerance
            if case.float_tolerance is not None
            else self._suite.float_tolerance
        )

        if sr.status != "ok":
            passed, reason = False, sr.status
        elif not sr.serializable:
            passed, reason = False, "non_serializable_return"
        elif values_equal(case.expected, sr.return_value, mode, tolerance):
            passed, reason = True, None
        else:
            passed, reason = False, "wrong_answer"

        return CaseResult(
            id=case.id,
            category=case.category,
            description=case.description,
            size=case.size,
            complexity_sample=case.complexity_sample,
            passed=passed,
            status=sr.status,
            failure_reason=reason,
            expected=_compact(case.expected),
            actual=_compact(sr.return_value),
            call_time_ms=_round(sr.call_time_ms, 4),
            total_time_ms=_round(sr.total_time_ms, 2),
            max_rss_mb=_round(sr.max_rss_mb, 2),
            tracemalloc_peak_kb=_round(sr.tracemalloc_peak_kb, 1),
            stdout=sr.stdout,
            stderr=sr.stderr,
            error=sr.error,
        )

    def _aggregate(
        self, code_path: str, target: str, results: list[CaseResult]
    ) -> EvaluationResult:
        """Compute accuracy, category stats, timing/memory summaries and score."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        accuracy = (100.0 * passed / total) if total else 0.0

        category_stats: dict[str, dict[str, float | int]] = {}
        for category in VALID_CATEGORIES:
            in_category = [r for r in results if r.category == category]
            if not in_category:
                continue
            category_passed = sum(1 for r in in_category if r.passed)
            category_stats[category] = {
                "passed": category_passed,
                "total": len(in_category),
                "accuracy_pct": round(100.0 * category_passed / len(in_category), 1),
            }

        call_times = [r.call_time_ms for r in results if r.status == "ok" and r.call_time_ms is not None]
        timing = None
        if call_times:
            timing = {
                "samples": len(call_times),
                "mean_ms": round(statistics.fmean(call_times), 4),
                "median_ms": round(statistics.median(call_times), 4),
                "p95_ms": round(_percentile(call_times, 0.95), 4),
                "min_ms": round(min(call_times), 4),
                "max_ms": round(max(call_times), 4),
            }

        rss_values = [r.max_rss_mb for r in results if r.max_rss_mb is not None]
        memory: dict[str, float] | None = None
        if rss_values:
            memory = {
                "peak_rss_mb": round(max(rss_values), 2),
                "mean_rss_mb": round(statistics.fmean(rss_values), 2),
            }
        traced = [
            r.tracemalloc_peak_kb for r in results if r.tracemalloc_peak_kb is not None
        ]
        if traced:
            memory = memory or {}
            memory["peak_tracemalloc_kb"] = round(max(traced), 1)

        complexity = estimate_complexity(self._complexity_points(results))
        score = self._score(category_stats)
        verdict = "PASS" if total > 0 and passed == total else "FAIL"

        return EvaluationResult(
            suite_name=self._suite.name,
            function_name=target,
            code_path=code_path,
            cases=results,
            passed_count=passed,
            total_count=total,
            accuracy_pct=round(accuracy, 1),
            category_stats=category_stats,
            timing_ms=timing,
            memory=memory,
            complexity=complexity,
            score=score,
            verdict=verdict,
        )

    @staticmethod
    def _complexity_points(results: list[CaseResult]) -> list[tuple[float, float]]:
        """Select ``(size, time)`` samples for the complexity fit.

        Prefers dedicated ``performance`` cases; falls back to any sized,
        successful case when fewer than three are available. Cases can opt
        out via ``"complexity_sample": false`` (e.g. stress tests whose input
        shape differs from the scaling series).
        """

        def usable(r: CaseResult) -> bool:
            return (
                r.complexity_sample
                and r.status == "ok"
                and r.size is not None
                and r.size > 0
                and r.call_time_ms is not None
            )

        performance = [r for r in results if r.category == "performance" and usable(r)]
        pool = performance if len(performance) >= 3 else [r for r in results if usable(r)]
        return [(float(r.size), float(r.call_time_ms)) for r in pool]  # type: ignore[arg-type]

    def _score(self, category_stats: dict[str, dict[str, float | int]]) -> float:
        """Weighted 0-100 score across the categories present in the suite."""
        if not category_stats:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for category, stats in category_stats.items():
            weight = self._suite.weights.get(
                category, DEFAULT_WEIGHTS.get(category, 1.0 / len(category_stats))
            )
            rate = stats["passed"] / stats["total"] if stats["total"] else 0.0
            weighted_sum += weight * rate
            weight_total += weight
        if weight_total <= 0:
            return 0.0
        return round(100.0 * weighted_sum / weight_total, 1)


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; robust for small samples."""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _round(value: float | None, digits: int) -> float | None:
    """``round`` that tolerates ``None``."""
    return None if value is None else round(value, digits)

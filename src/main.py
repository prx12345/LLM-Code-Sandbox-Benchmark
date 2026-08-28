#!/usr/bin/env python3
"""CLI entry point for the LLM Code Benchmark & Sandbox Validator.

Typical usage::

    python src/main.py --code benchmarks/candidate_solution.py \\
                       --tests benchmarks/test_cases.json

Exit codes:
    0  evaluation ran and met the success criterion
       (verdict PASS, or score >= ``--fail-under`` when provided)
    1  evaluation ran but did not meet the success criterion
    2  configuration / usage error (bad paths, malformed suite, ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both `python src/main.py` and `python -m src.main` invocations.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reporter  # noqa: E402
from evaluator import CaseResult, Evaluator, SuiteFormatError, TestCase, load_suite  # noqa: E402
from sandbox import Sandbox, SandboxError, SandboxLimits  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-code-benchmark",
        description=(
            "Execute AI-generated code in a resource-limited sandbox, verify it "
            "against a JSON test suite, benchmark runtime and memory, and emit "
            "a JSON report plus a Markdown scorecard."
        ),
    )
    parser.add_argument(
        "--code", required=True, type=Path, help="candidate .py file to evaluate"
    )
    parser.add_argument(
        "--tests", required=True, type=Path, help="JSON test-suite file"
    )
    parser.add_argument(
        "--function",
        default=None,
        help="override the target function name declared in the suite",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="default wall-clock limit per test case (default: 5.0)",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=512,
        metavar="MB",
        help="address-space limit per case; 0 disables (default: 512)",
    )
    parser.add_argument(
        "--cpu-limit-s",
        type=int,
        default=None,
        metavar="SECONDS",
        help="CPU-time limit per case (default: ceil(timeout) + 1)",
    )
    parser.add_argument(
        "--trace-alloc",
        action="store_true",
        help=(
            "also measure Python-level peak allocations via tracemalloc "
            "(note: inflates measured runtimes)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for the JSON report and Markdown scorecard (default: results/)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md", "both", "none"),
        default="both",
        help="which report files to write (default: both)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="SCORE",
        help=(
            "exit 0 iff score >= SCORE (default criterion: verdict must be PASS)"
        ),
    )
    parser.add_argument(
        "--show-scorecard",
        action="store_true",
        help="print the full Markdown scorecard to stdout",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print per-case progress"
    )
    return parser


def _progress_printer(index: int, total: int, case: TestCase, result: CaseResult) -> None:
    """Verbose per-case progress line."""
    mark = "PASS" if result.passed else "FAIL"
    time_ms = f"{result.call_time_ms:.3f} ms" if result.call_time_ms is not None else "—"
    print(
        f"[{index:>2}/{total}] {case.id:<24} {case.category:<12} {mark:<4} "
        f"{result.status:<16} {time_ms}"
    )


def _print_summary(report: dict, json_path: Path | None, md_path: Path | None) -> None:
    """Compact console summary after a run."""
    summary = report["summary"]
    print()
    print(
        f"Verdict: {summary['verdict']}   Score: {summary['score']:.1f}/100   "
        f"Accuracy: {summary['cases_passed']}/{summary['cases_total']} "
        f"({summary['accuracy_pct']:.1f}%)"
    )
    timing = summary.get("timing_ms")
    if timing:
        print(
            f"Exec time: mean {timing['mean_ms']:.3f} ms · "
            f"p95 {timing['p95_ms']:.3f} ms · max {timing['max_ms']:.3f} ms"
        )
    memory = summary.get("memory")
    if memory and memory.get("peak_rss_mb") is not None:
        print(f"Peak process RSS: {memory['peak_rss_mb']:.1f} MB")
    complexity = summary.get("complexity")
    if complexity:
        confidence = "" if complexity["reliable"] else " (low confidence)"
        contested = (
            f" · close second: {complexity['runner_up_label']}"
            if complexity.get("contested")
            else ""
        )
        print(
            f"Estimated complexity: {complexity['label']}{contested}{confidence} — "
            f"{complexity['sample_count']} samples"
        )
    if json_path:
        print(f"JSON report: {json_path}")
    if md_path:
        print(f"Scorecard:   {md_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)

    try:
        suite = load_suite(args.tests)
    except SuiteFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    limits = SandboxLimits(
        wall_time_s=args.timeout,
        cpu_time_s=args.cpu_limit_s,
        memory_mb=args.memory_limit_mb or None,
    )
    sandbox = Sandbox(limits, trace_python_allocations=args.trace_alloc)
    evaluator = Evaluator(sandbox, suite)

    try:
        evaluation = evaluator.run(
            args.code,
            function_name=args.function,
            progress=_progress_printer if args.verbose else None,
        )
    except SandboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = {
        "code": str(args.code),
        "tests": str(args.tests),
        "function": evaluation.function_name,
        "limits": {
            "wall_time_s": limits.wall_time_s,
            "cpu_time_s": limits.resolved_cpu_s(),
            "memory_mb": limits.memory_mb,
        },
        "trace_alloc": args.trace_alloc,
    }
    report = reporter.build_report(evaluation, config=config)

    stem = Path(args.code).stem
    json_path = md_path = None
    markdown: str | None = None
    if args.show_scorecard or args.format in ("md", "both"):
        markdown = reporter.render_scorecard(report)
    if args.format in ("json", "both"):
        json_path = reporter.write_json(report, args.output_dir / f"{stem}_report.json")
    if args.format in ("md", "both") and markdown is not None:
        md_path = reporter.write_markdown(markdown, args.output_dir / f"{stem}_scorecard.md")
    if args.show_scorecard and markdown is not None:
        print(markdown)

    _print_summary(report, json_path, md_path)

    if args.fail_under is not None:
        return 0 if evaluation.score >= args.fail_under else 1
    return 0 if evaluation.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

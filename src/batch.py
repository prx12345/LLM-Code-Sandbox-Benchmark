#!/usr/bin/env python3
"""Rank many candidate solutions against one test suite.

The single-candidate CLI (``main.py``) answers "is this solution correct?".
This module answers the question that actually comes up when grading model
output: "given N solutions to the same problem, which is best?"

Every ``.py`` file in a directory (or an explicit list of files) is evaluated
through the same sandbox and suite, then ranked by score with faster mean
runtime breaking ties. The result is a ``leaderboard.json`` (stable schema,
one entry per candidate) and a ``leaderboard.md`` table.

Typical usage::

    python src/batch.py --code-dir benchmarks --tests benchmarks/test_cases.json

Exit codes:
    0  every candidate evaluated and the top-ranked one met ``--fail-under``
       (or reached verdict PASS when ``--fail-under`` is not given)
    1  evaluations ran but the top candidate did not meet the criterion
    2  configuration / usage error (no candidates, malformed suite, ...)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import EvaluationResult, Evaluator, SuiteFormatError, load_suite  # noqa: E402
from sandbox import Sandbox, SandboxError, SandboxLimits  # noqa: E402

LEADERBOARD_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LeaderboardEntry:
    """One ranked candidate. ``rank`` is 1-based; ``error`` is set when the
    harness itself could not run the file (missing path, unreadable), in which
    case every metric is ``None`` and the entry sorts last."""

    rank: int
    candidate: str
    verdict: str | None
    score: float | None
    accuracy_pct: float | None
    cases_passed: int | None
    cases_total: int | None
    mean_time_ms: float | None
    peak_rss_mb: float | None
    complexity: str | None
    complexity_reliable: bool | None
    error: str | None = None


# --------------------------------------------------------------------------
# Candidate discovery
# --------------------------------------------------------------------------


def discover_candidates(
    code_dir: Path | None, code_files: list[Path] | None, *, exclude_names: tuple[str, ...] = ()
) -> list[Path]:
    """Collect candidate ``.py`` files, sorted by name for a deterministic order.

    Files whose stem is in ``exclude_names`` are skipped (``__init__``,
    ``conftest`` and the like). Explicit ``--code`` paths are kept even if they
    match an excluded name — the user asked for them by name.
    """
    found: list[Path] = []
    if code_dir is not None:
        if not code_dir.is_dir():
            raise SandboxError(f"--code-dir is not a directory: {code_dir}")
        found.extend(
            p for p in sorted(code_dir.glob("*.py")) if p.stem not in exclude_names
        )
    if code_files:
        found.extend(code_files)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _sort_key(item: tuple[str, EvaluationResult | None]) -> tuple[int, float, float, str]:
    """Higher score first, then lower mean runtime, then name for stability.

    Candidates the harness could not run (``None``) sort after everything.
    """
    name, evaluation = item
    if evaluation is None:
        return (1, 0.0, float("inf"), name)
    mean_ms = (evaluation.timing_ms or {}).get("mean_ms")
    return (0, -evaluation.score, mean_ms if mean_ms is not None else float("inf"), name)


def rank(
    outcomes: list[tuple[str, EvaluationResult | None, str | None]],
) -> list[LeaderboardEntry]:
    """Turn ``(candidate, evaluation, error)`` triples into ranked entries."""
    ordered = sorted(
        outcomes, key=lambda o: _sort_key((o[0], o[1]))
    )
    entries: list[LeaderboardEntry] = []
    for position, (name, evaluation, error) in enumerate(ordered, start=1):
        if evaluation is None:
            entries.append(
                LeaderboardEntry(
                    rank=position, candidate=name, verdict=None, score=None,
                    accuracy_pct=None, cases_passed=None, cases_total=None,
                    mean_time_ms=None, peak_rss_mb=None, complexity=None,
                    complexity_reliable=None, error=error or "unknown error",
                )
            )
            continue
        timing = evaluation.timing_ms or {}
        memory = evaluation.memory or {}
        entries.append(
            LeaderboardEntry(
                rank=position,
                candidate=name,
                verdict=evaluation.verdict,
                score=evaluation.score,
                accuracy_pct=evaluation.accuracy_pct,
                cases_passed=evaluation.passed_count,
                cases_total=evaluation.total_count,
                mean_time_ms=timing.get("mean_ms"),
                peak_rss_mb=memory.get("peak_rss_mb"),
                complexity=evaluation.complexity.label if evaluation.complexity else None,
                complexity_reliable=(
                    evaluation.complexity.reliable if evaluation.complexity else None
                ),
            )
        )
    return entries


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def build_leaderboard(
    entries: list[LeaderboardEntry], *, suite_name: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the JSON-serializable leaderboard document."""
    return {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": suite_name,
        "config": config,
        "candidates_total": len(entries),
        "candidates_passed": sum(1 for e in entries if e.verdict == "PASS"),
        "entries": [e.__dict__ for e in entries],
    }


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def render_leaderboard(board: dict[str, Any]) -> str:
    """Render the leaderboard as a Markdown table."""
    lines = [
        f"# Leaderboard — {board['suite']}",
        "",
        f"{board['candidates_passed']}/{board['candidates_total']} candidates passed · "
        f"generated {board['generated_at']}",
        "",
        "| # | Candidate | Verdict | Score | Accuracy | Mean time | Peak RSS | Complexity |",
        "|--:|---|---|--:|--:|--:|--:|---|",
    ]
    for e in board["entries"]:
        if e["error"]:
            lines.append(f"| {e['rank']} | `{e['candidate']}` | ERROR | — | — | — | — | {e['error']} |")
            continue
        acc = (
            f"{e['cases_passed']}/{e['cases_total']} ({e['accuracy_pct']:.0f}%)"
            if e["cases_total"] is not None
            else "—"
        )
        cx = e["complexity"] or "—"
        if e["complexity"] and e["complexity_reliable"] is False:
            cx += " (low confidence)"
        lines.append(
            f"| {e['rank']} | `{e['candidate']}` | {e['verdict']} | {_fmt(e['score'])} | {acc} | "
            f"{_fmt(e['mean_time_ms'], 3, ' ms')} | {_fmt(e['peak_rss_mb'], 1, ' MB')} | {cx} |"
        )
    lines.append("")
    lines.append(
        "Ranking: score descending, then mean runtime ascending. "
        "Candidates the harness could not run are listed last."
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the batch CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-code-benchmark-batch",
        description="Evaluate many candidate solutions against one suite and rank them.",
    )
    parser.add_argument("--code-dir", type=Path, help="directory of candidate .py files")
    parser.add_argument(
        "--code", type=Path, action="append", default=None,
        metavar="PATH", help="explicit candidate file (repeatable)",
    )
    parser.add_argument("--tests", required=True, type=Path, help="JSON test-suite file")
    parser.add_argument("--function", default=None, help="override the suite's target function")
    parser.add_argument("--timeout", type=float, default=5.0, metavar="SECONDS")
    parser.add_argument("--memory-limit-mb", type=int, default=512, metavar="MB")
    parser.add_argument("--cpu-limit-s", type=int, default=None, metavar="SECONDS")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="where leaderboard.json / leaderboard.md are written (default: results)",
    )
    parser.add_argument(
        "--format", choices=("json", "md", "both", "none"), default="both",
        help="which leaderboard files to write (default: both)",
    )
    parser.add_argument(
        "--fail-under", type=float, default=None, metavar="SCORE",
        help="exit 0 iff the top-ranked candidate scores >= SCORE",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the console table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the batch CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.code_dir is None and not args.code:
        print("error: provide --code-dir and/or at least one --code", file=sys.stderr)
        return 2

    try:
        suite = load_suite(args.tests)
        candidates = discover_candidates(
            args.code_dir, args.code, exclude_names=("__init__", "conftest", "setup")
        )
    except (SuiteFormatError, SandboxError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print("error: no candidate .py files found", file=sys.stderr)
        return 2

    limits = SandboxLimits(
        wall_time_s=args.timeout,
        cpu_time_s=args.cpu_limit_s,
        memory_mb=args.memory_limit_mb or None,
    )
    evaluator = Evaluator(Sandbox(limits), suite)

    outcomes: list[tuple[str, EvaluationResult | None, str | None]] = []
    for path in candidates:
        try:
            outcomes.append((path.name, evaluator.run(path, function_name=args.function), None))
        except SandboxError as exc:
            # One broken file must not abort the whole batch.
            outcomes.append((path.name, None, str(exc)))

    entries = rank(outcomes)
    config = {
        "tests": str(args.tests),
        "code_dir": str(args.code_dir) if args.code_dir else None,
        "code": [str(p) for p in args.code] if args.code else [],
        "function": args.function or suite.function_name,
        "limits": {
            "wall_time_s": limits.wall_time_s,
            "cpu_time_s": limits.resolved_cpu_s(),
            "memory_mb": limits.memory_mb,
        },
    }
    board = build_leaderboard(entries, suite_name=suite.name, config=config)
    markdown = render_leaderboard(board)

    if args.format in ("json", "both"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "leaderboard.json").write_text(
            json.dumps(board, indent=2), encoding="utf-8"
        )
    if args.format in ("md", "both"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "leaderboard.md").write_text(markdown, encoding="utf-8")
    if not args.quiet:
        print(markdown, end="")

    top = entries[0]
    if top.error is not None:
        return 1
    if args.fail_under is not None:
        return 0 if (top.score or 0.0) >= args.fail_under else 1
    return 0 if top.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

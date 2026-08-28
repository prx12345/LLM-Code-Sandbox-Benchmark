# 🧪 LLM Code Benchmark & Sandbox Validator

<!-- After pushing, replace <your-username> to activate the CI badge: -->
<!-- ![CI](https://github.com/<your-username>/llm-code-benchmark/actions/workflows/ci.yml/badge.svg) -->
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/runtime%20deps-none-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An automated evaluation harness for **AI-generated code**. It ingests a candidate
solution plus a reference problem specification, executes the code inside a
**restricted, resource-limited Python sandbox**, verifies correctness against a
ground-truth test suite (baseline + edge + performance cases), benchmarks runtime
and memory, fits an **empirical Big-O estimate**, and emits both a structured
**JSON report** and a human-readable **Markdown scorecard** with a weighted
0–100 score and a PASS/FAIL verdict.

Built for the workflows behind LLM code evaluation: HumanEval-style functional
correctness checking, RLHF response grading, model benchmarking, and CI
regression gates for generated code. **Zero runtime dependencies** — Python
3.10+ standard library only.

## Why this project?

Evaluating model-generated code well is harder than running `pytest` on it. This
repository demonstrates, end to end, the competencies that job entails:

1. **Automated correctness testing** — declarative JSON test suites with
   baseline/edge/performance categories, input *generators* for large cases,
   and pluggable comparison semantics (exact, order-insensitive, float-tolerant).
2. **Code sandboxing & resource governance** — every case runs in a fresh
   subprocess under wall-clock, CPU-time, address-space, file-size, descriptor
   and process-count limits, so infinite loops, allocation bombs and fork bombs
   are contained and reported instead of taking down the host.
3. **Algorithmic complexity benchmarking** — runtimes are measured *inside* the
   child (interpreter start-up never pollutes the numbers) and fitted against
   O(1)…O(n³) growth models by least squares, with explicit reliability flags —
   the harness distinguishes an O(n) submission from an O(n²) one even when both
   are 100% correct.
4. **Systematic LLM/RLHF evaluation practice** — deterministic weighted rubric
   scoring, machine-readable reports with a stable schema for dashboards and
   regression tracking, failure diffs with tracebacks for annotation work, and
   CI-friendly exit codes (`--fail-under`).
5. **Production engineering hygiene** — typed Python 3.10+, docstrings, robust
   exception handling, a 34-test self-test suite, GitHub Actions CI across three
   Python versions, and honest documentation of the tool's threat model and
   measurement limitations (see below).

## Architecture & workflow

```
  candidate_solution.py            test_cases.json
  (AI-generated code)              (problem spec: cases, expected
          │                         outputs, weights, generators)
          │                                 │
          ▼                                 ▼
  ┌─────────────────────────────────────────────────────┐
  │                  main.py  (CLI, argparse)           │
  └──────────────────────────┬──────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────┐
  │   evaluator.py — loads & validates the suite,       │
  │   materializes generated inputs, orchestrates runs, │
  │   compares outputs, aggregates & scores             │
  └───────────┬─────────────────────────────────────────┘
              │  one isolated subprocess per test case
              ▼
  ┌─────────────────────────────────────────────────────┐
  │   sandbox.py  ──spawns──▶  python -I sandbox_runner │
  │   • wall-clock timeout      • RLIMIT_CPU / _AS      │
  │   • process-group kill      • RLIMIT_FSIZE/NOFILE/  │
  │   • JSON in/out protocol      NPROC (fork-bomb cap) │
  │                             • stdout/stderr capture │
  │                             • in-child timing & RSS │
  └───────────┬─────────────────────────────────────────┘
              │  SandboxResult per case
              ▼
  ┌──────────────────────────┐   ┌─────────────────────┐
  │ complexity.py            │   │ reporter.py         │
  │ least-squares Big-O fit  │──▶│ report.json +       │
  │ over (size, time) points │   │ scorecard.md        │
  └──────────────────────────┘   └─────────────────────┘
```

**Per-case lifecycle:** materialize inputs → spawn isolated child → apply
rlimits → load candidate (compiled from source, bypassing stale `.pyc` caches) →
time the call from inside → capture prints/exceptions → serialize result → parent
kills the whole process group on timeout → judge (status + output comparison) →
aggregate.

## Quickstart

```bash
git clone https://github.com/<your-username>/llm-code-benchmark.git
cd llm-code-benchmark
pip install -r requirements.txt   # pytest only, for the harness's own tests

# Evaluate the reference candidate against the sample suite
python src/main.py --code benchmarks/candidate_solution.py \
                   --tests benchmarks/test_cases.json --verbose
```

Real output:

```
[ 1/17] classic_abcabcbb         baseline     PASS ok               0.005 ms
[ 2/17] all_repeats              baseline     PASS ok               0.006 ms
...
[16/17] perf_unique_8000         performance  PASS ok               1.703 ms
[17/17] stress_alphabet_208k     performance  PASS ok               33.653 ms

Verdict: PASS   Score: 100.0/100   Accuracy: 17/17 (100.0%)
Exec time: mean 2.260 ms · p95 33.653 ms · max 33.653 ms
Peak process RSS: 17.8 MB
Estimated complexity: O(n) · close second: O(n log n) — 5 samples
JSON report: results/candidate_solution_report.json
Scorecard:   results/candidate_solution_scorecard.md
```

### Three sample candidates, three stories

The repo ships three submissions for **Longest Substring Without Repeating
Characters** so every code path of the harness is demonstrable out of the box:

| Candidate | Verdict | Score | Accuracy | Mean exec | Estimated complexity |
| --- | --- | --- | --- | --- | --- |
| `candidate_solution.py` (sliding window) | ✅ PASS | 100.0 | 17/17 | ~2.3 ms | **linear-class**† |
| `candidate_bruteforce.py` (naive, correct) | ✅ PASS | 100.0 | 17/17 | ~215 ms | **O(n²)** — no close second |
| `candidate_buggy.py` (window bug) | ❌ FAIL | 83.3 | 15/17 | ~2.4 ms | linear-class† |

The brute-force candidate is *functionally identical* to the reference — only
the empirical complexity fit and timing profile tell them apart, which is
exactly the signal a code-quality evaluator needs. The buggy candidate trips
the two adversarial baseline cases (`abba`, `tmmzuxt`) that target the classic
"window moves backwards" mistake, and the scorecard renders an
expected-vs-actual diff for each failure.

† Reported as `O(n) · close second: O(n log n)`, the reverse, or occasionally a
single uncontested label — it varies with run-to-run timing noise.
Wall-clock data genuinely cannot separate the two here: dict growth and cache
effects make the theoretically-O(n) sliding window measure slightly superlinear
at large distinct-alphabet sizes. Rather than overclaim a single label, the
estimator reports the runner-up model and a `contested` flag whenever the
second-best fit is within 2× NRMSE of the best — while the O(n²) brute force is
unambiguous. Honest uncertainty reporting is a feature, not a bug, in an
evaluation tool (see *Measurement notes*).

## CLI reference

```
python src/main.py --code CANDIDATE.py --tests SUITE.json [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--code PATH` | required | Candidate `.py` file to evaluate |
| `--tests PATH` | required | JSON test-suite file |
| `--function NAME` | from suite | Override the target function name |
| `--timeout SECONDS` | `5.0` | Wall-clock limit per case (cases may override via `timeout_s`) |
| `--memory-limit-mb MB` | `512` | Address-space cap per case; `0` disables |
| `--cpu-limit-s SECONDS` | `ceil(timeout)+1` | CPU-time cap per case (`RLIMIT_CPU`) |
| `--trace-alloc` | off | Also record Python-level peak allocations (`tracemalloc`); inflates timings |
| `--output-dir DIR` | `results/` | Where reports are written |
| `--format {json,md,both,none}` | `both` | Which report files to write |
| `--show-scorecard` | off | Print the full Markdown scorecard to stdout |
| `--fail-under SCORE` | – | Exit 0 iff score ≥ SCORE (CI gate) |
| `-v, --verbose` | off | Per-case progress lines |

**Exit codes:** `0` success criterion met (verdict PASS, or score ≥
`--fail-under`) · `1` evaluation completed but criterion not met · `2`
configuration error (bad paths, malformed suite).

## Test-suite format

```jsonc
{
  "suite_name": "Longest Substring Without Repeating Characters",
  "function": "length_of_longest_substring",   // target callable in the candidate
  "comparison": "exact",                       // exact | unordered | float
  "float_tolerance": 1e-9,                     // used by "float" mode
  "weights": { "baseline": 0.5, "edge": 0.3, "performance": 0.2 },
  "cases": [
    {
      "id": "lookback_trap_abba",
      "category": "baseline",                  // baseline | edge | performance
      "args": ["abba"],                        // JSON-serializable positional args
      "expected": 2,
      "description": "Stale duplicate before the window."
    },
    {
      "id": "perf_unique_8000",
      "category": "performance",
      "args": [{ "$generate": { "type": "unique_chars", "count": 8000 } }],
      "expected": 8000,
      "timeout_s": 15,                         // per-case override
      "complexity_sample": true                // include in the Big-O fit
    }
  ]
}
```

Per-case optional fields: `kwargs`, `comparison`, `float_tolerance`,
`timeout_s`, `size` (explicit input size; otherwise inferred as `len()` of the
first sized argument), `complexity_sample` (default `true`).

**Input generators** keep huge inputs out of the JSON file:
`{"type": "repeat", "pattern": "...", "times": N}` ·
`{"type": "range", "stop": N, "start": 0, "step": 1}` ·
`{"type": "unique_chars", "count": N}` (N distinct characters — the worst case
for many string algorithms, used to drive the complexity fit).

## Scoring & verdict

* **Verdict** is strict: `PASS` only when *every* case passes.
* **Score** is a weighted rubric over category pass-rates
  (default `baseline 0.5 · edge 0.3 · performance 0.2`, override per suite),
  normalized over the categories present — so a submission that nails the happy
  path but ignores edge cases is visibly penalized without being zeroed.
* A case *passes* when the sandbox status is `ok` **and** the JSON-decoded
  return value matches `expected` under the configured comparison mode.
  Timeouts, `MemoryError`s, exceptions, crashes and non-JSON-serializable
  returns are distinct `failure_reason`s in the report.

## Isolation model (and its honest limits)

Each case runs in a fresh `python -I` subprocess with, on POSIX systems:
wall-clock timeout with **process-group SIGKILL** in the parent, `RLIMIT_CPU`
(kills busy loops even if the parent dies), `RLIMIT_AS` (turns allocation bombs
into a clean `MemoryError`), `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, and
`RLIMIT_NPROC` (caps fork bombs), all in a temp working directory with no
inherited environment. On Windows the rlimits degrade gracefully to the
wall-clock timeout alone.

> ⚠️ This is a **resource sandbox for benchmarking model-generated code, not a
> hardened security boundary**. The child can still read world-readable files
> and, on most systems, open network sockets. To evaluate genuinely untrusted
> or adversarial code, run this harness *inside* an OS-level sandbox (Docker,
> gVisor, Firecracker) — the two layers compose cleanly. Saying this out loud
> is part of the job: an evaluation tool that overstates its isolation is a
> liability.

## Measurement notes

* **Timing** is `time.perf_counter()` around the call, *inside* the child —
  interpreter start-up and IPC never pollute the numbers. Candidate module
  import time is reported separately (`load_time_ms`).
* **Memory** defaults to peak process RSS (`ru_maxrss`, includes ~15 MB of
  interpreter baseline). `--trace-alloc` adds `tracemalloc` peak for
  Python-level attribution, at a documented cost: tracing inflates runtimes.
* **Big-O estimation** is a least-squares fit of `t ≈ k·f(n)` over
  O(1)…O(n³) on the performance-category samples. Every estimate carries
  sample count, size span, normalized RMSE and a `reliable` flag; adjacent
  classes (O(n) vs O(n log n)) can be indistinguishable at small spans, and
  the report says so instead of guessing confidently.
* Candidate source is **compiled directly** (never via `.pyc` caches), which
  avoids a subtle stale-bytecode bug when files are rewritten quickly with
  same-size content — found and regression-tested while building this harness.

## Project structure

```
llm-code-benchmark/
├── src/
│   ├── main.py               # CLI (argparse), exit-code semantics
│   ├── evaluator.py          # suite loading/validation, orchestration, scoring
│   ├── sandbox.py            # parent-side controller: timeouts, kill, parsing
│   ├── sandbox_runner.py     # child-side: rlimits, capture, in-child timing
│   ├── complexity.py         # empirical Big-O fit with reliability flags
│   └── reporter.py           # JSON report + Markdown scorecard
├── benchmarks/
│   ├── candidate_solution.py     # O(n) reference submission
│   ├── candidate_bruteforce.py   # correct but O(n²) — complexity-detection demo
│   ├── candidate_buggy.py        # classic window bug — failure-reporting demo
│   └── test_cases.json           # 17 cases: baseline / edge / performance
├── tests/                    # 34-test self-test suite (pytest)
├── .github/workflows/ci.yml  # pytest + demo gate on Python 3.10–3.12
├── requirements.txt          # pytest only; runtime is stdlib-only
├── LICENSE                   # MIT
└── README.md
```

## Development

```bash
pytest -q          # run the harness's own 34 tests
python src/main.py --code benchmarks/candidate_buggy.py \
                   --tests benchmarks/test_cases.json --show-scorecard
```

CI runs the test suite on Python 3.10/3.11/3.12 and gates on the reference
candidate scoring 100 (`--fail-under 100`), so the harness's own regressions —
in sandboxing, comparison logic, or scoring — fail the build.

### Extending

* **New problem:** drop in a candidate `.py` and a suite `.json` — no code
  changes needed.
* **New comparison mode / generator:** add a branch in `evaluator.py`
  (`values_equal` / `_generate`) with a test.
* **New report format:** `reporter.render_scorecard` works off the JSON dict,
  so an HTML or CSV renderer is a single function away.
* **Batch evaluation of many candidates:** loop `main.main([...])` or shell
  out per file; reports are keyed by candidate filename.

## License

MIT — see [LICENSE](LICENSE).

# 🧪 LLM Code Benchmark & Sandbox Validator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/runtime%20deps-none-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An automated evaluation harness for **AI-generated code**. It ingests a candidate solution plus a reference problem specification, executes the code inside a **restricted, resource-limited Python sandbox**, verifies correctness against a ground-truth test suite (baseline + edge + performance cases), benchmarks runtime and memory, fits an **empirical Big-O estimate**, and emits both a structured **JSON report** and a human-readable **Markdown scorecard** with a weighted 0–100 score and a PASS/FAIL verdict.

Built for the workflows behind LLM code evaluation: HumanEval-style functional correctness checking, RLHF response grading, model benchmarking, and CI regression gates for generated code. **Zero runtime dependencies** — Python 3.10+ standard library only.

## Core Features & Design Decisions

Evaluating model-generated code requires robust isolation, performance benchmarking, and strict functional verification. This harness is designed around the following architectural pillars:

1. **Automated Correctness Testing:** Declarative JSON test suites with baseline/edge/performance categories, input *generators* for large cases, and pluggable comparison semantics (exact, order-insensitive, float-tolerant).
2. **Code Sandboxing & Resource Governance:** Every case runs in a fresh subprocess under wall-clock, CPU-time, address-space, file-size, descriptor, and process-count limits. Infinite loops, allocation bombs, and fork bombs are contained and reported instead of crashing the host.
3. **Algorithmic Complexity Benchmarking:** Runtimes are measured *inside* the child (interpreter start-up never pollutes the numbers) and fitted against O(1)…O(n³) growth models by least squares, with explicit reliability flags. The harness distinguishes an O(n) submission from an O(n²) one even when both are functionally correct.
4. **Systematic LLM/RLHF Evaluation:** Deterministic weighted rubric scoring, machine-readable reports with a stable schema for dashboards, failure diffs with tracebacks for annotation work, and CI-friendly exit codes (`--fail-under`).
5. **Production Engineering Hygiene:** Typed Python 3.10+, comprehensive docstrings, robust exception handling, and honest documentation of the tool's threat model and measurement limitations.

## Architecture & Workflow

```text
  candidate_solution.py            test_cases.json
  (AI-generated code)              (problem spec)
          │                         │
          ▼                         ▼
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
  │   • process-group kill      • RLIMIT_FSIZE/NOFILE   │
  │   • JSON in/out protocol    • stdout/stderr capture │
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

## Quickstart

```bash
git clone https://github.com/prx12345/LLM-Code-Sandbox-Benchmark.git
cd LLM-Code-Sandbox-Benchmark

# Evaluate the reference candidate against the sample suite
python src/main.py --code benchmarks/candidate_solution.py \
                   --tests benchmarks/test_cases.json --verbose
```

### CLI Reference

```text
python src/main.py --code CANDIDATE.py --tests SUITE.json [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--code PATH` | required | Candidate `.py` file to evaluate |
| `--tests PATH` | required | JSON test-suite file |
| `--timeout SECONDS` | `5.0` | Wall-clock limit per case |
| `--memory-limit-mb MB` | `512` | Address-space cap per case; `0` disables |
| `--format {json,md,both,none}` | `both` | Which report files to write |
| `--show-scorecard` | off | Print the full Markdown scorecard to stdout |
| `--fail-under SCORE` | – | Exit 0 iff score ≥ SCORE (CI gate) |

**Exit codes:** `0` success criterion met · `1` evaluation completed but criterion not met · `2` configuration error.

## Isolation Model

Each case runs in a fresh `python -I` subprocess with wall-clock timeout (process-group SIGKILL), `RLIMIT_CPU`, and `RLIMIT_AS`. 

> ⚠️ **Note:** This is a resource sandbox for benchmarking model-generated code, not a hardened security boundary. To evaluate genuinely adversarial code, run this harness inside an OS-level sandbox (Docker, gVisor) — the two layers compose cleanly.

## License

MIT

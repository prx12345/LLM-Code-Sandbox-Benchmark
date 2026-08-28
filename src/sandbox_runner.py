"""Child-process entry point executed *inside* the sandbox.

``sandbox.Sandbox`` launches this script in a separate, resource-limited
Python interpreter (``python -I``). The protocol is intentionally simple:

* **stdin**  -- a single JSON object describing the job: candidate code path,
  target function name, positional/keyword arguments, resource limits and
  measurement options.
* **stdout** -- a single JSON object describing the outcome: status, return
  value, captured streams, timing and memory measurements.

Wall-clock enforcement lives in the parent process (which can always kill
this one). This process applies best-effort CPU / address-space / file
limits on POSIX systems, captures anything the candidate prints, and times
the candidate call *from the inside* so interpreter start-up cost never
pollutes the benchmark numbers.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import traceback
import types
from typing import Any

#: Maximum number of captured stdout/stderr characters returned to the parent.
_STREAM_CAP: int = 64_000
#: Maximum size (in serialized characters) of a return value sent to the parent.
_RETURN_CAP: int = 1_000_000
#: Maximum bytes the candidate may write to any single file (POSIX only).
_FSIZE_LIMIT: int = 16 * 1024 * 1024


def _apply_resource_limits(memory_mb: int | None, cpu_seconds: int | None) -> None:
    """Apply best-effort POSIX resource limits to *this* process.

    Silently degrades on platforms without the :mod:`resource` module
    (e.g. Windows) -- the parent's wall-clock timeout still applies there.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return

    def _try_set(limit: int, soft: int, hard: int) -> None:
        try:
            resource.setrlimit(limit, (soft, hard))
        except (ValueError, OSError):  # pragma: no cover - platform quirks
            pass

    if memory_mb:
        limit_bytes = int(memory_mb) * 1024 * 1024
        _try_set(resource.RLIMIT_AS, limit_bytes, limit_bytes)
    if cpu_seconds:
        seconds = int(cpu_seconds)
        _try_set(resource.RLIMIT_CPU, seconds, seconds + 1)

    # Keep the candidate from filling the disk or hoarding descriptors.
    _try_set(resource.RLIMIT_FSIZE, _FSIZE_LIMIT, _FSIZE_LIMIT)
    _try_set(resource.RLIMIT_NOFILE, 128, 128)
    if hasattr(resource, "RLIMIT_NPROC"):
        # Blocks fork bombs / surprise subprocesses spawned by candidate code.
        _try_set(resource.RLIMIT_NPROC, 256, 256)


def _max_rss_mb() -> float | None:
    """Return the peak resident set size of this process in MiB, if known."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _truncate(text: str, cap: int = _STREAM_CAP) -> str:
    """Clip *text* to *cap* characters, appending a marker when truncated."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated, {len(text) - cap} chars omitted]"


def _load_function(code_path: str, function_name: str) -> Any:
    """Load the candidate module from *code_path* and return the target callable.

    Compiles the source directly instead of going through the import system:
    ``.pyc`` caches are validated only by mtime-seconds + file size, so a
    candidate file overwritten quickly with same-length content can silently
    execute *stale* bytecode. Compiling from source sidesteps that entirely
    and also keeps candidate runs from littering ``__pycache__`` directories.
    """
    sys.dont_write_bytecode = True
    with open(code_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    module = types.ModuleType("candidate_module")
    module.__file__ = code_path
    sys.modules["candidate_module"] = module  # supports dataclasses/pickle lookups
    exec(compile(source, code_path, "exec"), module.__dict__)  # noqa: S102
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise AttributeError(
            f"function {function_name!r} was not found (or is not callable) "
            f"in {code_path!r}"
        )
    return fn


def _serialize_return(value: Any) -> tuple[Any, bool, str | None]:
    """Return ``(payload, serializable, note)`` for the candidate's return value.

    Values that are not JSON-serializable (or absurdly large) are replaced by a
    clipped ``repr`` so the parent can still show something useful, while the
    ``serializable`` flag tells the evaluator that a faithful comparison is
    impossible.
    """
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)[:2_000], False, "return value is not JSON-serializable"
    if len(encoded) > _RETURN_CAP:
        return (
            encoded[:2_000] + "...",
            False,
            f"serialized return value exceeds {_RETURN_CAP} characters",
        )
    return value, True, None


def main() -> int:
    """Execute one sandboxed job described on stdin; emit one JSON result."""
    result: dict[str, Any] = {
        "status": "internal_error",
        "return_value": None,
        "serializable": False,
        "note": None,
        "stdout": "",
        "stderr": "",
        "load_time_ms": None,
        "call_time_ms": None,
        "tracemalloc_peak_kb": None,
        "timing_repeats": 1,
        "error": None,
    }

    try:
        config: dict[str, Any] = json.loads(sys.stdin.read())
    except (ValueError, OSError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        json.dump(result, sys.stdout)
        return 3

    _apply_resource_limits(
        config.get("memory_limit_mb"), config.get("cpu_time_limit_s")
    )
    trace_allocations = bool(config.get("trace_python_allocations", False))

    out_capture, err_capture = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out_capture), contextlib.redirect_stderr(
            err_capture
        ):
            load_start = time.perf_counter()
            fn = _load_function(config["code_path"], config["function_name"])
            result["load_time_ms"] = (time.perf_counter() - load_start) * 1000.0

            args = config.get("args") or []
            kwargs = config.get("kwargs") or {}

            if trace_allocations:
                # NOTE: tracemalloc adds per-allocation overhead, so timings
                # measured with it enabled are systematically inflated.
                import tracemalloc

                tracemalloc.start()

            # Min-of-N timing: repeated calls with the minimum taken is the
            # standard way to strip scheduler/allocator noise from benchmarks
            # (the minimum is the best estimate of the code's intrinsic cost).
            repeats = max(1, int(config.get("repeats", 1)))
            durations_ms: list[float] = []
            value: Any = None
            for _ in range(repeats):
                call_start = time.perf_counter()
                value = fn(*args, **kwargs)
                durations_ms.append((time.perf_counter() - call_start) * 1000.0)
            result["call_time_ms"] = min(durations_ms)
            result["timing_repeats"] = repeats

            if trace_allocations:
                import tracemalloc

                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                result["tracemalloc_peak_kb"] = peak / 1024.0

        payload, serializable, note = _serialize_return(value)
        result.update(
            status="ok", return_value=payload, serializable=serializable, note=note
        )
    except MemoryError:
        result.update(
            status="memory_exceeded",
            error={
                "type": "MemoryError",
                "message": "candidate exceeded the configured memory limit",
            },
        )
    except BaseException as exc:  # noqa: BLE001 - report *any* candidate failure
        result.update(
            status="runtime_error",
            error={
                "type": type(exc).__name__,
                "message": str(exc)[:2_000],
                "traceback": traceback.format_exc(limit=20)[-6_000:],
            },
        )

    result["stdout"] = _truncate(out_capture.getvalue())
    result["stderr"] = _truncate(err_capture.getvalue())
    result["max_rss_mb"] = _max_rss_mb()

    json.dump(result, sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

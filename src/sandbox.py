"""Parent-side sandbox controller.

Runs untrusted candidate code in a **separate, resource-limited Python
process** (see :mod:`sandbox_runner`) and returns a structured
:class:`SandboxResult`. Isolation layers, from the outside in:

1. **Wall-clock timeout** -- the parent kills the whole child process group
   if it does not finish in time (defeats ``time.sleep`` / deadlocks).
2. **CPU-time limit** (``RLIMIT_CPU``) -- kills busy loops even if the
   parent process itself dies.
3. **Address-space limit** (``RLIMIT_AS``) -- turns runaway allocations into
   a clean ``MemoryError`` instead of freezing the host.
4. **File/descriptor/process limits** -- blocks disk-filling and fork bombs.
5. ``python -I`` (isolated mode) plus a temp working directory -- no user
   site-packages, no environment inheritance, no writes into the repo.

.. warning::
   This is a *resource* sandbox for benchmarking model-generated code, not a
   hardened security boundary: the child can still read world-readable files
   and (on most systems) open network sockets. Run truly hostile code inside
   a container/VM (Docker, gVisor, Firecracker) in addition to this harness.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Extra wall-clock seconds granted on top of the per-case limit to cover
#: interpreter start-up and result serialization in the child.
_STARTUP_GRACE_S: float = 1.0

_RUNNER_PATH: Path = Path(__file__).resolve().with_name("sandbox_runner.py")


class SandboxError(RuntimeError):
    """Raised for harness-side misconfiguration (missing files, bad paths)."""


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits applied to every sandboxed execution.

    Attributes:
        wall_time_s: Wall-clock budget for a single call (parent-enforced).
        cpu_time_s: CPU-seconds budget (child-enforced via ``RLIMIT_CPU``).
            Defaults to ``ceil(wall_time_s) + 1`` when ``None``.
        memory_mb: Address-space cap in MiB (``RLIMIT_AS``); ``None`` disables.
    """

    wall_time_s: float = 5.0
    cpu_time_s: int | None = None
    memory_mb: int | None = 512

    def resolved_cpu_s(self, wall_time_s: float | None = None) -> int:
        """CPU limit to apply for a call with the given effective wall budget."""
        if self.cpu_time_s is not None:
            return int(self.cpu_time_s)
        return int(math.ceil(wall_time_s or self.wall_time_s)) + 1


@dataclass
class SandboxResult:
    """Outcome of a single sandboxed function call.

    ``status`` is one of ``ok``, ``timeout``, ``memory_exceeded``,
    ``runtime_error``, ``crashed`` or ``protocol_error``.
    """

    status: str
    return_value: Any = None
    serializable: bool = False
    note: str | None = None
    stdout: str = ""
    stderr: str = ""
    call_time_ms: float | None = None
    load_time_ms: float | None = None
    total_time_ms: float | None = None
    max_rss_mb: float | None = None
    tracemalloc_peak_kb: float | None = None
    timing_repeats: int = 1
    exit_code: int | None = None
    error: dict[str, str] | None = field(default=None)

    @property
    def ok(self) -> bool:
        """True when the candidate ran to completion without raising."""
        return self.status == "ok"


class Sandbox:
    """Executes candidate functions in isolated, resource-limited subprocesses."""

    def __init__(
        self,
        limits: SandboxLimits | None = None,
        *,
        python_executable: str | None = None,
        trace_python_allocations: bool = False,
    ) -> None:
        """Create a sandbox.

        Args:
            limits: Resource limits; defaults to :class:`SandboxLimits`.
            python_executable: Interpreter used for the child process.
            trace_python_allocations: Also measure Python-level peak
                allocations with ``tracemalloc``. Inflates measured runtimes,
                so it is off by default.
        """
        self.limits = limits or SandboxLimits()
        self._python = python_executable or sys.executable
        self._trace_allocations = trace_python_allocations
        if not _RUNNER_PATH.is_file():  # pragma: no cover - packaging error
            raise SandboxError(f"sandbox runner not found at {_RUNNER_PATH}")

    # ------------------------------------------------------------------ API

    def run(
        self,
        code_path: str | Path,
        function_name: str,
        args: list[Any] | tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        wall_time_s: float | None = None,
        repeats: int = 1,
    ) -> SandboxResult:
        """Execute ``function_name(*args, **kwargs)`` from *code_path* sandboxed.

        Args:
            code_path: Path to the candidate ``.py`` file.
            function_name: Name of the function to invoke.
            args: Positional arguments (must be JSON-serializable).
            kwargs: Keyword arguments (must be JSON-serializable).
            wall_time_s: Optional per-call override of the wall-clock limit.
            repeats: Call the function this many times and report the *minimum*
                duration (min-of-N de-noises benchmark timings). All calls
                share one wall-clock budget; intended for pure functions.

        Returns:
            A fully populated :class:`SandboxResult`; this method never raises
            for candidate failures, only for harness misconfiguration.
        """
        resolved_path = Path(code_path).resolve()
        if not resolved_path.is_file():
            raise SandboxError(f"candidate file not found: {resolved_path}")

        effective_wall = float(wall_time_s or self.limits.wall_time_s)
        payload = json.dumps(
            {
                "code_path": str(resolved_path),
                "function_name": function_name,
                "args": list(args),
                "kwargs": kwargs or {},
                "memory_limit_mb": self.limits.memory_mb,
                "cpu_time_limit_s": self.limits.resolved_cpu_s(effective_wall),
                "trace_python_allocations": self._trace_allocations,
                "repeats": max(1, int(repeats)),
            }
        )

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "cwd": tempfile.gettempdir(),
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True  # own process group

        started = time.perf_counter()
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [self._python, "-I", str(_RUNNER_PATH)], **popen_kwargs
        )
        try:
            out, err = proc.communicate(payload, timeout=effective_wall + _STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            out, err = proc.communicate()
            return SandboxResult(
                status="timeout",
                stderr=(err or "")[-4_000:],
                total_time_ms=(time.perf_counter() - started) * 1000.0,
                exit_code=proc.returncode,
                error={
                    "type": "Timeout",
                    "message": f"exceeded {effective_wall:.2f}s wall-clock limit",
                },
            )
        total_ms = (time.perf_counter() - started) * 1000.0
        return self._interpret(proc.returncode, out or "", err or "", total_ms)

    # ------------------------------------------------------------- internals

    def _kill(self, proc: subprocess.Popen[str]) -> None:
        """Forcefully terminate the child and (on POSIX) its whole group."""
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:  # pragma: no cover - Windows
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.kill()

    def _interpret(
        self, returncode: int, out: str, err: str, total_ms: float
    ) -> SandboxResult:
        """Turn raw child output into a :class:`SandboxResult`."""
        parsed = self._parse_json_tail(out)
        if parsed is None:
            return self._signal_or_protocol_result(returncode, out, err, total_ms)

        result = SandboxResult(
            status=str(parsed.get("status", "protocol_error")),
            return_value=parsed.get("return_value"),
            serializable=bool(parsed.get("serializable", False)),
            note=parsed.get("note"),
            stdout=parsed.get("stdout", ""),
            stderr=parsed.get("stderr", ""),
            call_time_ms=parsed.get("call_time_ms"),
            load_time_ms=parsed.get("load_time_ms"),
            total_time_ms=total_ms,
            max_rss_mb=parsed.get("max_rss_mb"),
            tracemalloc_peak_kb=parsed.get("tracemalloc_peak_kb"),
            timing_repeats=int(parsed.get("timing_repeats", 1)),
            exit_code=returncode,
            error=parsed.get("error"),
        )
        if err.strip() and not result.stderr:
            result.stderr = err[-4_000:]
        return result

    @staticmethod
    def _parse_json_tail(out: str) -> dict[str, Any] | None:
        """Parse the runner's JSON object from raw stdout, defensively."""
        text = out.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            last_line = text.rsplit("\n", 1)[-1]
            try:
                return json.loads(last_line)
            except ValueError:
                return None

    @staticmethod
    def _signal_or_protocol_result(
        returncode: int, out: str, err: str, total_ms: float
    ) -> SandboxResult:
        """Explain child deaths that produced no protocol output."""
        status, message = "protocol_error", "child produced no parseable result"
        if returncode < 0 and os.name == "posix":
            sig = -returncode
            name = signal.Signals(sig).name if sig in signal.Signals.__members__.values() else str(sig)
            if sig == signal.SIGXCPU:
                status, message = "timeout", "CPU-time limit exceeded (SIGXCPU)"
            elif sig == signal.SIGKILL:
                status, message = (
                    "crashed",
                    "killed by SIGKILL (possible OS OOM-killer or external kill)",
                )
            elif sig == getattr(signal, "SIGXFSZ", -1):
                status, message = "crashed", "file-size limit exceeded (SIGXFSZ)"
            else:
                status, message = "crashed", f"terminated by signal {name}"
        elif returncode != 0:
            status, message = "crashed", f"child exited with code {returncode}"
        return SandboxResult(
            status=status,
            stdout=out[-4_000:],
            stderr=err[-4_000:],
            total_time_ms=total_ms,
            exit_code=returncode,
            error={"type": "ChildProcessFailure", "message": message},
        )


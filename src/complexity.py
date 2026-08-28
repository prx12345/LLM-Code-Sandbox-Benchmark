"""Empirical time-complexity estimation from ``(input size, runtime)`` samples.

Given measured runtimes at several input sizes, fit each candidate growth
model ``t ≈ k · f(n)`` by least squares and report the model with the lowest
normalized residual. This is deliberately a *heuristic*: constant factors,
interpreter noise, caching and small size spans can all blur neighbouring
classes (``O(n)`` vs ``O(n log n)`` in particular), which is why every
estimate carries an explicit reliability flag and note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

#: Candidate growth models, ordered from slowest- to fastest-growing.
_MODELS: tuple[tuple[str, Callable[[float], float]], ...] = (
    ("O(1)", lambda n: 1.0),
    ("O(log n)", lambda n: math.log2(n) if n > 1 else 1.0),
    ("O(n)", lambda n: n),
    ("O(n log n)", lambda n: n * math.log2(n) if n > 1 else n),
    ("O(n^2)", lambda n: n * n),
    ("O(n^3)", lambda n: n * n * n),
)

#: Fits with a normalized RMSE above this are flagged as unreliable.
_MAX_RELIABLE_NRMSE: float = 0.35


@dataclass(frozen=True)
class ComplexityEstimate:
    """Best-fit growth class for a set of timing samples.

    Attributes:
        label: Big-O label of the best-fitting model, e.g. ``"O(n)"``.
        normalized_rmse: RMS residual of the fit divided by the mean runtime
            (lower is better; 0 means a perfect fit).
        sample_count: Number of ``(n, t)`` points used.
        size_span: ``max(n) / min(n)`` across the samples.
        reliable: True when there were enough samples, a wide enough size
            span, and a tight fit.
        note: Human-readable summary of how the estimate was obtained.
    """

    label: str
    normalized_rmse: float
    runner_up_label: str | None
    runner_up_nrmse: float | None
    sample_count: int
    size_span: float
    reliable: bool
    note: str

    @property
    def contested(self) -> bool:
        """True when the runner-up model fits nearly as well (within 2x NRMSE).

        Wall-clock measurements often cannot strongly separate adjacent
        growth classes (e.g. O(n) vs O(n log n) on modest size spans, or a
        theoretically-linear algorithm whose large-input runs pick up cache
        and container-growth effects). When this flag is set, treat the label
        as "best fit" rather than a definitive classification.
        """
        return (
            self.runner_up_nrmse is not None
            and self.runner_up_nrmse <= 2.0 * max(self.normalized_rmse, 1e-9)
        )


def estimate_complexity(
    points: Sequence[tuple[float, float]],
    *,
    min_samples: int = 4,
    min_size_span: float = 8.0,
) -> ComplexityEstimate | None:
    """Estimate the empirical time complexity of a function.

    Args:
        points: ``(input_size, runtime_ms)`` samples. Non-positive sizes and
            missing runtimes are ignored.
        min_samples: Minimum samples required for a *reliable* estimate.
        min_size_span: Minimum ``max(n)/min(n)`` required for a *reliable*
            estimate (small spans cannot separate growth classes).

    Returns:
        A :class:`ComplexityEstimate`, or ``None`` when fewer than two usable
        samples were provided.
    """
    usable = [
        (float(n), float(t))
        for n, t in points
        if n is not None and t is not None and float(n) > 0 and float(t) >= 0.0
    ]
    if len(usable) < 2:
        return None

    sizes = [n for n, _ in usable]
    times = [t for _, t in usable]
    mean_time = sum(times) / len(times)
    span = max(sizes) / min(sizes)

    fits: list[tuple[float, str]] = []
    for label, model in _MODELS:
        features = [model(n) for n in sizes]
        denom = sum(f * f for f in features)
        if denom <= 0.0:
            continue
        scale = sum(t * f for t, f in zip(times, features)) / denom
        residuals = [t - scale * f for t, f in zip(times, features)]
        rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        nrmse = rmse / mean_time if mean_time > 0 else math.inf
        fits.append((nrmse, label))

    if not fits:  # pragma: no cover - defensive; models cover all input
        return None
    fits.sort(key=lambda item: item[0])
    best_nrmse, best_label = fits[0]
    runner_up_nrmse, runner_up_label = fits[1] if len(fits) > 1 else (None, None)

    distinct_sizes = len(set(sizes))
    reliable = (
        distinct_sizes >= min_samples
        and span >= min_size_span
        and best_nrmse <= _MAX_RELIABLE_NRMSE
    )
    note = (
        f"least-squares fit over {len(usable)} samples "
        f"(sizes {int(min(sizes))}-{int(max(sizes))}, span x{span:.1f}, "
        f"normalized RMSE {best_nrmse:.3f}); empirical heuristic - "
        "adjacent classes such as O(n) and O(n log n) may be indistinguishable "
        "at small spans"
    )
    if runner_up_nrmse is not None and runner_up_nrmse <= 2.0 * max(best_nrmse, 1e-9):
        note += (
            f"; the measurement does not strongly separate {best_label} from "
            f"{runner_up_label} (NRMSE {best_nrmse:.3f} vs {runner_up_nrmse:.3f})"
        )
    return ComplexityEstimate(
        label=best_label,
        normalized_rmse=round(best_nrmse, 4),
        runner_up_label=runner_up_label,
        runner_up_nrmse=round(runner_up_nrmse, 4) if runner_up_nrmse is not None else None,
        sample_count=len(usable),
        size_span=round(span, 2),
        reliable=reliable,
        note=note,
    )

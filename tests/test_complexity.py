"""Tests for :mod:`complexity` — empirical growth-class fitting."""

from __future__ import annotations

from complexity import estimate_complexity

SIZES = [500, 1_000, 2_000, 4_000, 8_000]


def test_linear_data_is_labelled_o_n() -> None:
    points = [(n, 0.002 * n) for n in SIZES]
    estimate = estimate_complexity(points)
    assert estimate is not None
    assert estimate.label == "O(n)"
    assert estimate.reliable


def test_quadratic_data_is_labelled_o_n_squared() -> None:
    points = [(n, 1e-6 * n * n) for n in SIZES]
    estimate = estimate_complexity(points)
    assert estimate is not None
    assert estimate.label == "O(n^2)"
    assert estimate.reliable


def test_constant_data_is_labelled_o_1() -> None:
    points = [(n, 0.5) for n in SIZES]
    estimate = estimate_complexity(points)
    assert estimate is not None
    assert estimate.label == "O(1)"


def test_noise_tolerance_on_linear_data() -> None:
    noise = [1.07, 0.94, 1.02, 0.97, 1.05]
    points = [(n, 0.002 * n * k) for n, k in zip(SIZES, noise)]
    estimate = estimate_complexity(points)
    assert estimate is not None
    assert estimate.label in ("O(n)", "O(n log n)")  # neighbours; noise-dependent


def test_insufficient_points_returns_none() -> None:
    assert estimate_complexity([]) is None
    assert estimate_complexity([(1000, 1.0)]) is None


def test_small_span_is_flagged_unreliable() -> None:
    points = [(1_000, 1.0), (1_100, 1.1), (1_200, 1.2), (1_300, 1.3)]
    estimate = estimate_complexity(points)
    assert estimate is not None
    assert not estimate.reliable

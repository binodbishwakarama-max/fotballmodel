"""Validated multiclass forecast metrics for H/D/A probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

OUTCOMES: Final[tuple[str, str, str]] = ("H", "D", "A")


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One confidence bin for a one-vs-rest reliability curve."""

    count: int
    mean_confidence: float | None
    observed_frequency: float | None


def _validated_inputs(
    y_true: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    labels = np.asarray(y_true)
    probs = np.asarray(probabilities, dtype=np.float64)

    if labels.ndim != 1:
        raise ValueError("y_true must be a one-dimensional sequence of H/D/A labels")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true labels must be integers: 0=H, 1=D, 2=A")
    if labels.size == 0:
        raise ValueError("at least one outcome is required")
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("y_true labels must be 0=H, 1=D, or 2=A")
    if probs.shape != (labels.size, 3):
        raise ValueError("probabilities must have shape (number of outcomes, 3)")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("each probability row must sum to 1")

    return labels.astype(np.int64, copy=False), probs


def multiclass_log_loss(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Return mean categorical log loss for integer labels H=0, D=1, A=2."""
    if not 0 < epsilon < 1 / 3:
        raise ValueError("epsilon must be greater than 0 and less than 1/3")
    labels, probs = _validated_inputs(y_true, probabilities)
    clipped = np.clip(probs, epsilon, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return float(-np.log(clipped[np.arange(labels.size), labels]).mean())


def multiclass_brier_score(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Return mean per-match sum of squared H/D/A probability errors."""
    labels, probs = _validated_inputs(y_true, probabilities)
    actual = np.eye(3, dtype=np.float64)[labels]
    return float(np.square(probs - actual).sum(axis=1).mean())


def ranked_probability_score(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Return mean cumulative ranked probability score over H/D/A cut points."""
    labels, probs = _validated_inputs(y_true, probabilities)
    actual = np.eye(3, dtype=np.float64)[labels]
    forecast_cumulative = np.cumsum(probs, axis=1)[:, :-1]
    actual_cumulative = np.cumsum(actual, axis=1)[:, :-1]
    return float(np.square(forecast_cumulative - actual_cumulative).mean())


def reliability_curve_data(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    bins: int = 10,
) -> dict[str, tuple[ReliabilityBin, ...]]:
    """Return one-vs-rest reliability-bin counts and means for H, D, and A."""
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    labels, probs = _validated_inputs(y_true, probabilities)
    result: dict[str, tuple[ReliabilityBin, ...]] = {}

    for outcome_index, outcome in enumerate(OUTCOMES):
        confidence = probs[:, outcome_index]
        bin_indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
        outcome_bins: list[ReliabilityBin] = []

        for bin_index in range(bins):
            members = bin_indices == bin_index
            count = int(members.sum())
            if count == 0:
                outcome_bins.append(ReliabilityBin(0, None, None))
                continue
            outcome_bins.append(
                ReliabilityBin(
                    count=count,
                    mean_confidence=float(confidence[members].mean()),
                    observed_frequency=float((labels[members] == outcome_index).mean()),
                )
            )

        result[outcome] = tuple(outcome_bins)

    return result


def expected_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Return one-vs-rest ECE for each outcome and their unweighted macro average."""
    reliability = reliability_curve_data(y_true, probabilities, bins=bins)
    sample_count = sum(item.count for item in reliability["H"])
    ece_by_outcome: dict[str, float] = {}

    for outcome, outcome_bins in reliability.items():
        ece_by_outcome[outcome] = sum(
            item.count / sample_count * abs(item.mean_confidence - item.observed_frequency)
            for item in outcome_bins
            if item.count
            and item.mean_confidence is not None
            and item.observed_frequency is not None
        )

    ece_by_outcome["macro"] = float(np.mean(list(ece_by_outcome.values())))
    return ece_by_outcome

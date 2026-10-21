"""Hand-computed tests for multiclass forecast metrics."""

import math

import numpy as np
import pytest

from forecast.eval.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
    reliability_curve_data,
)

Y_TRUE = [0, 1]
PROBABILITIES = [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]]


def test_log_loss_matches_hand_computed_value() -> None:
    assert multiclass_log_loss(Y_TRUE, PROBABILITIES) == pytest.approx(
        -(math.log(0.8) + math.log(0.5)) / 2
    )


def test_brier_score_matches_hand_computed_value() -> None:
    assert multiclass_brier_score(Y_TRUE, PROBABILITIES) == pytest.approx(0.22)


def test_ranked_probability_score_matches_hand_computed_value() -> None:
    assert ranked_probability_score(Y_TRUE, PROBABILITIES) == pytest.approx(0.045)


def test_ece_returns_per_outcome_and_macro_values() -> None:
    scores = expected_calibration_error(Y_TRUE, PROBABILITIES, bins=2)

    assert scores == pytest.approx({"H": 0.2, "D": 0.3, "A": 0.2, "macro": 0.7 / 3})


def test_reliability_curve_includes_bin_counts_and_empty_bins() -> None:
    curves = reliability_curve_data(Y_TRUE, PROBABILITIES, bins=3)

    assert [item.count for item in curves["H"]] == [1, 0, 1]
    assert curves["H"][1].mean_confidence is None
    assert curves["H"][0].observed_frequency == 0.0
    assert curves["D"][1].observed_frequency == 1.0


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [
        ([0], [[0.5, 0.5]]),
        ([0], [[0.5, 0.5, 0.5]]),
        ([3], [[0.2, 0.3, 0.5]]),
        ([0], [[0.2, 0.3, 0.4]]),
        ([0], [[-0.1, 0.5, 0.6]]),
        ([0.5], [[0.2, 0.3, 0.5]]),
    ],
)
def test_metrics_reject_invalid_inputs(labels, probabilities) -> None:
    with pytest.raises(ValueError):
        multiclass_log_loss(labels, probabilities)


def test_metrics_accept_numpy_inputs() -> None:
    assert multiclass_brier_score(np.asarray(Y_TRUE), np.asarray(PROBABILITIES)) == pytest.approx(
        0.22
    )

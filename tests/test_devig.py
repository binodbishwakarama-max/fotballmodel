"""Tests for bookmaker overround removal."""

import numpy as np
import pytest

from forecast.eval.devig import devig


def test_proportional_method_normalizes_implied_probabilities() -> None:
    probabilities = devig([2.0, 3.5, 4.0], method="proportional")
    expected = np.asarray([0.5, 1 / 3.5, 0.25])
    expected /= expected.sum()

    assert probabilities == pytest.approx(expected)
    assert probabilities.sum() == pytest.approx(1.0)


def test_power_method_removes_overround() -> None:
    probabilities = devig([2.0, 3.5, 4.0], method="power")

    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0)
    assert probabilities[0] < 0.5


def test_power_method_preserves_fair_odds() -> None:
    probabilities = devig([2.0, 4.0, 4.0], method="power")

    assert probabilities == pytest.approx([0.5, 0.25, 0.25])


@pytest.mark.parametrize(
    ("odds", "method"),
    [
        ([2.0, 3.0], "power"),
        ([2.0, 3.0, 4.0, 5.0], "power"),
        ([1.0, 3.0, 4.0], "power"),
        ([2.0, float("nan"), 4.0], "power"),
        ([2.0, 3.0, 4.0], "unknown"),
    ],
)
def test_devig_rejects_invalid_odds_or_method(odds, method) -> None:
    with pytest.raises(ValueError):
        devig(odds, method=method)

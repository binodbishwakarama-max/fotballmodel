"""Remove bookmaker overround from decimal H/D/A odds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def devig(
    decimal_odds: Sequence[float],
    *,
    method: str = "power",
) -> NDArray[np.float64]:
    """Convert three decimal H/D/A prices into margin-removed probabilities."""
    odds = np.asarray(decimal_odds, dtype=np.float64)
    if odds.shape != (3,):
        raise ValueError("exactly three decimal odds are required in H/D/A order")
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1):
        raise ValueError("decimal odds must be finite and greater than 1")

    implied = 1.0 / odds
    if method == "proportional":
        return implied / implied.sum()
    if method == "power":
        return _power_devig(implied)
    raise ValueError("method must be 'proportional' or 'power'")


def _power_devig(implied: NDArray[np.float64]) -> NDArray[np.float64]:
    """Find exponent k for which the powered implied probabilities sum to one."""
    lower = 0.0
    upper = 1.0
    while np.power(implied, upper).sum() > 1.0:
        upper *= 2.0

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if np.power(implied, midpoint).sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint

    probabilities = np.power(implied, (lower + upper) / 2.0)
    return probabilities / probabilities.sum()

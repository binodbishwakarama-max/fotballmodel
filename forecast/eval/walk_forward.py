"""Chronological model evaluation and paired matchweek bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from forecast.eval.metrics import multiclass_log_loss

EstimatorFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Indices for a single chronological training/test season fold."""

    league: str
    test_season: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def make_walk_forward_folds(
    *,
    leagues: Sequence[str],
    seasons: Sequence[str],
    min_training_seasons: int = 3,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding per-league season folds without random splitting."""
    if len(leagues) != len(seasons):
        raise ValueError("leagues and seasons must have equal lengths")
    if min_training_seasons < 1:
        raise ValueError("min_training_seasons must be positive")
    unique = sorted(set(zip(leagues, seasons, strict=True)))
    result: list[WalkForwardFold] = []
    for league in sorted({value for value, _ in unique}):
        league_seasons = sorted({season for row_league, season in unique if row_league == league})
        for test_position in range(min_training_seasons, len(league_seasons)):
            test_season = league_seasons[test_position]
            train_indices = tuple(
                index
                for index, (row_league, season) in enumerate(zip(leagues, seasons, strict=True))
                if row_league == league and season < test_season
            )
            test_indices = tuple(
                index
                for index, (row_league, season) in enumerate(zip(leagues, seasons, strict=True))
                if row_league == league and season == test_season
            )
            if train_indices and test_indices:
                result.append(WalkForwardFold(league, test_season, train_indices, test_indices))
    return tuple(result)


class TemperatureScaler:
    """Chronologically fitted scalar temperature for multiclass probabilities."""

    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(
        self, probabilities: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> TemperatureScaler:
        """Fit temperature by minimizing chronological validation log loss."""
        _validate_probabilities(probabilities, labels)
        logits = np.log(np.clip(probabilities, 1e-15, 1.0))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            scaled = _softmax(logits / temperature)
            return multiclass_log_loss(labels, scaled)

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"Temperature calibration failed: {result.message}")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply a fitted temperature to probability rows."""
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("probabilities must have shape (n, 3)")
        return _softmax(np.log(np.clip(probabilities, 1e-15, 1.0)) / self.temperature)


def walk_forward_predict(
    features: pd.DataFrame,
    labels: Sequence[int],
    *,
    leagues: Sequence[str],
    seasons: Sequence[str],
    estimator_factory: EstimatorFactory,
    min_training_seasons: int = 3,
    calibrate: bool = True,
) -> pd.DataFrame:
    """Fit chronological models, calibrate on the prior season, and predict each fold."""
    y = np.asarray(labels, dtype=np.int64)
    if len(features) != y.size:
        raise ValueError("features and labels must have equal lengths")
    if len(leagues) != y.size or len(seasons) != y.size:
        raise ValueError("features, leagues, and seasons must have equal lengths")
    numeric = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] == 0:
        raise ValueError("at least one numeric feature is required")
    records: list[dict[str, Any]] = []

    for fold in make_walk_forward_folds(
        leagues=leagues,
        seasons=seasons,
        min_training_seasons=min_training_seasons,
    ):
        train = np.asarray(fold.train_indices)
        test = np.asarray(fold.test_indices)
        train_seasons = sorted({seasons[index] for index in train})
        calibrator: TemperatureScaler | None = None
        fit_indices = train

        if calibrate and len(train_seasons) >= 2:
            calibration_season = train_seasons[-1]
            inner_train = train[
                np.asarray([seasons[index] != calibration_season for index in train])
            ]
            calibration = train[
                np.asarray([seasons[index] == calibration_season for index in train])
            ]
            if inner_train.size and calibration.size:
                calibration_model = estimator_factory()
                calibration_model.fit(numeric.iloc[inner_train], y[inner_train])
                calibration_probs = _aligned_probabilities(
                    calibration_model, numeric.iloc[calibration]
                )
                calibrator = TemperatureScaler().fit(calibration_probs, y[calibration])

        model = estimator_factory()
        model.fit(numeric.iloc[fit_indices], y[fit_indices])
        raw_probs = _aligned_probabilities(model, numeric.iloc[test])
        calibrated_probs = calibrator.transform(raw_probs) if calibrator else raw_probs
        for row_index, raw, calibrated in zip(test, raw_probs, calibrated_probs, strict=True):
            records.append(
                {
                    "index": int(row_index),
                    "league": leagues[row_index],
                    "season": seasons[row_index],
                    "y_true": int(y[row_index]),
                    "p_home": float(calibrated[0]),
                    "p_draw": float(calibrated[1]),
                    "p_away": float(calibrated[2]),
                    "raw_p_home": float(raw[0]),
                    "raw_p_draw": float(raw[1]),
                    "raw_p_away": float(raw[2]),
                    "temperature": calibrator.temperature if calibrator else None,
                }
            )
    return pd.DataFrame.from_records(records)


def paired_matchweek_bootstrap(
    model_losses: Sequence[float],
    baseline_losses: Sequence[float],
    *,
    leagues: Sequence[str],
    seasons: Sequence[str],
    matchweeks: Sequence[int],
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return observed model-minus-baseline loss and paired 95% cluster-bootstrap CI."""
    model = np.asarray(model_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    if not (model.size == baseline.size == len(leagues) == len(seasons) == len(matchweeks)):
        raise ValueError("losses, leagues, seasons, and matchweeks must have equal lengths")
    if model.size == 0 or replicates < 1:
        raise ValueError("at least one observation and one bootstrap replicate are required")
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(baseline)):
        raise ValueError("losses must be finite")

    groups: dict[tuple[str, str], dict[int, list[int]]] = {}
    for index, (league, season, week) in enumerate(zip(leagues, seasons, matchweeks, strict=True)):
        groups.setdefault((league, season), {}).setdefault(int(week), []).append(index)
    rng = np.random.default_rng(seed)
    differences = model - baseline
    samples = np.empty(replicates, dtype=np.float64)

    for replicate in range(replicates):
        sampled: list[int] = []
        for weeks in groups.values():
            week_ids = list(weeks)
            selected = rng.choice(week_ids, size=len(week_ids), replace=True)
            for week in selected:
                sampled.extend(weeks[int(week)])
        samples[replicate] = float(differences[sampled].mean())

    observed = float(differences.mean())
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _aligned_probabilities(model: Any, features: pd.DataFrame) -> NDArray[np.float64]:
    probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), 3), dtype=np.float64)
    for column, class_value in enumerate(model.classes_):
        aligned[:, int(class_value)] = probabilities[:, column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    missing = row_sums[:, 0] <= 0
    if np.any(missing):
        raise ValueError("model produced an all-zero probability row")
    return aligned / row_sums


def _validate_probabilities(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> None:
    if probabilities.ndim != 2 or probabilities.shape != (labels.size, 3):
        raise ValueError("probabilities must have shape (n, 3)")
    if labels.ndim != 1 or not np.all(np.isin(labels, (0, 1, 2))):
        raise ValueError("labels must be one-dimensional H/D/A integers")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to 1")


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)

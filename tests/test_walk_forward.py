"""Tests for chronological folds, calibration, and clustered confidence intervals."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from forecast.eval.walk_forward import (
    TemperatureScaler,
    make_walk_forward_folds,
    paired_matchweek_bootstrap,
    walk_forward_predict,
)


def test_folds_are_expanding_and_never_train_on_test_season() -> None:
    seasons = ["2020-21"] * 2 + ["2021-22"] * 2 + ["2022-23"] * 2 + ["2023-24"] * 2
    folds = make_walk_forward_folds(
        leagues=["PL"] * len(seasons),
        seasons=seasons,
        min_training_seasons=3,
    )

    assert len(folds) == 1
    assert folds[0].test_season == "2023-24"
    assert set(seasons[index] for index in folds[0].train_indices) == {
        "2020-21",
        "2021-22",
        "2022-23",
    }
    assert set(seasons[index] for index in folds[0].test_indices) == {"2023-24"}


def test_walk_forward_prediction_returns_calibrated_test_season_only() -> None:
    seasons = [season for season in ("2020-21", "2021-22", "2022-23", "2023-24") for _ in range(12)]
    labels = np.asarray([index % 3 for index in range(len(seasons))])
    features = pd.DataFrame({"signal": labels.astype(float), "noise": np.arange(len(labels))})

    predictions = walk_forward_predict(
        features,
        labels,
        leagues=["PL"] * len(seasons),
        seasons=seasons,
        estimator_factory=lambda: LogisticRegression(max_iter=200),
        calibrate=True,
    )

    assert len(predictions) == 12
    assert set(predictions["season"]) == {"2023-24"}
    assert predictions[["p_home", "p_draw", "p_away"]].sum(axis=1).to_numpy() == pytest.approx(
        np.ones(12)
    )
    assert predictions["temperature"].notna().all()


def test_temperature_scaler_preserves_probability_simplex() -> None:
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    labels = np.asarray([0, 1, 2])
    scaler = TemperatureScaler().fit(probabilities, labels)
    transformed = scaler.transform(probabilities)

    assert np.all(transformed > 0)
    assert transformed.sum(axis=1) == pytest.approx(np.ones(3))


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    args = {
        "model_losses": [0.2, 0.3, 0.4, 0.5],
        "baseline_losses": [0.3, 0.4, 0.5, 0.6],
        "leagues": ["PL"] * 4,
        "seasons": ["2023-24"] * 4,
        "matchweeks": [1, 1, 2, 2],
        "replicates": 100,
        "seed": 7,
    }
    first = paired_matchweek_bootstrap(**args)
    second = paired_matchweek_bootstrap(**args)

    assert first == second
    assert first[0] == pytest.approx(-0.1)
    assert first[1] == pytest.approx(-0.1)
    assert first[2] == pytest.approx(-0.1)

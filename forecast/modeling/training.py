"""Reproducible LightGBM/logistic training with nested chronological validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from forecast.eval.metrics import multiclass_log_loss
from forecast.eval.walk_forward import (
    TemperatureScaler,
    _aligned_probabilities,
    make_walk_forward_folds,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True, slots=True)
class FoldArtifact:
    """Paths and provenance for one saved walk-forward model fold."""

    league: str
    test_season: str
    model_version: str
    model_path: Path
    metadata_path: Path


def lightgbm_pipeline(params: dict[str, Any], *, seed: int = 42) -> Pipeline:
    """Create a deterministic numeric-imputation and LightGBM pipeline."""
    classifier = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", classifier),
        ]
    )


def logistic_pipeline(*, seed: int = 42) -> Pipeline:
    """Create the regularized multinomial logistic comparison model."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def search_lightgbm_params(
    features: pd.DataFrame,
    labels: NDArray[np.int64],
    *,
    seasons: list[str],
    seed: int = 42,
    trials: int = 20,
) -> dict[str, Any]:
    """Tune parameters on an inner chronological validation season only."""
    if trials < 1:
        raise ValueError("trials must be positive")
    ordered_seasons = sorted(set(seasons))
    if len(ordered_seasons) < 2:
        return _default_params()
    validation_season = ordered_seasons[-1]
    train_indices = np.asarray(
        [index for index, season in enumerate(seasons) if season < validation_season]
    )
    validation_indices = np.asarray(
        [index for index, season in enumerate(seasons) if season == validation_season]
    )
    if not train_indices.size or not validation_indices.size:
        return _default_params()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80, step=10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        model = lightgbm_pipeline(params, seed=seed)
        model.fit(features.iloc[train_indices], labels[train_indices])
        probabilities = _aligned_probabilities(model, features.iloc[validation_indices])
        return multiclass_log_loss(labels[validation_indices], probabilities)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, n_jobs=1)
    return {**_default_params(), **study.best_params}


def train_walk_forward(
    features: pd.DataFrame,
    labels: list[int] | NDArray[np.int64],
    *,
    leagues: list[str],
    seasons: list[str],
    model_dir: Path,
    seed: int = 42,
    optuna_trials: int = 20,
    min_training_seasons: int = 3,
    kickoffs: list[datetime] | None = None,
    forecast_cutoffs: list[datetime] | None = None,
    label_available_at: list[datetime | None] | None = None,
) -> tuple[pd.DataFrame, tuple[FoldArtifact, ...]]:
    """Evaluate models by expanding season folds and save versioned fold artifacts."""
    y = np.asarray(labels, dtype=np.int64)
    if len(features) != y.size or len(leagues) != y.size or len(seasons) != y.size:
        raise ValueError("features, labels, leagues, and seasons must have equal lengths")
    if (kickoffs is None) != (label_available_at is None):
        raise ValueError("kickoffs and label_available_at must be provided together")
    if forecast_cutoffs is not None and kickoffs is None:
        raise ValueError("forecast_cutoffs require kickoffs and label_available_at")
    if kickoffs is not None and (
        len(kickoffs) != y.size or len(label_available_at or []) != y.size
    ):
        raise ValueError("kickoffs and label availability must match feature rows")
    if forecast_cutoffs is not None and len(forecast_cutoffs) != y.size:
        raise ValueError("forecast_cutoffs must match feature rows")
    if not np.all(np.isin(y, [0, 1, 2])):
        raise ValueError("labels must be H/D/A integers 0, 1, or 2")
    numeric = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] == 0:
        raise ValueError("training requires at least one numeric feature")
    folds = make_walk_forward_folds(
        leagues=leagues,
        seasons=seasons,
        min_training_seasons=min_training_seasons,
    )
    if not folds:
        raise ValueError(
            f"no walk-forward folds: need at least {min_training_seasons + 1} seasons per league"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    artifacts: list[FoldArtifact] = []

    for fold in folds:
        train = np.asarray(fold.train_indices, dtype=np.int64)
        test = np.asarray(fold.test_indices, dtype=np.int64)
        if kickoffs is not None and label_available_at is not None:
            eligible_test = [
                index
                for index in test
                if label_available_at[index] is not None
                and _as_utc(label_available_at[index]) >= _as_utc(kickoffs[index])
            ]
            test = np.asarray(eligible_test, dtype=np.int64)
            if test.size == 0:
                continue
            cutoffs = forecast_cutoffs or kickoffs
            test_cutoff = min(_as_utc(cutoffs[index]) for index in test)
            train = np.asarray(
                [
                    index
                    for index in train
                    if label_available_at[index] is not None
                    and _as_utc(label_available_at[index]) >= _as_utc(kickoffs[index])
                    and _as_utc(label_available_at[index]) < test_cutoff
                ],
                dtype=np.int64,
            )
            if len({seasons[index] for index in train}) < min_training_seasons:
                continue
        training_seasons = sorted({seasons[index] for index in train})
        calibration_season = training_seasons[-1]
        inner_training = train[
            np.asarray([seasons[index] != calibration_season for index in train])
        ]
        calibration = train[np.asarray([seasons[index] == calibration_season for index in train])]
        tune_seasons = [seasons[index] for index in inner_training]
        tune_features = numeric.iloc[inner_training].reset_index(drop=True)
        tune_labels = y[inner_training]
        params = (
            search_lightgbm_params(
                tune_features,
                tune_labels,
                seasons=tune_seasons,
                seed=seed,
                trials=optuna_trials,
            )
            if inner_training.size
            else _default_params()
        )

        calibrator: TemperatureScaler | None = None
        if inner_training.size and calibration.size and len(set(y[calibration])) >= 2:
            calibration_model = lightgbm_pipeline(params, seed=seed)
            calibration_model.fit(numeric.iloc[inner_training], y[inner_training])
            calibration_probs = _aligned_probabilities(calibration_model, numeric.iloc[calibration])
            calibrator = TemperatureScaler().fit(calibration_probs, y[calibration])

        lgbm = lightgbm_pipeline(params, seed=seed)
        lgbm.fit(numeric.iloc[train], y[train])
        logistic = logistic_pipeline(seed=seed)
        logistic.fit(numeric.iloc[train], y[train])
        raw = _aligned_probabilities(lgbm, numeric.iloc[test])
        calibrated = calibrator.transform(raw) if calibrator else raw
        logistic_probs = _aligned_probabilities(logistic, numeric.iloc[test])

        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "league": fold.league,
                    "season": fold.test_season,
                    "features": list(numeric.columns),
                    "params": params,
                    "seed": seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        model_version = f"lgbm-{fold.league}-{fold.test_season}-{content_hash}"
        fold_dir = model_dir / fold.league / fold.test_season
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / f"{model_version}.joblib"
        metadata_path = fold_dir / f"{model_version}.json"
        joblib.dump(
            {
                "model": lgbm,
                "calibrator": calibrator,
                "features": list(numeric.columns),
                "model_version": model_version,
            },
            model_path,
        )
        metadata = {
            "model_version": model_version,
            "league": fold.league,
            "test_season": fold.test_season,
            "training_seasons": training_seasons,
            "calibration_season": calibration_season if calibrator else None,
            "calibration_temperature": calibrator.temperature if calibrator else None,
            "parameters": params,
            "features": list(numeric.columns),
            "random_seed": seed,
            "git_sha": os.environ.get("GIT_SHA"),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            FoldArtifact(
                fold.league,
                fold.test_season,
                model_version,
                model_path,
                metadata_path,
            )
        )
        for local_index, index in enumerate(test):
            for name, probabilities in (
                ("logistic", logistic_probs),
                ("lightgbm_raw", raw),
                ("lightgbm_calibrated", calibrated),
            ):
                records.append(
                    {
                        "index": int(index),
                        "league": leagues[index],
                        "season": seasons[index],
                        "y_true": int(y[index]),
                        "forecaster": name,
                        "model_version": model_version
                        if name.startswith("lightgbm")
                        else "logistic",
                        "p_home": float(probabilities[local_index, 0]),
                        "p_draw": float(probabilities[local_index, 1]),
                        "p_away": float(probabilities[local_index, 2]),
                    }
                )
    if not artifacts:
        raise ValueError(
            "no leakage-safe walk-forward folds: historical training labels were not "
            "observed before eligible test-season kickoffs"
        )
    return pd.DataFrame.from_records(records), tuple(artifacts)


def _default_params() -> dict[str, Any]:
    return {
        "n_estimators": 200,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
    }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

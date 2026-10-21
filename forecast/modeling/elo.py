"""Football Elo baseline with home advantage and learned draw probabilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

ELO_MEAN = 1500.0
HOME_ADVANTAGE = 65.0
K_FACTOR = 20.0


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class EloBaseline:
    """Replay verified past results and learn H/D/A probabilities from rating gaps."""

    def __init__(
        self,
        *,
        home_advantage: float = HOME_ADVANTAGE,
        k_factor: float = K_FACTOR,
        season_regression: float = 0.75,
        random_seed: int = 42,
    ) -> None:
        if not 0 <= season_regression <= 1:
            raise ValueError("season_regression must be between 0 and 1")
        self.home_advantage = home_advantage
        self.k_factor = k_factor
        self.season_regression = season_regression
        self.random_seed = random_seed
        self.ratings: dict[str, float] = {}
        self.current_season: str | None = None
        self.classifier: LogisticRegression | None = None

    def fit(self, matches: Iterable[Mapping[str, Any]], *, through: datetime) -> EloBaseline:
        """Fit rating state and ordered-outcome model using available past results only."""
        cutoff = _utc(through)
        self.ratings.clear()
        self.current_season = None
        self.classifier = None
        ordered = sorted(
            (
                row
                for row in matches
                if row.get("result_available_at") is not None
                and _utc(row["result_available_at"]) < cutoff
                and _utc(row["kickoff_utc"]) < cutoff
                and row.get("home_goals") is not None
                and row.get("away_goals") is not None
            ),
            key=lambda row: (_utc(row["kickoff_utc"]), str(row.get("id", ""))),
        )
        x: list[list[float]] = []
        y: list[int] = []
        for row in ordered:
            season = str(row["season"])
            self._start_season(season)
            home = str(row["home_team_id"])
            away = str(row["away_team_id"])
            home_rating = self.ratings.get(home, ELO_MEAN)
            away_rating = self.ratings.get(away, ELO_MEAN)
            x.append([home_rating - away_rating + self.home_advantage])
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            y.append(0 if home_goals > away_goals else 1 if home_goals == away_goals else 2)
            self._update(home, away, home_rating, away_rating, home_goals, away_goals)

        if len(set(y)) < 2:
            raise ValueError("Elo draw/outcome model requires at least two observed outcomes")
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed,
        )
        self.classifier.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
        return self

    def predict_proba(
        self, home_team: str, away_team: str, *, season: str | None = None
    ) -> list[float]:
        """Predict H/D/A probabilities from current team ratings."""
        if self.classifier is None:
            raise RuntimeError("EloBaseline must be fitted before prediction")
        if season is not None:
            self._start_season(season)
        difference = (
            self.ratings.get(home_team, ELO_MEAN)
            - self.ratings.get(away_team, ELO_MEAN)
            + self.home_advantage
        )
        predicted = self.classifier.predict_proba(np.asarray([[difference]], dtype=np.float64))[0]
        probabilities = np.zeros(3, dtype=np.float64)
        probabilities[self.classifier.classes_.astype(int)] = predicted
        return probabilities.tolist()

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        *,
        season: str,
    ) -> None:
        """Update ratings after a newly available result."""
        self._start_season(season)
        home = self.ratings.get(home_team, ELO_MEAN)
        away = self.ratings.get(away_team, ELO_MEAN)
        self._update(home_team, away_team, home, away, home_goals, away_goals)

    def _start_season(self, season: str) -> None:
        if self.current_season is not None and season != self.current_season:
            self.ratings = {
                team: ELO_MEAN + self.season_regression * (rating - ELO_MEAN)
                for team, rating in self.ratings.items()
            }
        self.current_season = season

    def _update(
        self,
        home_team: str,
        away_team: str,
        home_rating: float,
        away_rating: float,
        home_goals: int,
        away_goals: int,
    ) -> None:
        expected_home = 1 / (1 + 10 ** (-(home_rating + self.home_advantage - away_rating) / 400))
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        goal_difference = abs(home_goals - away_goals)
        multiplier = (
            math.log1p(goal_difference) * 2.2 / (abs(home_rating - away_rating) * 0.001 + 2.2)
            if goal_difference
            else 1.0
        )
        delta = self.k_factor * multiplier * (actual_home - expected_home)
        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta

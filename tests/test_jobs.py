"""Tests for scheduled live prediction policy and idempotency."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.config import Settings
from forecast.db import create_schema, get_engine
from forecast.jobs import predict_upcoming
from forecast.models import LedgerBatch, Match, Prediction, Team


class FixedModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, _features):
        return np.asarray([[0.55, 0.25, 0.20]])


class IdentityCalibrator:
    def transform(self, probabilities):
        return probabilities


def add_history(session: Session, home_id: int, away_id: int, now: datetime) -> None:
    results = ((2, 0), (0, 0), (0, 1), (3, 1), (1, 1), (0, 2))
    for index, (home_goals, away_goals) in enumerate(results):
        kickoff = now - timedelta(days=30 - index)
        session.add(
            Match(
                id=f"history-{index}",
                league="PL",
                season="2025-26",
                kickoff_utc=kickoff,
                home_team_id=home_id if index % 2 == 0 else away_id,
                away_team_id=away_id if index % 2 == 0 else home_id,
                home_goals=home_goals,
                away_goals=away_goals,
                result="H" if home_goals > away_goals else "A" if home_goals < away_goals else "D",
                result_available_at=kickoff + timedelta(hours=3),
                status="finished",
                source="test",
            )
        )


def test_live_prediction_publishes_model_and_elo_once(tmp_path, monkeypatch) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    now = datetime(2026, 10, 21, 18, tzinfo=UTC)
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        session.add_all((home, away))
        session.flush()
        add_history(session, home.id, away.id, now)
        session.add(
            Match(
                id="upcoming",
                league="PL",
                season="2026-27",
                kickoff_utc=now + timedelta(hours=25),
                home_team_id=home.id,
                away_team_id=away.id,
                status="scheduled",
                source="test",
            )
        )
        session.commit()

    settings = replace(
        Settings.from_env(),
        models_dir=tmp_path / "models",
        ledger_dir=tmp_path / "ledger",
    )
    settings.models_dir.mkdir()
    monkeypatch.setattr(
        "forecast.jobs.load_production_model",
        lambda **_kwargs: {
            "model_version": "model-v1",
            "features": ["elo_difference", "home_points_per_game_last_3"],
            "model": FixedModel(),
            "calibrator": IdentityCalibrator(),
        },
    )
    dry_run = predict_upcoming(
        engine,
        settings=settings,
        now=now,
        dry_run=True,
    )
    assert dry_run["predictions_created"] == 2
    assert dry_run["items"][0]["p_home"] == pytest.approx(0.55)
    assert dry_run["items"][0]["elo"] is not None
    assert dry_run["batch_id"] is None
    with Session(engine) as session:
        assert session.scalar(select(Prediction.id)) is None

    published = predict_upcoming(
        engine,
        settings=settings,
        now=now,
    )
    assert published["predictions_created"] == 2
    assert (tmp_path / "ledger" / f"2026-10-21_{published['batch_id']}.json").exists()
    with Session(engine) as session:
        rows = session.scalars(select(Prediction)).all()
        assert {row.model_version for row in rows} == {"model-v1", "elo-baseline-v1"}
        assert len(rows) == 2
        assert session.query(LedgerBatch).count() == 1

    retry = predict_upcoming(engine, settings=settings, now=now)
    assert retry["predictions_created"] == 0
    assert retry["batch_id"] is None
    with Session(engine) as session:
        assert session.query(Prediction).count() == 2


def test_policy_rejects_window_shorter_than_day(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 24"):
        predict_upcoming(
            get_engine("sqlite:///:memory:"),
            settings=Settings.from_env(),
            now=datetime.now(UTC),
            window_hours=12,
        )

"""Contract tests for the read-only scoreboard API."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from forecast.api.app import create_app
from forecast.config import Settings
from forecast.db import create_schema, get_engine
from forecast.models import LedgerBatch, Match, Prediction, Score, Team


def test_read_only_api_endpoints(tmp_path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    now = datetime.now(UTC).replace(microsecond=0)
    kickoff = now + timedelta(hours=25)
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        session.add_all((home, away))
        session.flush()
        future = Match(
            id="future",
            league="PL",
            season="2026-27",
            kickoff_utc=kickoff,
            home_team_id=home.id,
            away_team_id=away.id,
            status="scheduled",
            source="test",
        )
        past = Match(
            id="past",
            league="PL",
            season="2026-27",
            kickoff_utc=now - timedelta(days=2),
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=2,
            away_goals=1,
            result="H",
            result_available_at=now - timedelta(days=2) + timedelta(hours=3),
            status="finished",
            source="test",
        )
        session.add_all((future, past))
        session.flush()
        batch = LedgerBatch(
            batch_id="batch-1",
            created_at=now,
            sha256="b" * 64,
            n_predictions=1,
            timestamp_proof_status="pending",
        )
        session.add(batch)
        session.flush()
        session.add_all(
            (
                Prediction(
                    id="prediction-future",
                    match_id="future",
                    model_version="v1",
                    forecast_policy_version="t-minus-24h-v1",
                    p_home=0.6,
                    p_draw=0.25,
                    p_away=0.15,
                    created_at=now,
                    features_hash="a" * 64,
                    batch_id=batch.batch_id,
                ),
                Prediction(
                    id="prediction-past",
                    match_id="past",
                    model_version="v1",
                    forecast_policy_version="t-minus-24h-v1",
                    p_home=0.6,
                    p_draw=0.25,
                    p_away=0.15,
                    created_at=now - timedelta(days=2, hours=25),
                    features_hash="c" * 64,
                    batch_id=batch.batch_id,
                ),
                Score(
                    match_id="past",
                    forecaster="model",
                    version="v1",
                    metric="log_loss",
                    value=0.51,
                    computed_at=now,
                    metadata_json={"league": "PL", "season": "2026-27"},
                ),
            )
        )
        session.commit()

    settings = replace(Settings.from_env(), ledger_dir=tmp_path)
    client = TestClient(create_app(engine=engine, settings=settings))

    assert client.get("/health").json() == {"status": "ok", "database": "ok"}
    upcoming = client.get("/predictions/upcoming?league=PL")
    assert upcoming.status_code == 200
    upcoming_items = upcoming.json()["items"]
    assert len(upcoming_items) == 1
    assert upcoming_items[0]["match"]["home_team"] == "Home"
    assert upcoming_items[0]["prediction"]["p_home"] == 0.6
    assert upcoming.headers["cache-control"].startswith("public")

    detail = client.get("/predictions/future")
    assert detail.status_code == 200
    assert detail.json()["ledger_batch_hash"] == "b" * 64
    assert client.get("/predictions/missing").status_code == 404

    scorecard = client.get("/scorecard").json()
    assert scorecard["matches_scored"] == 1
    assert scorecard["scores"][0]["value"] == 0.51
    calibration = client.get("/scorecard/calibration").json()
    assert calibration["matches_scored"] == 1
    assert calibration["bins"][0]["count"] == 1

    recent = client.get("/matches/recent").json()["items"]
    assert len(recent) == 1
    assert recent[0]["result"] == "H"
    assert client.get("/verify/batch-1").json()["verified"] is False

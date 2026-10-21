"""Tests for persisted scores, market coverage, and paired loss intervals."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.eval.scorecard import evaluate_forecasts
from forecast.models import LedgerBatch, Match, Odds, Prediction, Score, Team


def test_scorecard_compares_models_only_on_shared_closing_market_matches() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    now = datetime(2026, 10, 21, 18, tzinfo=UTC)
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        session.add_all((home, away))
        session.flush()
        matches = []
        for index, result in enumerate(("H", "D")):
            kickoff = now - timedelta(days=7 + index)
            match = Match(
                id=f"match-{index}",
                league="PL",
                season="2026-27",
                kickoff_utc=kickoff,
                matchweek=1,
                home_team_id=home.id,
                away_team_id=away.id,
                home_goals=1 if result == "H" else 0,
                away_goals=0,
                result=result,
                result_available_at=kickoff + timedelta(hours=3),
                status="finished",
                source="test",
            )
            matches.append(match)
            session.add(match)
        batch = LedgerBatch(
            batch_id="batch",
            created_at=now - timedelta(days=8),
            sha256="b" * 64,
            n_predictions=4,
        )
        session.add(batch)
        session.flush()
        for match, probabilities, version in (
            (matches[0], (0.7, 0.2, 0.1), "model-a"),
            (matches[1], (0.1, 0.6, 0.3), "model-a"),
            (matches[0], (0.6, 0.25, 0.15), "model-b"),
            (matches[1], (0.2, 0.5, 0.3), "model-b"),
        ):
            session.add(
                Prediction(
                    id=f"{version}-{match.id}",
                    match_id=match.id,
                    model_version=version,
                    forecast_policy_version="t-minus-24h-v1",
                    p_home=probabilities[0],
                    p_draw=probabilities[1],
                    p_away=probabilities[2],
                    created_at=now - timedelta(days=8),
                    features_hash="a" * 64,
                    batch_id="batch",
                )
            )
        session.add(
            Odds(
                match_id="match-0",
                source="Pinnacle",
                home=2.0,
                draw=3.5,
                away=4.0,
                kind="close",
                available_at=None,
                retrieved_at=now,
                availability_verified=False,
            )
        )
        session.commit()

        report = evaluate_forecasts(
            session,
            computed_at=now,
            bootstrap_replicates=100,
            seed=42,
        )
        session.commit()

        assert report["model_matches_scored"] == 2
        assert report["market_matches_scored"] == 1
        assert report["market_missing_matches"] == 1
        assert report["market_coverage"] == {"PL:2026-27": 1}
        assert set(report["paired_model_market_intervals"]) == {
            "model-a:t-minus-24h-v1:PL:2026-27",
            "model-b:t-minus-24h-v1:PL:2026-27",
        }
        model_scores = session.scalars(
            select(Score).where(Score.forecaster == "model", Score.metric == "log_loss")
        ).all()
        assert len(model_scores) == 4
        market_scores = session.scalars(
            select(Score).where(Score.forecaster == "market", Score.metric == "log_loss")
        ).all()
        assert len(market_scores) == 1
        assert market_scores[0].metadata_json["availability"] == "retrospective_unverified"
        assert report["scores_written"] == 15

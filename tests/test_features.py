"""Leakage guards for per-fixture feature generation."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.features.build import build_features
from forecast.models import Match, Team


def utc(value: str) -> datetime:
    """Parse an ISO UTC test timestamp."""
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def test_feature_builder_uses_only_results_available_before_cutoff() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    cutoff = utc("2026-10-21T12:00:00+00:00")
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        rival = Team(canonical_name="Rival", league="PL")
        session.add_all((home, away, rival))
        session.flush()
        session.add_all(
            (
                Match(
                    id="old",
                    league="PL",
                    season="2026-27",
                    kickoff_utc=utc("2026-10-10T14:00:00+00:00"),
                    home_team_id=home.id,
                    away_team_id=rival.id,
                    home_goals=3,
                    away_goals=0,
                    result="H",
                    result_available_at=utc("2026-10-10T17:00:00+00:00"),
                    status="finished",
                    source="test",
                ),
                Match(
                    id="late-result",
                    league="PL",
                    season="2026-27",
                    kickoff_utc=utc("2026-10-20T14:00:00+00:00"),
                    home_team_id=away.id,
                    away_team_id=rival.id,
                    home_goals=0,
                    away_goals=2,
                    result="A",
                    result_available_at=utc("2026-10-21T12:00:00+00:00"),
                    status="finished",
                    source="test",
                ),
                Match(
                    id="target",
                    league="PL",
                    season="2026-27",
                    kickoff_utc=utc("2026-10-22T19:00:00+00:00"),
                    home_team_id=home.id,
                    away_team_id=away.id,
                    status="scheduled",
                    source="test",
                ),
            )
        )
        session.commit()

        features = build_features(
            session,
            match_id="target",
            as_of_kickoff=cutoff,
        )
        assert features.values["home_matches_last_3"] == 1
        assert features.values["away_matches_last_3"] == 0
        assert features.values["season_progress"] == pytest.approx(1 / 380)
        assert features.values["home_elo"] > 1500

        future = session.get(Match, "late-result")
        assert future is not None
        future.home_goals = 9
        future.away_goals = 0
        future.result = "H"
        session.commit()
        after_future_mutation = build_features(
            session,
            match_id="target",
            as_of_kickoff=cutoff,
        )

    assert features.values == after_future_mutation.values
    assert features.features_hash == after_future_mutation.features_hash


def test_feature_builder_rejects_cutoff_at_kickoff() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    kickoff = utc("2026-10-22T19:00:00+00:00")
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        session.add_all((home, away))
        session.flush()
        session.add(
            Match(
                id="target",
                league="PL",
                season="2026-27",
                kickoff_utc=kickoff,
                home_team_id=home.id,
                away_team_id=away.id,
                status="scheduled",
                source="test",
            )
        )
        session.commit()

        with pytest.raises(ValueError, match="strictly before kickoff"):
            build_features(session, match_id="target", as_of_kickoff=kickoff)

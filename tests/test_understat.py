import json
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.ingestion.understat import UnderstatClient, ingest_understat_xg
from forecast.models import ExpectedGoals, Match, Team


def understat_match() -> dict:
    return {
        "id": "1",
        "isResult": True,
        "datetime": "2020-08-15 15:00:00",
        "h": {"title": "North FC"},
        "a": {"title": "South FC"},
        "xG": {"h": "1.25", "a": "0.75"},
    }


def test_understat_client_parses_embedded_json() -> None:
    payload = [understat_match()]
    embedded = json.dumps(payload, separators=(",", ":"))
    html = f"var datesData = JSON.parse('{embedded}');"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=html, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    understat = UnderstatClient(client=client, base_url="https://example.test")

    assert understat.get_matches(league="PL", season="2020-21") == payload
    assert requests[0].url.path == "/league/EPL/2020"
    client.close()


def test_xg_ingestion_is_idempotent_and_records_first_observation() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    observed_at = datetime(2025, 1, 1, tzinfo=UTC)
    with Session(engine) as session:
        home = Team(canonical_name="North FC", league="PL")
        away = Team(canonical_name="South FC", league="PL")
        session.add_all([home, away])
        session.flush()
        fixture = Match(
            id="fixture-1",
            league="PL",
            season="2020-21",
            kickoff_utc=datetime(2020, 8, 15, 15, tzinfo=UTC),
            kickoff_time_known=True,
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=2,
            away_goals=1,
            result="H",
            result_available_at=datetime(2020, 8, 15, 17, tzinfo=UTC),
            status="finished",
            source="test",
            source_match_id="fixture-1",
        )
        session.add(fixture)
        session.flush()

        assert ingest_understat_xg(
            session,
            league="PL",
            season="2020-21",
            matches=[understat_match()],
            retrieved_at=observed_at,
        ) == {"read": 1, "inserted": 1, "unchanged": 0, "unmatched": 0}
        assert ingest_understat_xg(
            session,
            league="PL",
            season="2020-21",
            matches=[{**understat_match(), "xG": {"h": "2.0", "a": "1.0"}}],
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        ) == {"read": 1, "inserted": 0, "unchanged": 1, "unmatched": 0}

        record = session.scalar(select(ExpectedGoals))
        assert record is not None
        assert record.home_xg == 1.25
        assert record.available_at.replace(tzinfo=UTC) == observed_at
        assert record.retrieved_at.replace(tzinfo=UTC) == observed_at
        assert record.availability_verified is True


def test_xg_ingestion_rejects_naive_retrieval_timestamp() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, pytest.raises(ValueError, match="timezone-aware"):
        ingest_understat_xg(
            session,
            league="PL",
            season="2020-21",
            matches=[],
            retrieved_at=datetime(2025, 1, 1),
        )

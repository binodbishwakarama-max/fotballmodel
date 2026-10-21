"""Tests for the live football-data.org client and fixture synchronization."""

from datetime import UTC, date, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.ingestion.football_data_org import FootballDataClient, sync_api_matches
from forecast.ingestion.historical import IngestionResult, _ingest_csv
from forecast.models import Match, MatchSource, Team


def api_match(status: str = "SCHEDULED") -> dict:
    return {
        "id": 505,
        "utcDate": "2025-10-22T19:00:00Z",
        "status": status,
        "season": {"startDate": "2026-08-01"},
        "homeTeam": {"name": "North FC"},
        "awayTeam": {"name": "South FC"},
        "score": {"fullTime": {"home": 2, "away": 1}},
    }


def create_seeded_database(engine) -> None:
    create_schema(engine)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "seed.csv"
        csv_path.write_text(
            "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n22/10/2025,20:00,North FC,South FC,,,\n",
            encoding="utf-8",
        )
        _ingest_csv(csv_path, "PL", "2026-27", engine, IngestionResult(), False)


def test_sync_api_fixtures_and_results_is_idempotent() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_seeded_database(engine)
    retrieved_at = datetime(2026, 10, 21, 18, tzinfo=UTC)

    with Session(engine) as session:
        counters = sync_api_matches(
            session,
            league="PL",
            matches=[api_match()],
            retrieved_at=retrieved_at,
        )
        session.commit()
        assert counters == {"read": 1, "inserted": 0, "updated": 0, "unchanged": 1}

        fixture = session.scalar(select(Match))
        assert fixture is not None
        assert (
            session.scalar(
                select(MatchSource.source_match_id).where(MatchSource.source == "football-data.org")
            )
            == "505"
        )
        assert fixture.source_match_id.startswith("PL:")
        assert fixture.status == "scheduled"
        assert fixture.result_available_at is None

        finished = sync_api_matches(
            session,
            league="PL",
            matches=[api_match("FINISHED")],
            retrieved_at=retrieved_at,
        )
        session.commit()
        assert finished["updated"] == 1
        assert fixture.result == "H"
        assert fixture.result_available_at.replace(tzinfo=UTC) == retrieved_at

        repeated = sync_api_matches(
            session,
            league="PL",
            matches=[api_match("FINISHED")],
            retrieved_at=datetime(2026, 10, 22, 22, tzinfo=UTC),
        )
        session.commit()
        assert repeated["unchanged"] == 1
        assert fixture.result_available_at.replace(tzinfo=UTC) == retrieved_at
        assert session.query(Match).count() == 1
        assert session.query(Team).count() == 2


def test_api_sync_rejects_unknown_team_names() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_seeded_database(engine)
    payload = api_match()
    payload["homeTeam"]["name"] = "Unrecognized FC"
    with Session(engine) as session, pytest.raises(ValueError, match="Unknown team name"):
        sync_api_matches(session, league="PL", matches=[payload])


def test_api_sync_rejects_result_observed_before_kickoff() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_seeded_database(engine)
    with Session(engine) as session, pytest.raises(ValueError, match="before its kickoff"):
        sync_api_matches(
            session,
            league="PL",
            matches=[api_match("FINISHED")],
            retrieved_at=datetime(2025, 10, 21, 18, tzinfo=UTC),
        )


def test_live_client_retries_rate_limit_and_uses_cache(tmp_path) -> None:
    requests = []
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "slow"}),
        httpx.Response(200, json={"matches": [api_match()]}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    sleeps = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = FootballDataClient(
        token="test",
        cache_dir=tmp_path,
        client=client,
        sleep=sleeps.append,
    )
    now = datetime(2026, 10, 21, 18, tzinfo=UTC)
    first = api.get_matches(
        league="PL",
        date_from=date(2026, 10, 21),
        date_to=date(2026, 10, 23),
        now=now,
    )
    second = api.get_matches(
        league="PL",
        date_from=date(2026, 10, 21),
        date_to=date(2026, 10, 23),
        now=now,
    )

    assert len(first) == 1
    assert second == first
    assert len(requests) == 2
    assert requests[0].headers["X-Auth-Token"] == "test"
    assert sleeps == [2.0]
    client.close()


def test_live_client_requires_token_and_rejects_invalid_league(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        FootballDataClient(token="", cache_dir=tmp_path)
    api = FootballDataClient(token="test", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unsupported league"):
        api.get_matches(league="CL", date_from=date.today(), date_to=date.today())
    api.close()

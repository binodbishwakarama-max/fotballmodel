"""Tests for historical source parsing and idempotent persistence."""

from datetime import UTC, date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.ingestion.historical import (
    _ingest_csv,
    _parse_kickoff,
    available_seasons,
    season_start_code,
)
from forecast.ingestion.teams import normalize_team_name, resolve_team
from forecast.models import IngestionRun, Match, Odds, Team


def test_season_codes_and_recent_seasons() -> None:
    assert season_start_code("2025-26") == "2526"
    assert available_seasons(today=date(2026, 10, 21), completed_count=2) == (
        ("2024-25", True),
        ("2025-26", True),
        ("2026-27", False),
    )
    with pytest.raises(ValueError, match="Invalid football season"):
        season_start_code("2025-27")


def test_team_name_normalization_handles_accents_and_punctuation() -> None:
    assert normalize_team_name("Atlético-Madrid") == "atletico madrid"


def test_kickoff_timezone_conversion_and_missing_time_flag() -> None:
    kickoff, known = _parse_kickoff("21/10/2026", "20:00", "PL")

    assert kickoff.isoformat() == "2026-10-21T19:00:00+00:00"
    assert known
    placeholder, known = _parse_kickoff("21/10/2026", "", "PL")
    assert placeholder.tzinfo == UTC
    assert not known


@pytest.mark.parametrize(
    ("match_date", "match_time"),
    [
        ("29/03/2026", "01:30"),
        ("25/10/2026", "01:30"),
    ],
)
def test_kickoff_parser_rejects_dst_ambiguous_or_missing_local_times(
    match_date: str, match_time: str
) -> None:
    with pytest.raises(ValueError):
        _parse_kickoff(match_date, match_time, "PL")


def test_unknown_secondary_source_team_fails_loudly() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, pytest.raises(ValueError, match="Unknown team name"):
        resolve_team(
            session,
            source="new-source",
            raw_name="Unknown FC",
            league="PL",
        )


def test_historical_csv_ingestion_is_idempotent_and_flags_odds_provenance(
    tmp_path,
) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSCH,PSCD,PSCA,B365CH,B365CD,B365CA\n"
        "21/10/2025,20:00,North FC,South FC,2,1,H,2.0,3.5,4.0,2.1,3.4,4.1\n",
        encoding="utf-8",
    )

    from forecast.ingestion.historical import IngestionResult

    first = IngestionResult()
    _ingest_csv(csv_file, "PL", "2025-26", engine, first, False)
    second = IngestionResult()
    _ingest_csv(csv_file, "PL", "2025-26", engine, second, False)

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Match)) == 1
        assert session.scalar(select(func.count()).select_from(Team)) == 2
        assert session.scalar(select(func.count()).select_from(Odds)) == 2
        match = session.scalar(select(Match))
        assert match is not None
        assert match.result == "H"
        assert match.result_available_at is not None
        assert match.result_available_at.replace(tzinfo=UTC) > match.kickoff_utc.replace(tzinfo=UTC)
        assert session.scalar(select(func.count()).select_from(IngestionRun)) == 2
        odds = session.scalars(select(Odds)).all()
        assert {item.kind for item in odds} == {"close"}
        assert all(item.available_at is None for item in odds)
        assert all(not item.availability_verified for item in odds)
    assert first.matches_inserted == 1
    assert second.matches_existing == 1
    assert second.odds_inserted == 0

"""Understat match xG retrieval with first-observed availability provenance."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.ingestion.teams import resolve_team
from forecast.models import ExpectedGoals, Match

UNDERSTAT_LEAGUES = {"PL": "EPL", "PD": "La_liga"}
_DATES_DATA = re.compile(r"datesData\s*=\s*JSON\.parse\('((?:\\.|[^'])*)'\)")
_MATCH_TIME_TOLERANCE_SECONDS = 36 * 60 * 60


class UnderstatClient:
    """Fetch and parse a season's completed match data from Understat."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://understat.com",
    ) -> None:
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={"User-Agent": "BWB-Forecast/0.1 (+https://github.com/)"},
        )
        self._owns_client = client is None
        self.base_url = base_url.rstrip("/")

    def close(self) -> None:
        """Close the HTTP client when owned by this instance."""
        if self._owns_client:
            self.client.close()

    def get_matches(self, *, league: str, season: str) -> list[dict[str, Any]]:
        """Fetch a completed season and extract Understat's embedded match JSON."""
        if league not in UNDERSTAT_LEAGUES:
            raise ValueError(f"Unsupported Understat league: {league}")
        season_start = _season_start(season)
        response = self.client.get(
            f"{self.base_url}/league/{UNDERSTAT_LEAGUES[league]}/{season_start}"
        )
        response.raise_for_status()
        match = _DATES_DATA.search(response.text)
        if match is None:
            raise ValueError("Understat page does not contain datesData")
        try:
            matches = json.loads(match.group(1).replace("\\'", "'"))
        except json.JSONDecodeError as exc:
            raise ValueError("Understat datesData is not valid JSON") from exc
        if not isinstance(matches, list):
            raise ValueError("Understat datesData must be a list")
        if not all(isinstance(item, dict) for item in matches):
            raise ValueError("Understat datesData contains a non-object match")
        return matches


def ingest_understat_xg(
    session: Session,
    *,
    league: str,
    season: str,
    matches: list[dict[str, Any]],
    retrieved_at: datetime | None = None,
) -> dict[str, int]:
    """Attach xG to known fixtures; never backdate its first-observed timestamp."""
    if league not in UNDERSTAT_LEAGUES:
        raise ValueError(f"Unsupported Understat league: {league}")
    _season_start(season)
    observed_at = retrieved_at or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    observed_at = observed_at.astimezone(UTC)
    counters = {"read": 0, "inserted": 0, "unchanged": 0, "unmatched": 0}

    for payload in matches:
        counters["read"] += 1
        if not isinstance(payload, dict):
            raise ValueError("Understat match payload must be an object")
        if payload.get("isResult") is not True:
            counters["unchanged"] += 1
            continue
        home_payload = payload.get("h")
        away_payload = payload.get("a")
        if not isinstance(home_payload, dict) or not isinstance(away_payload, dict):
            raise ValueError("Understat match is missing home/away team objects")
        home_name = _required_text(home_payload, "title")
        away_name = _required_text(away_payload, "title")
        home_team = resolve_team(
            session,
            source="Understat",
            raw_name=home_name,
            league=league,
        )
        away_team = resolve_team(
            session,
            source="Understat",
            raw_name=away_name,
            league=league,
        )
        kickoff = _parse_kickoff(payload.get("datetime"))
        xg = payload.get("xG")
        if not isinstance(xg, dict):
            raise ValueError("Understat match is missing xG values")
        home_xg = _finite_nonnegative(xg.get("h"), "home xG")
        away_xg = _finite_nonnegative(xg.get("a"), "away xG")
        fixture = _find_fixture(
            session,
            league=league,
            season=season,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            kickoff=kickoff,
        )
        if fixture is None:
            counters["unmatched"] += 1
            continue
        existing = session.scalar(
            select(ExpectedGoals).where(
                ExpectedGoals.match_id == fixture.id,
                ExpectedGoals.source == "Understat",
            )
        )
        if existing is not None:
            counters["unchanged"] += 1
            continue
        session.add(
            ExpectedGoals(
                match_id=fixture.id,
                home_xg=home_xg,
                away_xg=away_xg,
                source="Understat",
                available_at=observed_at,
                retrieved_at=observed_at,
                availability_verified=True,
            )
        )
        counters["inserted"] += 1
    session.flush()
    return counters


def _find_fixture(
    session: Session,
    *,
    league: str,
    season: str,
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime,
) -> Match | None:
    candidates = session.scalars(
        select(Match).where(
            Match.league == league,
            Match.season == season,
            Match.home_team_id == home_team_id,
            Match.away_team_id == away_team_id,
            Match.status == "finished",
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
    ).all()
    nearby = [
        (abs((_utc(candidate.kickoff_utc) - kickoff).total_seconds()), candidate)
        for candidate in candidates
        if abs((_utc(candidate.kickoff_utc) - kickoff).total_seconds())
        <= _MATCH_TIME_TOLERANCE_SECONDS
    ]
    if not nearby:
        return None
    nearby.sort(key=lambda item: item[0])
    if len(nearby) > 1 and nearby[0][0] == nearby[1][0]:
        raise ValueError("Understat match ambiguously maps to multiple known fixtures")
    return nearby[0][1]


def _season_start(season: str) -> int:
    match = re.fullmatch(r"(20\d{2})-(\d{2})", season)
    if match is None or int(match.group(2)) != (int(match.group(1)) + 1) % 100:
        raise ValueError(f"Invalid football season: {season!r}; expected YYYY-YY")
    return int(match.group(1))


def _parse_kickoff(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Understat match is missing datetime")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
            parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid Understat match datetime: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Understat match datetime must include a timezone")
    return parsed.astimezone(UTC)


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Understat match is missing team {key}")
    return item.strip()


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Understat {label} must be numeric") from exc
    if parsed < 0 or not parsed < float("inf"):
        raise ValueError(f"Understat {label} must be finite and non-negative")
    return parsed


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

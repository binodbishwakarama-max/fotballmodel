"""Rate-limited football-data.org client and live fixture/result synchronization."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.ingestion.teams import resolve_team
from forecast.models import Match, MatchSource

logger = logging.getLogger(__name__)
LEAGUE_CODES = {"PL": "PL", "PD": "PD"}
RATE_LIMIT_BACKOFF_SECONDS = 60
CACHE_SECONDS = 300


class FootballDataClient:
    """Small disk-cached API client that respects 429 and transient server failures."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str = "https://api.football-data.org/v4",
        cache_dir: Path = Path("data/cache/football-data-org"),
        client: httpx.Client | None = None,
        sleep=time.sleep,
        max_attempts: int = 4,
    ) -> None:
        if not token.strip():
            raise ValueError("football-data.org API token cannot be empty")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0))
        self._owns_client = client is None
        self.sleep = sleep
        self.max_attempts = max_attempts

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            self.client.close()

    def get_matches(
        self,
        *,
        league: str,
        date_from: date,
        date_to: date,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch competition matches for a date window, using a short disk cache."""
        if league not in LEAGUE_CODES:
            raise ValueError(f"Unsupported league: {league}")
        if date_to < date_from:
            raise ValueError("date_to must be on or after date_from")
        params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}
        cache_key = hashlib.sha256(
            f"{league}|{params['dateFrom']}|{params['dateTo']}".encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        now_utc = (now or datetime.now(UTC)).astimezone(UTC)
        cached = self._read_cache(cache_path, now_utc)
        if cached is not None:
            return cached

        payload = self._get_json(f"/competitions/{LEAGUE_CODES[league]}/matches", params)
        matches = payload.get("matches")
        if not isinstance(matches, list):
            raise ValueError("football-data.org response has no matches list")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"fetched_at": now_utc.isoformat(), "matches": matches},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return matches

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_attempts):
            try:
                response = self.client.get(
                    url,
                    params=params,
                    headers={"X-Auth-Token": self.token},
                )
            except httpx.TransportError:
                if attempt + 1 == self.max_attempts:
                    raise
                self.sleep(min(2**attempt, 30.0))
                continue

            if response.status_code == 429:
                if attempt + 1 == self.max_attempts:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                delay = _retry_delay(retry_after, attempt)
                logger.warning("football-data.org rate limit; retrying in %.1f seconds", delay)
                self.sleep(delay)
                continue
            if response.status_code in (500, 502, 503, 504) and attempt + 1 < self.max_attempts:
                self.sleep(min(2**attempt, 30.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("football-data.org response must be a JSON object")
            return payload
        raise RuntimeError("football-data.org request exhausted retries")

    @staticmethod
    def _read_cache(path: Path, now: datetime) -> list[dict[str, Any]] | None:
        if not path.exists():
            return None
        content = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(content["fetched_at"])
        if now - fetched_at.astimezone(UTC) > timedelta(seconds=CACHE_SECONDS):
            return None
        matches = content["matches"]
        if not isinstance(matches, list):
            raise ValueError(f"Invalid football-data.org cache file: {path}")
        return matches


def sync_api_matches(
    session: Session,
    *,
    league: str,
    matches: list[dict[str, Any]],
    retrieved_at: datetime | None = None,
) -> dict[str, int]:
    """Idempotently insert fixtures and settle finished matches from API payloads."""
    retrieved_at = (retrieved_at or datetime.now(UTC)).astimezone(UTC)
    counters = {"read": 0, "inserted": 0, "updated": 0, "unchanged": 0}
    for payload in matches:
        counters["read"] += 1
        if not isinstance(payload, dict):
            raise ValueError("fixture payload must be an object")
        source_id = payload.get("id")
        source_match_id = str(source_id) if source_id is not None else None
        if source_match_id is None:
            raise ValueError("football-data.org fixture is missing its id")
        kickoff_raw = payload.get("utcDate")
        if not isinstance(kickoff_raw, str):
            raise ValueError(f"fixture {source_match_id} is missing utcDate")
        kickoff = datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00"))
        if kickoff.tzinfo is None:
            raise ValueError(f"fixture {source_match_id} has a timezone-naive utcDate")
        kickoff = kickoff.astimezone(UTC)
        home_name = _team_name(payload, "homeTeam", source_match_id)
        away_name = _team_name(payload, "awayTeam", source_match_id)
        home = resolve_team(
            session,
            source="football-data.org",
            raw_name=home_name,
            league=league,
        )
        away = resolve_team(
            session,
            source="football-data.org",
            raw_name=away_name,
            league=league,
        )
        season_info = payload.get("season")
        season_start = season_info.get("startDate") if isinstance(season_info, dict) else None
        if not isinstance(season_start, str):
            season_start_year = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
        else:
            season_start_year = date.fromisoformat(season_start).year
        season = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
        matchweek_value = payload.get("matchday")
        matchweek = matchweek_value if isinstance(matchweek_value, int) else None
        status = _match_status(payload.get("status"))
        score = payload.get("score")
        full_time = score.get("fullTime", {}) if isinstance(score, dict) else {}
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        finished = status == "finished"
        if finished and (not isinstance(home_goals, int) or not isinstance(away_goals, int)):
            raise ValueError(f"finished fixture {source_match_id} has no full-time score")
        result = (
            ("H" if home_goals > away_goals else "A" if home_goals < away_goals else "D")
            if finished
            else None
        )
        if finished and retrieved_at < kickoff:
            raise ValueError(f"finished fixture {source_match_id} was observed before its kickoff")

        fixture = session.scalar(
            select(Match)
            .join(MatchSource, MatchSource.match_id == Match.id)
            .where(
                Match.league == league,
                MatchSource.source == "football-data.org",
                MatchSource.source_match_id == source_match_id,
            )
        )
        if fixture is None:
            fixture = session.scalar(
                select(Match).where(
                    Match.league == league,
                    Match.kickoff_utc == kickoff,
                    Match.home_team_id == home.id,
                    Match.away_team_id == away.id,
                )
            )
        if fixture is None:
            fixture = Match(
                id=f"fdorg-{league}-{source_match_id}",
                league=league,
                season=season,
                kickoff_utc=kickoff,
                kickoff_time_known=True,
                matchweek=matchweek,
                home_team_id=home.id,
                away_team_id=away.id,
                home_goals=home_goals if finished else None,
                away_goals=away_goals if finished else None,
                result=result,
                result_available_at=retrieved_at if finished else None,
                status=status,
                source="football-data.org",
                source_match_id=source_match_id,
            )
            session.add(fixture)
            counters["inserted"] += 1
        else:
            changed = False
            if fixture.matchweek is None and matchweek is not None:
                fixture.matchweek = matchweek
                changed = True
            if fixture.status != status:
                fixture.status = status
                changed = True
            if finished and (
                fixture.home_goals != home_goals
                or fixture.away_goals != away_goals
                or fixture.result != result
            ):
                fixture.home_goals = home_goals
                fixture.away_goals = away_goals
                fixture.result = result
                fixture.result_available_at = retrieved_at
                changed = True
            if finished and fixture.result_available_at is None:
                fixture.result_available_at = retrieved_at
                changed = True
            if changed:
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
        match_source = session.scalar(
            select(MatchSource).where(
                MatchSource.source == "football-data.org",
                MatchSource.source_match_id == source_match_id,
            )
        )
        if match_source is None:
            session.add(
                MatchSource(
                    match_id=fixture.id,
                    source="football-data.org",
                    source_match_id=source_match_id,
                    first_seen_at=retrieved_at,
                    available_at=retrieved_at,
                )
            )
    return counters


def _team_name(payload: dict[str, Any], key: str, fixture_id: str) -> str:
    team = payload.get(key)
    name = team.get("name") if isinstance(team, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"fixture {fixture_id} has no {key}.name")
    return name.strip()


def _match_status(value: Any) -> str:
    if value in {"FINISHED", "AWARDED"}:
        return "finished"
    if value in {"POSTPONED", "CANCELLED", "SUSPENDED"}:
        return "postponed"
    return "scheduled"


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after is not None:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            try:
                parsed = datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT")
                return max(1.0, (parsed.replace(tzinfo=UTC) - datetime.now(UTC)).total_seconds())
            except ValueError:
                pass
    return min(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1), 300.0)

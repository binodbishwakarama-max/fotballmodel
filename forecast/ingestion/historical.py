"""football-data.co.uk CSV ingestion with explicit quality/provenance reporting."""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from forecast.config import SUPPORTED_LEAGUES
from forecast.ingestion.teams import resolve_team
from forecast.models import IngestionRun, Match, MatchSource, Odds

logger = logging.getLogger(__name__)
SOURCE = "football-data.co.uk"
LEAGUE_CODES = {"PL": "E0", "PD": "SP1"}
LEAGUE_TIMEZONES = {"PL": "Europe/London", "PD": "Europe/Madrid"}
EXPECTED_MATCHES_PER_FULL_SEASON = 380
ODDS_COLUMNS = (
    ("Pinnacle", "close", ("PSCH", "PSCD", "PSCA")),
    ("Bet365", "close", ("B365CH", "B365CD", "B365CA")),
    ("Pinnacle", "open", ("PSH", "PSD", "PSA")),
    ("Bet365", "open", ("B365H", "B365D", "B365A")),
)


@dataclass(slots=True)
class IngestionResult:
    """Quality counters collected while ingesting one or more CSV files."""

    files_read: int = 0
    rows_read: int = 0
    matches_inserted: int = 0
    matches_existing: int = 0
    odds_inserted: int = 0
    rows_rejected: int = 0
    missing_kickoff_time: int = 0
    missing_closing_odds: int = 0
    unmatched_names: list[str] = field(default_factory=list)
    season_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format an actionable text summary without hiding incomplete data."""
        lines = [
            "Historical ingestion quality report",
            f"Files read: {self.files_read}",
            f"Rows read: {self.rows_read}",
            f"Matches inserted: {self.matches_inserted}",
            f"Matches already present: {self.matches_existing}",
            f"Odds observations inserted: {self.odds_inserted}",
            f"Rows rejected: {self.rows_rejected}",
            f"Matches with unknown kickoff time: {self.missing_kickoff_time}",
            f"Matches without verified closing odds: {self.missing_closing_odds}",
        ]
        for season, count in sorted(self.season_counts.items()):
            lines.append(f"Matches {season}: {count}")
        lines.extend(f"WARNING: {warning}" for warning in self.warnings)
        lines.extend(f"UNKNOWN TEAM: {name}" for name in sorted(set(self.unmatched_names)))
        return "\n".join(lines)


def available_seasons(
    *,
    today: date | None = None,
    completed_count: int = 10,
) -> tuple[tuple[str, bool], ...]:
    """Return recent completed seasons and the currently active season."""
    today = today or datetime.now(UTC).date()
    season_start_year = today.year if today.month >= 7 else today.year - 1
    first_completed_start = season_start_year - completed_count
    seasons = [
        (f"{year}-{(year + 1) % 100:02d}", True)
        for year in range(first_completed_start, season_start_year)
    ]
    seasons.append((f"{season_start_year}-{(season_start_year + 1) % 100:02d}", False))
    return tuple(seasons)


def season_start_code(season: str) -> str:
    """Convert a season such as 2025-26 into the source's 2526 path segment."""
    match = re.fullmatch(r"(20\d{2})-(\d{2})", season)
    if match is None or int(match.group(2)) != (int(match.group(1)) + 1) % 100:
        raise ValueError(f"Invalid football season: {season!r}; expected YYYY-YY")
    return match.group(1)[-2:] + match.group(2)


def download_season(
    *,
    league: str,
    season: str,
    raw_dir: Path,
    client: httpx.Client | None = None,
) -> Path:
    """Download and cache one source CSV while recording its retrieval timestamp."""
    if league not in LEAGUE_CODES:
        raise ValueError(f"Unsupported league: {league}")
    url = (
        "https://www.football-data.co.uk/mmz4281/"
        f"{season_start_code(season)}/{LEAGUE_CODES[league]}.csv"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{season_start_code(season)}_{LEAGUE_CODES[league]}.csv"
    own_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True)
    try:
        response = client.get(url)
        response.raise_for_status()
        content = response.content
    finally:
        if own_client:
            client.close()

    if not content.strip():
        raise ValueError(f"Downloaded an empty CSV from {url}")
    target.write_bytes(content)
    manifest_path = raw_dir / "manifest.csv"
    is_new = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("path", "source_url", "retrieved_at", "sha256", "bytes"),
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "path": target.name,
                "source_url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    return target


def ingest_historical(
    *,
    engine: Engine,
    data_dir: Path,
    leagues: tuple[str, ...] = ("PL", "PD"),
    seasons: tuple[str, ...] | None = None,
    download: bool = False,
    client: httpx.Client | None = None,
) -> IngestionResult:
    """Ingest local historical season CSVs, optionally downloading missing files."""
    all_seasons = available_seasons()
    requested_seasons = seasons or tuple(season for season, _ in all_seasons)
    completed = {season: is_completed for season, is_completed in all_seasons}
    result = IngestionResult()

    for league in leagues:
        if league not in SUPPORTED_LEAGUES:
            raise ValueError(f"Unsupported league: {league}")
        for season in requested_seasons:
            season_start_code(season)
            path = data_dir / "raw" / f"{season_start_code(season)}_{LEAGUE_CODES[league]}.csv"
            if not path.exists():
                if not download:
                    result.warnings.append(f"Missing source file: {path}")
                    continue
                try:
                    download_season(
                        league=league,
                        season=season,
                        raw_dir=data_dir / "raw",
                        client=client,
                    )
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    result.warnings.append(f"Download failed for {league} {season}: {exc}")
                    continue
            _ingest_csv(path, league, season, engine, result, completed.get(season, False))
    if result.unmatched_names:
        raise ValueError(result.summary())
    if result.files_read == 0:
        raise FileNotFoundError(
            "No historical CSV files were ingested. Provide season files or use --download."
        )
    return result


def _ingest_csv(
    path: Path,
    league: str,
    season: str,
    engine: Engine,
    result: IngestionResult,
    season_completed: bool,
) -> None:
    content = path.read_bytes()
    decoded = _decode_csv(content, path)
    reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = {field.strip() for field in (reader.fieldnames or [])}
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - fieldnames
    if missing:
        raise ValueError(f"{path} missing required CSV columns: {sorted(missing)}")

    started_at = datetime.now(UTC)
    matches_inserted_before = result.matches_inserted
    rows_rejected_before = result.rows_rejected
    run = IngestionRun(
        source=SOURCE,
        started_at=started_at,
        status="running",
        details={"league": league, "season": season, "source_path": path.name},
    )
    with Session(engine) as session:
        session.add(run)
        session.flush()
        for row_number, row in enumerate(reader, start=2):
            result.rows_read += 1
            run.rows_read += 1
            try:
                home_name = _required_text(row, "HomeTeam")
                away_name = _required_text(row, "AwayTeam")
                kickoff, time_known = _parse_kickoff(row.get("Date"), row.get("Time"), league)
                home_team = resolve_team(
                    session,
                    source=SOURCE,
                    raw_name=home_name,
                    league=league,
                    allow_primary_bootstrap=True,
                )
                away_team = resolve_team(
                    session,
                    source=SOURCE,
                    raw_name=away_name,
                    league=league,
                    allow_primary_bootstrap=True,
                )
                home_goals, away_goals, outcome, status = _parse_result(row)
                if status == "finished" and kickoff >= started_at:
                    raise ValueError("completed result is observed before its fixture kickoff")
                match_id = _match_id(league, season, kickoff, home_team.id, away_team.id)
                existing = session.get(Match, match_id)
                if existing is not None:
                    result.matches_existing += 1
                    match = existing
                else:
                    match = Match(
                        id=match_id,
                        league=league,
                        season=season,
                        kickoff_utc=kickoff,
                        kickoff_time_known=time_known,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        result=outcome,
                        result_available_at=started_at if status == "finished" else None,
                        status=status,
                        source=SOURCE,
                        source_match_id=f"{league}:{season}:{row_number}",
                    )
                    session.add(match)
                    result.matches_inserted += 1
                if (
                    session.scalar(
                        select(MatchSource.id).where(
                            MatchSource.source == SOURCE,
                            MatchSource.source_match_id == f"{league}:{season}:{row_number}",
                        )
                    )
                    is None
                ):
                    session.add(
                        MatchSource(
                            match_id=match_id,
                            source=SOURCE,
                            source_match_id=f"{league}:{season}:{row_number}",
                            first_seen_at=started_at,
                            available_at=started_at,
                        )
                    )
                if not time_known:
                    result.missing_kickoff_time += 1
                if status == "finished":
                    key = f"{league}:{season}"
                    result.season_counts[key] = result.season_counts.get(key, 0) + 1

                odds_rows = _parse_odds(row)
                if not any(kind == "close" for _, kind, _ in odds_rows):
                    result.missing_closing_odds += 1
                for odds_source, kind, prices in odds_rows:
                    query = select(Odds.id).where(
                        Odds.match_id == match_id,
                        Odds.source == odds_source,
                        Odds.kind == kind,
                        Odds.available_at.is_(None),
                    )
                    if session.scalar(query) is not None:
                        continue
                    session.add(
                        Odds(
                            match_id=match_id,
                            source=odds_source,
                            home=prices[0],
                            draw=prices[1],
                            away=prices[2],
                            kind=kind,
                            available_at=None,
                            retrieved_at=datetime.now(UTC),
                            availability_verified=False,
                        )
                    )
                    result.odds_inserted += 1
            except (ValueError, KeyError) as exc:
                result.rows_rejected += 1
                logger.error("Rejecting row %s in %s: %s", row_number, path, exc)
                raise ValueError(f"Invalid row {row_number} in {path}: {exc}") from exc

        season_match_count = result.season_counts.get(f"{league}:{season}", 0)
        if season_completed:
            if season_match_count != EXPECTED_MATCHES_PER_FULL_SEASON:
                result.warnings.append(
                    f"{league} {season}: observed {season_match_count} completed matches; "
                    f"expected {EXPECTED_MATCHES_PER_FULL_SEASON} in a full season"
                )
        run.rows_inserted = result.matches_inserted - matches_inserted_before
        run.rows_rejected = result.rows_rejected - rows_rejected_before
        run.finished_at = datetime.now(UTC)
        run.status = "completed"
        run.details = {
            **run.details,
            "missing_kickoff_time": result.missing_kickoff_time,
            "missing_closing_odds": result.missing_closing_odds,
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "season_match_count": season_match_count,
        }
        session.commit()
    result.files_read += 1


def _decode_csv(content: bytes, path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV {path}")


def _required_text(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"missing {key}")
    return value.strip()


def _parse_kickoff(
    raw_date: str | None,
    raw_time: str | None,
    league: str,
) -> tuple[datetime, bool]:
    if raw_date is None or not raw_date.strip():
        raise ValueError("missing Date")
    try:
        parsed_date = datetime.strptime(raw_date.strip(), "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError(f"invalid Date {raw_date!r}; expected DD/MM/YYYY") from exc

    time_known = raw_time is not None and bool(raw_time.strip())
    if time_known:
        try:
            parsed_time = datetime.strptime(raw_time.strip(), "%H:%M").time()
        except ValueError as exc:
            raise ValueError(f"invalid Time {raw_time!r}; expected HH:MM") from exc
    else:
        parsed_time = time(12, 0)

    try:
        naive_local = datetime.combine(parsed_date, parsed_time)
        zone = ZoneInfo(LEAGUE_TIMEZONES[league])
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Timezone data unavailable for {league}") from exc
    local_time = naive_local.replace(tzinfo=zone, fold=0)
    other_fold = naive_local.replace(tzinfo=zone, fold=1)
    if local_time.utcoffset() != other_fold.utcoffset():
        raise ValueError(f"ambiguous local kickoff time {naive_local} in {zone.key}")
    if local_time.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive_local:
        raise ValueError(f"nonexistent local kickoff time {naive_local} in {zone.key}")
    return local_time.astimezone(UTC), time_known


def _parse_result(row: dict[str, str | None]) -> tuple[int | None, int | None, str | None, str]:
    home_raw = (row.get("FTHG") or "").strip()
    away_raw = (row.get("FTAG") or "").strip()
    result_raw = (row.get("FTR") or "").strip().upper()
    if not home_raw and not away_raw and not result_raw:
        return None, None, None, "scheduled"
    if not home_raw or not away_raw:
        raise ValueError("partial final score")
    try:
        home_goals, away_goals = int(float(home_raw)), int(float(away_raw))
    except ValueError as exc:
        raise ValueError("invalid final score") from exc
    expected_result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
    if result_raw and result_raw != expected_result:
        raise ValueError(f"FTR {result_raw!r} conflicts with score")
    return home_goals, away_goals, expected_result, "finished"


def _parse_odds(row: dict[str, str | None]) -> list[tuple[str, str, tuple[float, float, float]]]:
    parsed: list[tuple[str, str, tuple[float, float, float]]] = []
    for source, kind, columns in ODDS_COLUMNS:
        values = tuple(_parse_price(row.get(column)) for column in columns)
        if all(value is not None for value in values):
            parsed.append((source, kind, (values[0], values[1], values[2])))
    return parsed


def _parse_price(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        price = float(value.strip())
    except ValueError:
        return None
    return price if price > 1 else None


def _match_id(league: str, season: str, kickoff: datetime, home_id: int, away_id: int) -> str:
    key = f"{league}|{season}|{kickoff.isoformat()}|{home_id}|{away_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

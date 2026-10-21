"""Per-fixture feature construction with strict source-availability filters."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from forecast.models import ExpectedGoals, Match, Team

FORM_WINDOWS = (3, 5, 10)
XG_WINDOWS = (5, 10)
LEAGUE_PRIORS = {"points": 1.35, "goals": 1.35, "goals_against": 1.35}
ELO_MEAN = 1500.0


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Feature vector, stable hash, and count of cold-start/unavailable values."""

    values: dict[str, float | int | str | None]
    features_hash: str
    fallback_count: int


@dataclass(frozen=True, slots=True)
class _History:
    match: Match
    home_team: Team
    away_team: Team


def _utc(value: datetime) -> datetime:
    """Normalize a database timestamp to UTC even with SQLite's naive results."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _available_history(
    session: Session,
    *,
    league: str,
    cutoff: datetime,
    exclude_match_id: str | None,
) -> list[_History]:
    home_alias = aliased(Team)
    away_alias = aliased(Team)
    statement = (
        select(Match, home_alias, away_alias)
        .join(home_alias, Match.home_team_id == home_alias.id)
        .join(away_alias, Match.away_team_id == away_alias.id)
        .where(
            Match.league == league,
            Match.status == "finished",
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
    )
    if exclude_match_id is not None:
        statement = statement.where(Match.id != exclude_match_id)
    histories: list[_History] = []
    for match, home, away in session.execute(statement):
        if match.result_available_at is None:
            continue
        if _utc(match.kickoff_utc) >= cutoff:
            continue
        if _utc(match.result_available_at) >= cutoff:
            continue
        histories.append(_History(match, home, away))
    histories.sort(key=lambda item: (_utc(item.match.kickoff_utc), item.match.id))
    return histories


def build_features(
    session: Session,
    *,
    match_id: str,
    as_of_kickoff: datetime,
) -> FeatureSnapshot:
    """Build one fixture's numeric features using only verified pre-cutoff data."""
    cutoff = _utc(as_of_kickoff)
    fixture = session.get(Match, match_id)
    if fixture is None:
        raise ValueError(f"Unknown fixture: {match_id}")
    if not fixture.kickoff_time_known:
        raise ValueError(f"Fixture {match_id} has no verified kickoff time")
    if cutoff >= _utc(fixture.kickoff_utc):
        raise ValueError("forecast cutoff must be strictly before kickoff")

    home_team = session.get(Team, fixture.home_team_id)
    away_team = session.get(Team, fixture.away_team_id)
    if home_team is None or away_team is None:
        raise ValueError(f"Fixture {match_id} references an unknown team")
    history = _available_history(
        session,
        league=fixture.league,
        cutoff=cutoff,
        exclude_match_id=match_id,
    )
    home_matches = _team_matches(history, home_team.id)
    away_matches = _team_matches(history, away_team.id)
    league_matches = [item.match for item in history]
    current_season_matches = [match for match in league_matches if match.season == fixture.season]
    goals_prior = _mean(
        [float(match.home_goals) for match in league_matches if match.home_goals is not None],
        LEAGUE_PRIORS["goals"],
    )
    away_goals_prior = _mean(
        [float(match.away_goals) for match in league_matches if match.away_goals is not None],
        LEAGUE_PRIORS["goals"],
    )
    points_prior = _mean(_points_for_matches(league_matches), LEAGUE_PRIORS["points"])
    features: dict[str, float | int | str | None] = {
        "league": fixture.league,
        "season": fixture.season,
        "league_pl": int(fixture.league == "PL"),
        "league_pd": int(fixture.league == "PD"),
        "season_progress": min(1.0, len(current_season_matches) / 380.0),
        "home_elo": ELO_MEAN,
        "away_elo": ELO_MEAN,
        "elo_difference": 0.0,
    }
    fallbacks = 0
    elo_ratings = _elo_ratings(history)
    home_elo = elo_ratings.get(home_team.id, ELO_MEAN)
    away_elo = elo_ratings.get(away_team.id, ELO_MEAN)
    features.update(
        {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo + 65.0,
        }
    )

    for side, matches, venue in (
        ("home", home_matches, "home"),
        ("away", away_matches, "away"),
    ):
        for window in FORM_WINDOWS:
            recent = matches[-window:]
            if not recent:
                fallbacks += 1
            _add_form_features(
                features,
                side=side,
                suffix=f"last_{window}",
                matches=recent,
                team_id=home_team.id if side == "home" else away_team.id,
                prior_points=points_prior,
                prior_goals=goals_prior if side == "home" else away_goals_prior,
                prior_goals_against=away_goals_prior if side == "home" else goals_prior,
            )
        venue_matches = [
            item
            for item in matches
            if (
                item.match.home_team_id == home_team.id
                if side == "home"
                else item.match.away_team_id == away_team.id
            )
        ]
        _add_form_features(
            features,
            side=side,
            suffix=f"{venue}_venue",
            matches=venue_matches[-5:],
            team_id=home_team.id if side == "home" else away_team.id,
            prior_points=points_prior,
            prior_goals=goals_prior if side == "home" else away_goals_prior,
            prior_goals_against=away_goals_prior if side == "home" else goals_prior,
        )

    xg_records = _available_xg(session, history, cutoff)
    for side, team_id in (("home", home_team.id), ("away", away_team.id)):
        team_xg = [record for record in xg_records if record[0] == team_id]
        for window in XG_WINDOWS:
            recent = team_xg[-window:]
            features[f"{side}_xg_for_last_{window}"] = (
                _mean([item[1] for item in recent], float("nan")) if recent else None
            )
            features[f"{side}_xg_against_last_{window}"] = (
                _mean([item[2] for item in recent], float("nan")) if recent else None
            )
            features[f"{side}_xg_overperformance_last_{window}"] = (
                _mean([item[3] for item in recent], float("nan")) if recent else None
            )
            if not recent:
                fallbacks += 1

    _add_rest_features(features, "home", home_matches, home_team.id, cutoff)
    _add_rest_features(features, "away", away_matches, away_team.id, cutoff)
    h2h = [
        item
        for item in history
        if {item.match.home_team_id, item.match.away_team_id} == {home_team.id, away_team.id}
    ][-5:]
    home_h2h_points: list[float] = []
    away_h2h_points: list[float] = []
    for item in h2h:
        home_side = item.match.home_team_id == home_team.id
        goals_for = item.match.home_goals if home_side else item.match.away_goals
        goals_against = item.match.away_goals if home_side else item.match.home_goals
        points = _match_points(goals_for, goals_against)
        home_h2h_points.append(points)
        away_h2h_points.append(_match_points(goals_against, goals_for))
    features["home_h2h_points_last_5"] = _mean(home_h2h_points, points_prior)
    features["away_h2h_points_last_5"] = _mean(away_h2h_points, points_prior)
    features["h2h_matches_last_5"] = len(h2h)
    if not h2h:
        fallbacks += 2

    for key, value in tuple(features.items()):
        if isinstance(value, float) and math.isnan(value):
            features[key] = None
    canonical = json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return FeatureSnapshot(
        values=features,
        features_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        fallback_count=fallbacks,
    )


def _team_matches(history: list[_History], team_id: int) -> list[_History]:
    return [
        item
        for item in history
        if item.match.home_team_id == team_id or item.match.away_team_id == team_id
    ]


def _points_for_matches(matches: list[Match]) -> list[float]:
    return [
        _match_points(match.home_goals, match.away_goals)
        + _match_points(match.away_goals, match.home_goals)
        for match in matches
        if match.home_goals is not None and match.away_goals is not None
    ]


def _match_points(goals_for: int | None, goals_against: int | None) -> float:
    if goals_for is None or goals_against is None:
        return 0.0
    return 3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0


def _mean(values: list[float], fallback: float) -> float:
    return sum(values) / len(values) if values else fallback


def _shrink(value: float, count: int, prior: float, prior_weight: float = 5.0) -> float:
    return (value * count + prior * prior_weight) / (count + prior_weight)


def _add_form_features(
    target: dict[str, float | int | str | None],
    *,
    side: str,
    suffix: str,
    matches: list[_History],
    team_id: int,
    prior_points: float,
    prior_goals: float,
    prior_goals_against: float,
) -> None:
    points: list[float] = []
    goals_for: list[float] = []
    goals_against: list[float] = []
    for item in matches:
        is_home = item.match.home_team_id == team_id
        scored = item.match.home_goals if is_home else item.match.away_goals
        conceded = item.match.away_goals if is_home else item.match.home_goals
        if scored is None or conceded is None:
            continue
        goals_for.append(float(scored))
        goals_against.append(float(conceded))
        points.append(_match_points(scored, conceded))

    count = len(points)
    raw_points = _mean(points, prior_points)
    raw_goals = _mean(goals_for, prior_goals)
    raw_against = _mean(goals_against, prior_goals_against)
    target[f"{side}_points_per_game_{suffix}"] = _shrink(raw_points, count, prior_points)
    target[f"{side}_goals_for_{suffix}"] = _shrink(raw_goals, count, prior_goals)
    target[f"{side}_goals_against_{suffix}"] = _shrink(raw_against, count, prior_goals_against)
    target[f"{side}_goal_difference_{suffix}"] = (
        target[f"{side}_goals_for_{suffix}"] - target[f"{side}_goals_against_{suffix}"]
    )
    target[f"{side}_matches_{suffix}"] = count


def _add_rest_features(
    target: dict[str, float | int | str | None],
    side: str,
    matches: list[_History],
    team_id: int,
    cutoff: datetime,
) -> None:
    last_played = _utc(matches[-1].match.kickoff_utc) if matches else None
    target[f"{side}_rest_days"] = (
        min(45.0, max(0.0, (cutoff - last_played).total_seconds() / 86400.0))
        if last_played
        else 7.0
    )
    recent_cutoff = cutoff - timedelta(days=14)
    target[f"{side}_matches_last_14_days"] = sum(
        _utc(item.match.kickoff_utc) >= recent_cutoff for item in matches
    )


def _available_xg(
    session: Session,
    history: list[_History],
    cutoff: datetime,
) -> list[tuple[int, float, float, float]]:
    known_matches = {item.match.id: item.match for item in history}
    if not known_matches:
        return []
    statement = select(ExpectedGoals).where(
        ExpectedGoals.match_id.in_(known_matches),
        ExpectedGoals.availability_verified.is_(True),
        ExpectedGoals.available_at.is_not(None),
    )
    by_team: dict[int, list[tuple[datetime, float, float, float]]] = defaultdict(list)
    for record in session.scalars(statement):
        if record.available_at is None or _utc(record.available_at) >= cutoff:
            continue
        match = known_matches[record.match_id]
        if match.home_goals is None or match.away_goals is None:
            continue
        by_team[match.home_team_id].append(
            (
                _utc(match.kickoff_utc),
                record.home_xg,
                record.away_xg,
                record.home_xg - match.home_goals,
            )
        )
        by_team[match.away_team_id].append(
            (
                _utc(match.kickoff_utc),
                record.away_xg,
                record.home_xg,
                record.away_xg - match.away_goals,
            )
        )
    result = []
    for team_id, values in by_team.items():
        result.extend(
            (team_id, xg_for, xg_against, overperformance)
            for _, xg_for, xg_against, overperformance in sorted(values)
        )
    result.sort(key=lambda value: value[0])
    return result


def _elo_ratings(history: list[_History]) -> dict[int, float]:
    ratings: dict[int, float] = {}
    current_season: str | None = None
    for item in history:
        match = item.match
        if match.home_goals is None or match.away_goals is None:
            continue
        if current_season is not None and match.season != current_season:
            ratings = {
                team_id: ELO_MEAN + 0.75 * (rating - ELO_MEAN)
                for team_id, rating in ratings.items()
            }
        current_season = match.season
        home = ratings.get(match.home_team_id, ELO_MEAN)
        away = ratings.get(match.away_team_id, ELO_MEAN)
        expected_home = 1 / (1 + 10 ** (-(home + 65.0 - away) / 400))
        actual_home = (
            1.0
            if match.home_goals > match.away_goals
            else (0.5 if match.home_goals == match.away_goals else 0.0)
        )
        goal_difference = abs(match.home_goals - match.away_goals)
        multiplier = math.log1p(goal_difference) * (2.2 / (abs(home - away) * 0.001 + 2.2))
        delta = 20.0 * multiplier * (actual_home - expected_home)
        ratings[match.home_team_id] = home + delta
        ratings[match.away_team_id] = away - delta
    return ratings

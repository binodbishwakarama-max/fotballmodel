"""Per-fixture public scorecard metrics and fair market coverage."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.eval.walk_forward import paired_matchweek_bootstrap
from forecast.ledger.ledger import _utc
from forecast.models import Match, Odds, Prediction, Score

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
MARKET_SOURCE_PREFERENCE = ("Pinnacle", "Bet365")


def evaluate_forecasts(
    session: Session,
    *,
    computed_at: datetime | None = None,
    bootstrap_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score settled canonical predictions, Elo, and valid closing markets."""
    timestamp = _utc(computed_at or datetime.now(UTC))
    rows = session.execute(
        select(Prediction, Match)
        .join(Match, Match.id == Prediction.match_id)
        .where(Match.status == "finished", Match.result.in_(tuple(OUTCOME_INDEX)))
        .order_by(Match.league, Match.season, Match.kickoff_utc)
    ).all()
    canonical: dict[tuple[str, str, str], tuple[Prediction, Match]] = {}
    for prediction, match in rows:
        key = (match.id, prediction.model_version, prediction.forecast_policy_version)
        existing = canonical.get(key)
        if existing is None or _utc(prediction.created_at) < _utc(existing[0].created_at):
            canonical[key] = (prediction, match)

    scores: list[dict[str, Any]] = []
    model_rows: dict[tuple[str, str, str], list[tuple[Match, list[float], int]]] = {}
    for prediction, match in canonical.values():
        label = OUTCOME_INDEX[match.result]
        probabilities = [prediction.p_home, prediction.p_draw, prediction.p_away]
        values = _per_match_metrics(label, probabilities)
        version = f"{prediction.model_version}:{prediction.forecast_policy_version}"
        model_rows.setdefault((version, match.league, match.season), []).append(
            (match, probabilities, label)
        )
        for metric, value in values.items():
            scores.append(
                {
                    "match_id": match.id,
                    "forecaster": "elo" if prediction.model_version.startswith("elo-") else "model",
                    "version": version,
                    "metric": metric,
                    "value": value,
                    "metadata": {
                        "league": match.league,
                        "season": match.season,
                        "model_version": prediction.model_version,
                        "forecast_policy_version": prediction.forecast_policy_version,
                    },
                }
            )

    market_coverage: dict[tuple[str, str], int] = {}
    market_rows: dict[tuple[str, str], list[tuple[Match, list[float], int]]] = {}
    market_missing_match_ids: set[str] = set()
    market_scored_match_ids: dict[tuple[str, str], set[str]] = {}
    for model_key, forecasts in model_rows.items():
        _, league, season = model_key
        for match, _, label in forecasts:
            key = (league, season)
            already_selected = {item[0].id for item in market_rows.get(key, [])}
            if match.id in already_selected:
                continue
            odds_rows = session.scalars(
                select(Odds)
                .where(Odds.match_id == match.id, Odds.kind == "close")
                .order_by(Odds.retrieved_at.desc())
            ).all()
            selected_odds = _preferred_closing_odds(odds_rows)
            if selected_odds is None:
                market_missing_match_ids.add(match.id)
                continue
            from forecast.eval.devig import devig

            market_probs = devig(
                [selected_odds.home, selected_odds.draw, selected_odds.away],
                method="power",
            ).tolist()
            market_rows.setdefault(key, []).append((match, market_probs, label))
            market_scored_match_ids.setdefault(key, set()).add(match.id)
            for metric, value in _per_match_metrics(label, market_probs).items():
                scores.append(
                    {
                        "match_id": match.id,
                        "forecaster": "market",
                        "version": f"closing-power-{selected_odds.source}",
                        "metric": metric,
                        "value": value,
                        "metadata": {
                            "league": league,
                            "season": season,
                            "source": selected_odds.source,
                            "kind": "close",
                            "availability": (
                                "verified"
                                if selected_odds.availability_verified
                                else "retrospective_unverified"
                            ),
                        },
                    }
                )

    paired_intervals: dict[str, dict[str, float]] = {}
    for model_key, forecasts in model_rows.items():
        version, league, season = model_key
        model_by_match = {match.id: (match, probs, label) for match, probs, label in forecasts}
        market_by_match = {
            match.id: (match, probs, label)
            for match, probs, label in market_rows.get((league, season), [])
        }
        common_ids = sorted(set(model_by_match) & set(market_by_match))
        if not common_ids:
            continue
        model_log_losses: list[float] = []
        market_log_losses: list[float] = []
        match_leagues: list[str] = []
        match_seasons: list[str] = []
        matchweeks: list[int] = []
        for match_id in common_ids:
            match, model_probs, label = model_by_match[match_id]
            _, market_probs, _ = market_by_match[match_id]
            model_log_losses.append(_per_match_metrics(label, model_probs)["log_loss"])
            market_log_losses.append(_per_match_metrics(label, market_probs)["log_loss"])
            match_leagues.append(league)
            match_seasons.append(season)
            matchweeks.append(_matchweek_cluster(match))
        difference, lower, upper = paired_matchweek_bootstrap(
            model_log_losses,
            market_log_losses,
            leagues=match_leagues,
            seasons=match_seasons,
            matchweeks=matchweeks,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        paired_intervals[f"{version}:{league}:{season}"] = {
            "matches": len(common_ids),
            "model_minus_market_log_loss": difference,
            "ci_95_low": lower,
            "ci_95_high": upper,
            "bootstrap_seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
            "cluster": "official matchday when available; otherwise UTC calendar week",
        }

    for key, interval in paired_intervals.items():
        version, league, season = key.rsplit(":", maxsplit=2)
        for score in scores:
            if (
                score["forecaster"] == "model"
                and score["version"] == version
                and score["metric"] == "log_loss"
                and score["metadata"]["league"] == league
                and score["metadata"]["season"] == season
            ):
                score["metadata"]["confidence_interval"] = {
                    "comparison": "market",
                    "lower": interval["ci_95_low"],
                    "upper": interval["ci_95_high"],
                    "difference": interval["model_minus_market_log_loss"],
                }

    _persist_scores(session, scores, timestamp)
    market_coverage = {key: len(match_ids) for key, match_ids in market_scored_match_ids.items()}
    total_model_matches = len({match.id for _, match in canonical.values()})
    total_market_matches = sum(market_coverage.values())
    return {
        "computed_at": timestamp.isoformat(),
        "model_matches_scored": total_model_matches,
        "market_matches_scored": total_market_matches,
        "market_missing_matches": len(market_missing_match_ids),
        "market_coverage": {
            f"{league}:{season}": count
            for (league, season), count in sorted(market_coverage.items())
        },
        "paired_model_market_intervals": paired_intervals,
        "scores_written": len(scores),
    }


def _per_match_metrics(label: int, probabilities: list[float]) -> dict[str, float]:
    selected = probabilities[label]
    log_loss = -math.log(max(1e-15, selected))
    actual = np.eye(3, dtype=np.float64)[label]
    probs = np.asarray(probabilities, dtype=np.float64)
    brier = float(np.square(probs - actual).sum())
    forecast_cumulative = np.cumsum(probs)[:2]
    actual_cumulative = np.cumsum(actual)[:2]
    rps = float(np.square(forecast_cumulative - actual_cumulative).mean())
    return {"log_loss": log_loss, "brier": brier, "rps": rps}


def _preferred_closing_odds(odds_rows: list[Odds]) -> Odds | None:
    valid = [
        odds
        for odds in odds_rows
        if all(math.isfinite(value) and value > 1 for value in (odds.home, odds.draw, odds.away))
    ]
    for source in MARKET_SOURCE_PREFERENCE:
        for odds in valid:
            if odds.source == source:
                return odds
    return valid[0] if valid else None


def _matchweek_cluster(match: Match) -> int:
    if match.matchweek is not None:
        return match.matchweek
    kickoff = _utc(match.kickoff_utc)
    return kickoff.isocalendar().week


def _persist_scores(
    session: Session,
    scores: list[dict[str, Any]],
    computed_at: datetime,
) -> None:
    for item in scores:
        existing = session.scalar(
            select(Score).where(
                Score.match_id == item["match_id"],
                Score.forecaster == item["forecaster"],
                Score.version == item["version"],
                Score.metric == item["metric"],
            )
        )
        if existing is None:
            session.add(
                Score(
                    match_id=item["match_id"],
                    forecaster=item["forecaster"],
                    version=item["version"],
                    metric=item["metric"],
                    value=item["value"],
                    computed_at=computed_at,
                    metadata_json=item["metadata"],
                )
            )
        else:
            existing.value = item["value"]
            existing.computed_at = computed_at
            existing.metadata_json = item["metadata"]

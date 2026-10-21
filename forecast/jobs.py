"""Live ingestion, publication, settlement, and scorecard jobs."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from forecast.config import SUPPORTED_LEAGUES, Settings
from forecast.db import Engine
from forecast.eval.scorecard import evaluate_forecasts
from forecast.features.build import build_features
from forecast.ingestion.football_data_org import FootballDataClient, sync_api_matches
from forecast.ledger.ledger import FORECAST_POLICY_VERSION, create_batch
from forecast.modeling.elo import EloBaseline
from forecast.modeling.registry import load_production_model
from forecast.models import Match, Prediction, Team


def sync_live_fixtures(
    engine: Engine,
    *,
    settings: Settings,
    date_from: date,
    date_to: date,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch and persist scheduled fixtures or results for both supported leagues."""
    if settings.football_data_org_token is None:
        raise RuntimeError("FOOTBALL_DATA_ORG_TOKEN is required for live fixture ingestion")
    fetched_at = _aware_utc(now or datetime.now(UTC))
    client = FootballDataClient(
        token=settings.football_data_org_token,
        base_url=settings.football_data_org_base_url,
        cache_dir=settings.data_dir / "cache" / "football-data-org",
    )
    results: dict[str, Any] = {}
    try:
        with Session(engine) as session:
            for league in SUPPORTED_LEAGUES:
                payload = client.get_matches(
                    league=league,
                    date_from=date_from,
                    date_to=date_to,
                    now=fetched_at,
                )
                results[league] = sync_api_matches(
                    session,
                    league=league,
                    matches=payload,
                    retrieved_at=fetched_at,
                )
            session.commit()
    finally:
        client.close()
    return results


def settle_results(
    engine: Engine,
    *,
    settings: Settings,
    now: datetime | None = None,
    days_back: int = 14,
) -> dict[str, Any]:
    """Refresh recent API results without modifying any stored prediction."""
    if days_back < 1:
        raise ValueError("days_back must be positive")
    current = _aware_utc(now or datetime.now(UTC))
    return sync_live_fixtures(
        engine,
        settings=settings,
        date_from=(current - timedelta(days=days_back)).date(),
        date_to=current.date(),
        now=current,
    )


def predict_upcoming(
    engine: Engine,
    *,
    settings: Settings,
    model_dir: Path | None = None,
    ledger_dir: Path | None = None,
    now: datetime | None = None,
    window_hours: int = 72,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish one model and Elo forecast per eligible fixture in the policy window."""
    if window_hours < 24:
        raise ValueError("window_hours must be at least 24 for the canonical forecast policy")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    models_path = model_dir or settings.models_dir
    output_ledger = ledger_dir or settings.ledger_dir
    latest_allowed_kickoff = min(window_hours, 30)
    fixtures = []
    with Session(engine) as session:
        home_alias = aliased(Team)
        away_alias = aliased(Team)
        statement = (
            select(Match, home_alias, away_alias)
            .join(home_alias, Match.home_team_id == home_alias.id)
            .join(away_alias, Match.away_team_id == away_alias.id)
            .where(
                Match.status == "scheduled",
                Match.kickoff_time_known.is_(True),
                Match.kickoff_utc > current + timedelta(hours=24),
                Match.kickoff_utc <= current + timedelta(hours=latest_allowed_kickoff),
            )
            .order_by(Match.kickoff_utc)
        )
        fixtures = session.execute(statement).all()
        if not fixtures:
            return {"fixtures_seen": 0, "predictions_created": 0, "batch_id": None, "items": []}

        elo = _fit_live_elo(session, current)
        pending: list[dict[str, Any]] = []
        preview: list[dict[str, Any]] = []
        for match, home, away in fixtures:
            bundle = load_production_model(model_dir=models_path, league=match.league)
            model_version = str(bundle["model_version"])
            calibrator = bundle.get("calibrator")
            if calibrator is None:
                raise ValueError(f"production model {model_version} has no calibrator")
            existing_model = session.scalar(
                select(Prediction.id).where(
                    Prediction.match_id == match.id,
                    Prediction.model_version == model_version,
                    Prediction.forecast_policy_version == FORECAST_POLICY_VERSION,
                )
            )
            if existing_model is not None:
                continue

            feature_snapshot = build_features(
                session,
                match_id=match.id,
                as_of_kickoff=current,
            )
            feature_names = bundle.get("features")
            model = bundle.get("model")
            if model is None or not isinstance(feature_names, list) or not feature_names:
                raise ValueError(
                    f"production model {model_version} is missing features or estimator"
                )
            row = pd.DataFrame(
                [[feature_snapshot.values.get(name) for name in feature_names]],
                columns=feature_names,
            )
            raw_probabilities = _aligned_probabilities(model, row)[0]
            probabilities = calibrator.transform(raw_probabilities.reshape(1, -1))[0]
            model_prediction = {
                "match_id": match.id,
                "kickoff_utc": _database_utc(match.kickoff_utc),
                "model_version": model_version,
                "forecast_policy_version": FORECAST_POLICY_VERSION,
                "p_home": float(probabilities[0]),
                "p_draw": float(probabilities[1]),
                "p_away": float(probabilities[2]),
                "features_hash": feature_snapshot.features_hash,
            }
            elo_existing = session.scalar(
                select(Prediction.id).where(
                    Prediction.match_id == match.id,
                    Prediction.model_version == "elo-baseline-v1",
                    Prediction.forecast_policy_version == FORECAST_POLICY_VERSION,
                )
            )
            rating_hash = _elo_features_hash(elo, home.id, away.id)
            elo_prediction = {
                "match_id": match.id,
                "kickoff_utc": _database_utc(match.kickoff_utc),
                "model_version": "elo-baseline-v1",
                "forecast_policy_version": FORECAST_POLICY_VERSION,
                "p_home": 0.0,
                "p_draw": 0.0,
                "p_away": 0.0,
                "features_hash": rating_hash,
            }
            if elo_existing is None:
                elo_probs = elo.predict_proba(
                    str(home.id),
                    str(away.id),
                    season=match.season,
                )
                elo_prediction.update(
                    {
                        "p_home": elo_probs[0],
                        "p_draw": elo_probs[1],
                        "p_away": elo_probs[2],
                    }
                )
                pending.append(elo_prediction)
            pending.append(model_prediction)
            preview.append(
                {
                    "match_id": match.id,
                    "league": match.league,
                    "kickoff_utc": _database_utc(match.kickoff_utc).isoformat(),
                    "home_team": home.canonical_name,
                    "away_team": away.canonical_name,
                    "model_version": model_version,
                    "p_home": model_prediction["p_home"],
                    "p_draw": model_prediction["p_draw"],
                    "p_away": model_prediction["p_away"],
                    "elo": (
                        {
                            "p_home": elo_prediction["p_home"],
                            "p_draw": elo_prediction["p_draw"],
                            "p_away": elo_prediction["p_away"],
                        }
                        if elo_existing is None
                        else None
                    ),
                    "feature_fallback_count": feature_snapshot.fallback_count,
                }
            )
        if dry_run or not pending:
            return {
                "fixtures_seen": len(fixtures),
                "predictions_created": len(pending),
                "batch_id": None,
                "items": preview,
                "dry_run": dry_run,
            }
        batch, path = create_batch(
            session,
            predictions=pending,
            ledger_dir=output_ledger,
            created_at=current,
        )
        return {
            "fixtures_seen": len(fixtures),
            "predictions_created": len(pending),
            "batch_id": batch.batch_id,
            "ledger_file": str(path),
            "items": preview,
            "dry_run": False,
        }


def refresh_scorecard(
    engine: Engine,
    *,
    now: datetime | None = None,
    bootstrap_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Recompute and persist scorecard metrics for newly settled fixtures."""
    with Session(engine) as session:
        report = evaluate_forecasts(
            session,
            computed_at=now,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        session.commit()
        return report


def _fit_live_elo(session: Session, current: datetime) -> EloBaseline:
    matches = session.scalars(select(Match).where(Match.status == "finished")).all()
    records = [
        {
            "id": match.id,
            "season": match.season,
            "kickoff_utc": match.kickoff_utc,
            "result_available_at": match.result_available_at,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
        }
        for match in matches
    ]
    return EloBaseline().fit(records, through=current)


def _elo_features_hash(model: EloBaseline, home_id: int, away_id: int) -> str:
    import hashlib
    import json

    payload = {
        "home_rating": model.ratings.get(str(home_id), 1500.0),
        "away_rating": model.ratings.get(str(away_id), 1500.0),
        "home_advantage": model.home_advantage,
        "season_regression": model.season_regression,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("job timestamps must include a timezone")
    return value.astimezone(UTC)


def _database_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _aligned_probabilities(model: Any, row: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(row), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    aligned = np.zeros((len(row), 3), dtype=np.float64)
    aligned[:, classes] = probabilities
    totals = aligned.sum(axis=1, keepdims=True)
    if np.any(totals <= 0) or not np.all(np.isfinite(aligned)):
        raise ValueError("production model returned invalid class probabilities")
    return aligned / totals

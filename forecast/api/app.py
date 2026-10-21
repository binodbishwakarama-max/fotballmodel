"""Read-only FastAPI endpoints for published forecasts and scorecards."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased

from forecast.config import Settings
from forecast.db import get_engine
from forecast.ledger.ledger import verify_file
from forecast.models import LedgerBatch, Match, Prediction, Score, Team


class ApiModel(BaseModel):
    """Pydantic response model allowing ORM-derived values."""

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(ApiModel):
    status: str
    database: str


class UpcomingItem(ApiModel):
    match: dict[str, Any]
    prediction: dict[str, Any]


class UpcomingResponse(ApiModel):
    items: list[UpcomingItem]


class RecentResponse(ApiModel):
    items: list[dict[str, Any]]


class ScoresResponse(ApiModel):
    scores: list[dict[str, Any]]
    matches_scored: int
    last_updated: datetime | None


class CalibrationResponse(ApiModel):
    bins: list[dict[str, Any]]
    outcomes: list[str]
    matches_scored: int


def create_app(
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create an API app, with injectable database dependencies for testing."""
    settings = settings or Settings.from_env()
    database_engine = engine or get_engine(
        settings.database_url or "sqlite:///data/bwb_forecast.db"
    )
    app = FastAPI(
        title="BWB Forecast API",
        version="0.1.0",
        description="Read-only forecasts, public ledger verification, and measured scores.",
    )
    app.state.engine = database_engine
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.web_origin],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def get_session(request: Request):
        with Session(request.app.state.engine) as session:
            yield session

    session_dependency = Depends(get_session)

    def add_cache(response: Response) -> None:
        response.headers["Cache-Control"] = "public, max-age=30, stale-while-revalidate=60"

    @app.get("/health", response_model=HealthResponse)
    def health(session: Session = session_dependency) -> HealthResponse:
        try:
            session.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return HealthResponse(status="ok", database="ok")

    @app.get("/predictions/upcoming", response_model=UpcomingResponse)
    def upcoming(
        response: Response,
        session: Session = session_dependency,
        league: str | None = Query(default=None, pattern="^(PL|PD)$"),
    ) -> UpcomingResponse:
        now = datetime.now(UTC)
        home_alias = aliased(Team)
        away_alias = aliased(Team)
        statement = (
            select(Match, Prediction, home_alias, away_alias)
            .join(Prediction, Prediction.match_id == Match.id)
            .join(home_alias, Match.home_team_id == home_alias.id)
            .join(away_alias, Match.away_team_id == away_alias.id)
            .where(
                Match.kickoff_utc > now,
                Match.status == "scheduled",
                Match.kickoff_time_known.is_(True),
            )
            .order_by(Match.kickoff_utc, Prediction.created_at.desc())
        )
        if league is not None:
            statement = statement.where(Match.league == league)
        rows = session.execute(statement).all()
        seen: set[tuple[str, str]] = set()
        items = []
        for match, prediction, home, away in rows:
            key = (match.id, prediction.forecast_policy_version)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                UpcomingItem(
                    match=_match_json(match, home, away),
                    prediction=_prediction_json(prediction),
                )
            )
        add_cache(response)
        return UpcomingResponse(items=items)

    @app.get("/predictions/{match_id}")
    def prediction_detail(
        match_id: str,
        response: Response,
        request: Request,
        session: Session = session_dependency,
    ) -> dict[str, Any]:
        match = session.get(Match, match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="match not found")
        prediction = session.scalar(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
        )
        if prediction is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        batch = session.get(LedgerBatch, prediction.batch_id)
        if batch is None:
            raise HTTPException(status_code=500, detail="prediction ledger batch is missing")
        settings = request.app.state.settings
        batch_time = _utc(batch.created_at)
        filename = f"{batch_time.date().isoformat()}_{batch.batch_id}.json"
        public_path = "/".join(
            part for part in (settings.ledger_public_path.strip("/"), filename) if part
        )
        public_url = (
            f"{settings.ledger_public_url.rstrip('/')}/blob/"
            f"{settings.ledger_public_branch}/{public_path}"
            if settings.ledger_public_url
            else None
        )
        add_cache(response)
        return {
            "match_id": match.id,
            "league": match.league,
            "season": match.season,
            "kickoff_utc": _utc(match.kickoff_utc),
            "model_version": prediction.model_version,
            "forecast_policy_version": prediction.forecast_policy_version,
            "p_home": prediction.p_home,
            "p_draw": prediction.p_draw,
            "p_away": prediction.p_away,
            "created_at": _utc(prediction.created_at),
            "ledger_batch_id": batch.batch_id,
            "ledger_batch_hash": batch.sha256,
            "ledger_file_url": public_url,
        }

    @app.get("/scorecard", response_model=ScoresResponse)
    def scorecard(
        response: Response,
        session: Session = session_dependency,
    ) -> ScoresResponse:
        score_rows = session.scalars(select(Score).order_by(Score.computed_at)).all()
        grouped: dict[tuple[str, str, str, str | None, str | None], list[Score]] = defaultdict(list)
        for score in score_rows:
            meta = score.metadata_json or {}
            grouped[
                (
                    score.forecaster,
                    score.version,
                    score.metric,
                    meta.get("league"),
                    meta.get("season"),
                )
            ].append(score)
        results: list[dict[str, Any]] = []
        for (forecaster, version, metric, league, season), rows in sorted(grouped.items()):
            values = [row.value for row in rows]
            results.append(
                {
                    "forecaster": forecaster,
                    "version": version,
                    "metric": metric,
                    "value": sum(values) / len(values),
                    "sample_count": len(values),
                    "league": league,
                    "season": season,
                    "confidence_interval": _confidence_interval(rows),
                }
            )
        add_cache(response)
        return ScoresResponse(
            scores=results,
            matches_scored=len({row.match_id for row in score_rows if row.metric == "log_loss"}),
            last_updated=max((_utc(row.computed_at) for row in score_rows), default=None),
        )

    @app.get("/scorecard/calibration", response_model=CalibrationResponse)
    def calibration(
        response: Response,
        session: Session = session_dependency,
        bins: int = Query(default=10, ge=1, le=50),
    ) -> CalibrationResponse:
        rows = session.execute(
            select(Prediction, Match)
            .join(Match, Prediction.match_id == Match.id)
            .where(Match.status == "finished", Match.result.is_not(None))
            .order_by(Prediction.created_at.desc())
        ).all()
        latest_by_match = {}
        for prediction, match in rows:
            latest_by_match.setdefault(match.id, (prediction, match))
        rows = list(latest_by_match.values())
        outcomes = ("Home", "Draw", "Away")
        data: list[dict[str, Any]] = []
        for outcome_index, outcome in enumerate(outcomes):
            for bin_index in range(bins):
                values = []
                positives = []
                for prediction, match in rows:
                    probabilities = (prediction.p_home, prediction.p_draw, prediction.p_away)
                    confidence = probabilities[outcome_index]
                    current_bin = min(int(confidence * bins), bins - 1)
                    if current_bin != bin_index:
                        continue
                    values.append(confidence)
                    positives.append(int(match.result == ("H", "D", "A")[outcome_index]))
                if values:
                    data.append(
                        {
                            "outcome": outcome,
                            "bin": bin_index,
                            "mean_predicted": sum(values) / len(values),
                            "fraction_positive": sum(positives) / len(positives),
                            "count": len(values),
                        }
                    )
        add_cache(response)
        return CalibrationResponse(
            bins=data,
            outcomes=list(outcomes),
            matches_scored=len({match.id for _, match in rows}),
        )

    @app.get("/matches/recent", response_model=RecentResponse)
    def recent_matches(
        response: Response,
        session: Session = session_dependency,
        limit: int = Query(default=25, ge=1, le=100),
        league: str | None = Query(default=None, pattern="^(PL|PD)$"),
    ) -> RecentResponse:
        home_alias = aliased(Team)
        away_alias = aliased(Team)
        statement = (
            select(Match, Prediction, home_alias, away_alias)
            .join(Prediction, Prediction.match_id == Match.id)
            .join(home_alias, Match.home_team_id == home_alias.id)
            .join(away_alias, Match.away_team_id == away_alias.id)
            .where(Match.status == "finished", Match.result.is_not(None))
            .order_by(Match.kickoff_utc.desc(), Prediction.created_at.desc())
        )
        if league is not None:
            statement = statement.where(Match.league == league)
        items = []
        seen_match_ids: set[str] = set()
        for match, prediction, home, away in session.execute(statement):
            if match.id in seen_match_ids:
                continue
            seen_match_ids.add(match.id)
            items.append(
                {
                    "match": _match_json(match, home, away),
                    "prediction": _prediction_json(prediction),
                    "result": match.result,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                }
            )
            if len(items) >= limit:
                break
        add_cache(response)
        return RecentResponse(items=items)

    @app.get("/verify/{batch_id}")
    def verify_batch(
        batch_id: str,
        request: Request,
        session: Session = session_dependency,
    ) -> dict[str, Any]:
        batch = session.get(LedgerBatch, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="ledger batch not found")
        created_at = _utc(batch.created_at)
        filename = f"{created_at.date().isoformat()}_{batch.batch_id}.json"
        app_settings = request.app.state.settings
        ledger_path = app_settings.ledger_dir / filename
        verification = verify_file(
            ledger_path,
            session=session,
            public_repository_url=app_settings.ledger_public_url,
            public_branch=app_settings.ledger_public_branch,
            public_path=app_settings.ledger_public_path,
            require_public=True,
        )
        return {
            "batch_id": batch_id,
            "sha256": verification.sha256,
            "verified": verification.verified,
            "local_file": verification.local_file,
            "database_status": verification.database_status,
            "public_status": verification.public_status,
            "timestamp_proof_status": verification.timestamp_proof_status,
            "reason": verification.reason,
        }

    return app


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _match_json(match: Match, home: Team, away: Team) -> dict[str, Any]:
    return {
        "match_id": match.id,
        "league": match.league,
        "season": match.season,
        "kickoff_utc": _utc(match.kickoff_utc),
        "home_team": home.canonical_name,
        "away_team": away.canonical_name,
        "status": match.status,
        "result": match.result,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
    }


def _prediction_json(prediction: Prediction) -> dict[str, Any]:
    return {
        "model_version": prediction.model_version,
        "forecast_policy_version": prediction.forecast_policy_version,
        "p_home": prediction.p_home,
        "p_draw": prediction.p_draw,
        "p_away": prediction.p_away,
        "created_at": _utc(prediction.created_at),
        "batch_id": prediction.batch_id,
    }


def _confidence_interval(rows: list[Score]) -> dict[str, float] | None:
    intervals = [
        row.metadata_json.get("confidence_interval")
        for row in rows
        if isinstance(row.metadata_json, dict)
        and isinstance(row.metadata_json.get("confidence_interval"), dict)
    ]
    if not intervals:
        return None
    return intervals[-1]


app = create_app()

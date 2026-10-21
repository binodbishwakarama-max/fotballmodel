"""Relational schema for fixtures, provenance, forecasts, and scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast.db import Base


class Team(Base):
    """Canonical football team."""

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league", "canonical_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(160), index=True)
    league: Mapped[str] = mapped_column(String(8), index=True)
    aliases: Mapped[list[TeamAlias]] = relationship(back_populates="team")


class TeamAlias(Base):
    """Source-specific name mapped to a canonical team."""

    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("source", "normalized_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(80))
    raw_name: Mapped[str] = mapped_column(String(160))
    normalized_name: Mapped[str] = mapped_column(String(160))
    team: Mapped[Team] = relationship(back_populates="aliases")


class Match(Base):
    """Scheduled or completed fixture with explicit result availability."""

    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("league", "season", "kickoff_utc", "home_team_id", "away_team_id"),
        UniqueConstraint("source", "source_match_id"),
        CheckConstraint("home_team_id != away_team_id", name="different_teams"),
        Index("ix_matches_kickoff", "kickoff_utc"),
    )

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    league: Mapped[str] = mapped_column(String(8), index=True)
    season: Mapped[str] = mapped_column(String(9), index=True)
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kickoff_time_known: Mapped[bool] = mapped_column(default=True)
    matchweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[str | None] = mapped_column(String(1), nullable=True)
    result_available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    source: Mapped[str] = mapped_column(String(80))
    source_match_id: Mapped[str | None] = mapped_column(String(100), nullable=True)


class MatchSource(Base):
    """External identifier and source provenance for a canonical fixture."""

    __tablename__ = "match_sources"
    __table_args__ = (UniqueConstraint("source", "source_match_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(80))
    source_match_id: Mapped[str] = mapped_column(String(100))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Odds(Base):
    """Raw decimal odds with type and availability provenance."""

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("match_id", "source", "kind", "available_at"),
        CheckConstraint("home > 1 AND draw > 1 AND away > 1", name="valid_decimal_odds"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(80))
    home: Mapped[float] = mapped_column(Float)
    draw: Mapped[float] = mapped_column(Float)
    away: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(8))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    availability_verified: Mapped[bool] = mapped_column(default=False)


class ExpectedGoals(Base):
    """Match expected-goals observations and their provenance."""

    __tablename__ = "xg"
    __table_args__ = (UniqueConstraint("match_id", "source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    home_xg: Mapped[float] = mapped_column(Float)
    away_xg: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(80))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    availability_verified: Mapped[bool] = mapped_column(default=False)


class IngestionRun(Base):
    """Auditable data-ingestion run and quality counters."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20))
    rows_read: Mapped[int] = mapped_column(Integer, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Prediction(Base):
    """Append-only published forecast record."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "model_version", "forecast_policy_version", name="uq_prediction_policy"
        ),
        CheckConstraint("p_home >= 0 AND p_draw >= 0 AND p_away >= 0", name="nonnegative_probs"),
        CheckConstraint(
            "p_home + p_draw + p_away BETWEEN 0.999999 AND 1.000001",
            name="probabilities_sum_to_one",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    model_version: Mapped[str] = mapped_column(String(120))
    forecast_policy_version: Mapped[str] = mapped_column(String(80))
    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    features_hash: Mapped[str] = mapped_column(String(64))
    batch_id: Mapped[str] = mapped_column(ForeignKey("ledger_batches.batch_id"))


class LedgerBatch(Base):
    """Canonical prediction batch and its content hash."""

    __tablename__ = "ledger_batches"

    batch_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    n_predictions: Mapped[int] = mapped_column(Integer)
    public_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp_proof_status: Mapped[str] = mapped_column(String(20), default="pending")
    timestamp_proof_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Score(Base):
    """Computed scorecard value for a model/baseline and scoring slice."""

    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint("match_id", "forecaster", "version", "metric", name="uq_match_score"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    forecaster: Mapped[str] = mapped_column(String(40))
    version: Mapped[str] = mapped_column(String(120))
    metric: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FeatureSnapshot(Base):
    """Persisted per-fixture feature vector and provenance."""

    __tablename__ = "feature_snapshots"
    __table_args__ = (UniqueConstraint("match_id", "forecast_cutoff"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    forecast_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    features_hash: Mapped[str] = mapped_column(String(64))
    unavailable_feature_count: Mapped[int] = mapped_column(Integer, default=0)


class PredictionExplanation(Base):
    """Factual explanation generated once for a published prediction."""

    __tablename__ = "prediction_explanations"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True
    )
    explanation: Mapped[str] = mapped_column(String(1000))
    cited_fact_keys: Mapped[list[str]] = mapped_column(JSON)
    facts_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    evaluation_status: Mapped[str] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

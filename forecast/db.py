"""Database engine and schema initialization."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Declarative base for database models."""


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine and prepare a local SQLite data directory."""
    from forecast.config import normalize_database_url

    database_url = normalize_database_url(database_url)
    if database_url.startswith("sqlite:///") and not database_url.startswith("sqlite:////"):
        relative_path = database_url.removeprefix("sqlite:///").split("?", maxsplit=1)[0]
        if relative_path and relative_path != ":memory:":
            Path(relative_path).parent.mkdir(parents=True, exist_ok=True)
    engine_options = {}
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        engine_options = {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    engine = create_engine(database_url, future=True, **engine_options)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_schema(engine: Engine) -> None:
    """Create tables and append-only triggers for an empty or existing database."""
    from forecast import models as _models  # noqa: F401

    Base.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS predictions_no_update "
                    "BEFORE UPDATE ON predictions BEGIN "
                    "SELECT RAISE(ABORT, 'predictions are append-only'); END"
                )
            )
            connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS predictions_no_delete "
                    "BEFORE DELETE ON predictions BEGIN "
                    "SELECT RAISE(ABORT, 'predictions are append-only'); END"
                )
            )
    elif engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE OR REPLACE FUNCTION reject_prediction_mutation()
                    RETURNS trigger AS $$
                    BEGIN
                        RAISE EXCEPTION 'predictions are append-only';
                    END;
                    $$ LANGUAGE plpgsql
                    """
                )
            )
            connection.execute(
                text("DROP TRIGGER IF EXISTS predictions_no_mutation ON predictions")
            )
            connection.execute(
                text(
                    """
                    CREATE TRIGGER predictions_no_mutation
                    BEFORE UPDATE OR DELETE ON predictions
                    FOR EACH ROW EXECUTE FUNCTION reject_prediction_mutation()
                    """
                )
            )

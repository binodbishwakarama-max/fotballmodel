"""Tamper and append-only tests for prediction ledger batches."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.ledger.ledger import create_batch, verify_file
from forecast.models import LedgerBatch, Match, Prediction, Team

CREATED_AT = datetime(2026, 10, 21, 18, tzinfo=UTC)
KICKOFF_AT = datetime(2026, 10, 22, 19, tzinfo=UTC)


def seed_match(session: Session) -> None:
    home = Team(canonical_name="Home", league="PL")
    away = Team(canonical_name="Away", league="PL")
    session.add_all((home, away))
    session.flush()
    session.add(
        Match(
            id="match-1",
            league="PL",
            season="2026-27",
            kickoff_utc=KICKOFF_AT,
            kickoff_time_known=True,
            home_team_id=home.id,
            away_team_id=away.id,
            status="scheduled",
            source="test",
        )
    )
    session.commit()


def prediction() -> dict:
    return {
        "match_id": "match-1",
        "kickoff_utc": KICKOFF_AT,
        "model_version": "test-model",
        "p_home": 0.6,
        "p_draw": 0.25,
        "p_away": 0.15,
        "features_hash": "a" * 64,
    }


def test_batch_creates_canonical_json_and_verifies_db(tmp_path: Path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        seed_match(session)
        batch, path = create_batch(
            session,
            predictions=[prediction()],
            ledger_dir=tmp_path,
            created_at=CREATED_AT,
        )
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        assert '"p_home":0.600000' in raw
        assert payload["batch_id"] == batch.batch_id
        assert payload["sha256"] == batch.sha256
        assert batch.n_predictions == 1
        verification = verify_file(path, session=session, require_public=False)
        assert verification.verified
        assert verification.database_status == "match"
        assert verification.public_status == "not_configured"


def test_tampered_file_fails_verification(tmp_path: Path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        seed_match(session)
        _, path = create_batch(
            session,
            predictions=[prediction()],
            ledger_dir=tmp_path,
            created_at=CREATED_AT,
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace("0.600000", "0.610000"),
            encoding="utf-8",
        )
        verification = verify_file(path, session=session, require_public=False)

    assert not verification.verified
    assert "hash does not match" in verification.reason


def test_publication_verification_reconstructs_from_database(tmp_path: Path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        seed_match(session)
        _, path = create_batch(
            session,
            predictions=[prediction()],
            ledger_dir=tmp_path,
            created_at=CREATED_AT,
        )
        path.unlink()
        verification = verify_file(path, session=session, require_public=False)

    assert verification.verified
    assert verification.database_status == "match"
    assert verification.local_file.startswith("database-reconstructed:")


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_prediction_rows_are_append_only(tmp_path: Path, operation: str) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        seed_match(session)
        _, _ = create_batch(
            session,
            predictions=[prediction()],
            ledger_dir=tmp_path,
            created_at=CREATED_AT,
        )
        statement = (
            update(Prediction).values(p_home=0.5) if operation == "update" else delete(Prediction)
        )
        with pytest.raises(IntegrityError, match="append-only"):
            session.execute(statement)
            session.commit()
        session.rollback()
        assert session.query(Prediction).count() == 1
        assert session.query(LedgerBatch).count() == 1


def test_batch_rejects_late_or_unknown_fixtures(tmp_path: Path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        seed_match(session)
        late = datetime(2026, 10, 22, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="24 to 30 hours"):
            create_batch(
                session,
                predictions=[prediction()],
                ledger_dir=tmp_path,
                created_at=late,
            )
        unknown = prediction()
        unknown["match_id"] = "missing"
        with pytest.raises(ValueError, match="unknown match"):
            create_batch(
                session,
                predictions=[unknown],
                ledger_dir=tmp_path,
                created_at=CREATED_AT,
            )
        assert session.query(Prediction).count() == 0
        assert session.query(LedgerBatch).count() == 0

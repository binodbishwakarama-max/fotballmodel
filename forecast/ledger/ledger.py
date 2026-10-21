"""Canonical append-only prediction batches and public-reference verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from forecast.models import LedgerBatch, Match, Prediction

FORECAST_POLICY_VERSION = "t-minus-24h-v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LedgerVerification:
    """Detailed outcome of local hash, database, and optional public-reference checks."""

    verified: bool
    batch_id: str | None
    sha256: str | None
    local_file: str
    database_status: str
    public_status: str
    timestamp_proof_status: str
    reason: str | None = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_json(value: Any) -> bytes:
    """Serialize JSON with sorted keys and fixed six-place decimal float tokens."""
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, str):
        return _json_string(value).encode("utf-8")
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit non-finite numbers")
        return f"{value:.6f}".encode("ascii")
    if isinstance(value, list):
        return b"[" + b",".join(canonical_json(item) for item in value) + b"]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical JSON object keys must be strings")
        entries = [canonical_json(key) + b":" + canonical_json(value[key]) for key in sorted(value)]
        return b"{" + b",".join(entries) + b"}"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_batch_bytes(session: Session, batch: LedgerBatch) -> bytes:
    """Rebuild a batch file exactly from durable database rows and check its digest."""
    rows = session.execute(
        select(Prediction, Match.kickoff_utc)
        .join(Match, Match.id == Prediction.match_id)
        .where(Prediction.batch_id == batch.batch_id)
        .order_by(
            Prediction.match_id,
            Prediction.model_version,
            Prediction.forecast_policy_version,
        )
    ).all()
    predictions = [
        {
            "match_id": prediction.match_id,
            "kickoff_utc": _utc(kickoff).isoformat().replace("+00:00", "Z"),
            "model_version": prediction.model_version,
            "forecast_policy_version": prediction.forecast_policy_version,
            "p_home": prediction.p_home,
            "p_draw": prediction.p_draw,
            "p_away": prediction.p_away,
            "features_hash": prediction.features_hash,
        }
        for prediction, kickoff in rows
    ]
    if not predictions or len(predictions) != batch.n_predictions:
        raise ValueError(f"prediction count does not match ledger batch {batch.batch_id}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch.batch_id,
        "created_at": _utc(batch.created_at).isoformat().replace("+00:00", "Z"),
        "predictions": predictions,
    }
    computed_hash = hashlib.sha256(canonical_json(body)).hexdigest()
    if computed_hash != batch.sha256:
        raise ValueError(f"database rows do not reproduce ledger hash for {batch.batch_id}")
    return canonical_json({**body, "sha256": computed_hash}) + b"\n"


def validate_forecast_window(kickoff_utc: datetime, created_at: datetime) -> None:
    """Require canonical forecasts to be published 24–30 hours before kickoff."""
    delta = _utc(kickoff_utc) - _utc(created_at)
    if not (24 * 3600 <= delta.total_seconds() <= 30 * 3600):
        raise ValueError(
            "canonical forecast must be published 24 to 30 hours before kickoff "
            f"(observed {delta.total_seconds() / 3600:.2f} hours)"
        )


def create_batch(
    session: Session,
    *,
    predictions: list[dict[str, Any]],
    ledger_dir: Path,
    created_at: datetime | None = None,
) -> tuple[LedgerBatch, Path]:
    """Validate and persist an immutable prediction batch and canonical ledger file."""
    if created_at is not None and created_at.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    timestamp = _utc(created_at or datetime.now(UTC))
    if not predictions:
        raise ValueError("cannot create an empty ledger batch")
    batch_id = str(uuid.uuid4())
    canonical_predictions = []
    unique_keys: set[tuple[str, str, str]] = set()
    for item in predictions:
        required = {
            "match_id",
            "kickoff_utc",
            "model_version",
            "p_home",
            "p_draw",
            "p_away",
            "features_hash",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"prediction is missing required fields: {sorted(missing)}")
        kickoff = _parse_datetime(item["kickoff_utc"])
        validate_forecast_window(kickoff, timestamp)
        match = session.get(Match, str(item["match_id"]))
        if match is None:
            raise ValueError(f"prediction references unknown match {item['match_id']!r}")
        if not match.kickoff_time_known:
            raise ValueError(f"match {match.id} has no verified kickoff time")
        if _utc(match.kickoff_utc) != kickoff:
            raise ValueError(f"provided kickoff does not match database fixture {match.id}")
        probabilities = [
            float(item["p_home"]),
            float(item["p_draw"]),
            float(item["p_away"]),
        ]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("forecast probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("forecast probabilities must sum to one")
        policy = str(item.get("forecast_policy_version", FORECAST_POLICY_VERSION))
        key = (str(item["match_id"]), str(item["model_version"]), policy)
        if key in unique_keys:
            raise ValueError(f"duplicate prediction in batch: {key}")
        unique_keys.add(key)
        features_hash = str(item["features_hash"])
        if re.fullmatch(r"[0-9a-f]{64}", features_hash) is None:
            raise ValueError("features_hash must be a lowercase SHA-256 hex digest")
        canonical_predictions.append(
            {
                "match_id": str(item["match_id"]),
                "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
                "model_version": str(item["model_version"]),
                "forecast_policy_version": policy,
                "p_home": probabilities[0],
                "p_draw": probabilities[1],
                "p_away": probabilities[2],
                "features_hash": features_hash,
            }
        )
    canonical_predictions.sort(
        key=lambda item: (item["match_id"], item["model_version"], item["forecast_policy_version"])
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "predictions": canonical_predictions,
    }
    digest = hashlib.sha256(canonical_json(body)).hexdigest()
    file_content = canonical_json({**body, "sha256": digest}) + b"\n"
    filename = f"{timestamp.date().isoformat()}_{batch_id}.json"
    path = ledger_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"ledger batch path already exists: {path}")
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as ledger_file:
            staging_path = Path(ledger_file.name)
            ledger_file.write(file_content)
            ledger_file.flush()
            os.fsync(ledger_file.fileno())
        os.replace(staging_path, path)
    except OSError:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        raise

    batch = LedgerBatch(
        batch_id=batch_id,
        created_at=timestamp,
        sha256=digest,
        n_predictions=len(canonical_predictions),
        timestamp_proof_status="pending",
    )
    session.add(batch)
    try:
        session.flush()
        for item in canonical_predictions:
            session.add(
                Prediction(
                    id=str(uuid.uuid4()),
                    match_id=item["match_id"],
                    model_version=item["model_version"],
                    forecast_policy_version=item["forecast_policy_version"],
                    p_home=item["p_home"],
                    p_draw=item["p_draw"],
                    p_away=item["p_away"],
                    created_at=timestamp,
                    features_hash=item["features_hash"],
                    batch_id=batch_id,
                )
            )
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        path.unlink(missing_ok=True)
        raise
    return batch, path


def verify_file(
    path: Path,
    *,
    session: Session | None = None,
    public_repository_url: str | None = None,
    public_branch: str = "main",
    public_path: str = "",
    require_public: bool = True,
    client: httpx.Client | None = None,
) -> LedgerVerification:
    """Verify canonical file bytes, optional DB metadata, public copy, and proof status."""
    ledger_content: bytes | None = None
    reconstructed = False
    if path.is_file():
        try:
            ledger_content = path.read_bytes()
        except OSError as exc:
            return LedgerVerification(
                False, None, None, str(path), "not_checked", "not_checked", "not_checked", str(exc)
            )
    elif session is not None:
        filename_parts = path.stem.split("_", maxsplit=1)
        batch = session.get(LedgerBatch, filename_parts[1]) if len(filename_parts) == 2 else None
        if batch is not None:
            try:
                ledger_content = canonical_batch_bytes(session, batch)
                reconstructed = True
            except (TypeError, ValueError) as exc:
                return LedgerVerification(
                    False,
                    batch.batch_id,
                    batch.sha256,
                    f"database-reconstructed:{path}" if reconstructed else str(path),
                    "mismatch",
                    "not_checked",
                    batch.timestamp_proof_status,
                    str(exc),
                )
    if ledger_content is None:
        return LedgerVerification(
            False,
            None,
            None,
            str(path),
            "not_checked",
            "not_checked",
            "not_checked",
            "ledger file does not exist",
        )
    try:
        payload = json.loads(ledger_content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ledger root must be a JSON object")
        stored_hash = payload.pop("sha256", None)
        if not isinstance(stored_hash, str) or re.fullmatch(r"[0-9a-f]{64}", stored_hash) is None:
            raise ValueError("missing or malformed SHA-256 hash")
        computed_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
        if computed_hash != stored_hash:
            raise ValueError("ledger file hash does not match its content")
        batch_id = payload.get("batch_id")
        predictions = payload.get("predictions")
        if not isinstance(batch_id, str) or not isinstance(predictions, list):
            raise ValueError("ledger file is missing batch metadata")
        unique_prediction_keys = {
            (
                item.get("match_id"),
                item.get("model_version"),
                item.get("forecast_policy_version"),
            )
            for item in predictions
            if isinstance(item, dict)
        }
        if len(predictions) != len(unique_prediction_keys):
            raise ValueError("ledger contains duplicate match forecasts")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return LedgerVerification(
            False, None, None, str(path), "not_checked", "not_checked", "not_checked", str(exc)
        )

    database_status = "not_checked"
    proof_status = "pending"
    if session is not None:
        batch = session.get(LedgerBatch, batch_id)
        if batch is None:
            database_status = "missing"
        elif batch.sha256 != stored_hash or batch.n_predictions != len(predictions):
            database_status = "mismatch"
        else:
            database_status = "match"
            proof_status = batch.timestamp_proof_status

    public_status = "not_configured"
    if public_repository_url:
        try:
            public_content = _fetch_public_copy(
                public_repository_url,
                public_branch,
                "/".join(part for part in (public_path.strip("/"), path.name) if part),
                client=client,
            )
        except (httpx.HTTPError, ValueError) as exc:
            public_status = f"error: {exc}"
        else:
            try:
                public_payload = json.loads(public_content)
                public_hash = public_payload.get("sha256")
                public_computed = hashlib.sha256(
                    canonical_json(
                        {key: value for key, value in public_payload.items() if key != "sha256"}
                    )
                ).hexdigest()
                if public_hash == stored_hash and public_computed == stored_hash:
                    public_status = "match"
                else:
                    public_status = "mismatch"
            except (json.JSONDecodeError, TypeError, ValueError):
                public_status = "invalid"

    local_database_ok = database_status in {"not_checked", "match"}
    public_ok = (
        public_status == "match"
        if require_public
        else public_status
        in {
            "match",
            "not_configured",
        }
    )
    verified = local_database_ok and public_ok
    reason = None
    if database_status in {"missing", "mismatch"}:
        reason = "database batch metadata is missing or does not match"
    elif require_public and public_status != "match":
        reason = "public repository copy could not be independently verified"
    return LedgerVerification(
        verified,
        batch_id,
        stored_hash,
        f"database-reconstructed:{path}" if reconstructed else str(path),
        database_status,
        public_status,
        proof_status,
        reason,
    )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("kickoff_utc must include a timezone")
        return _utc(value)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("kickoff_utc must include a timezone")
        return _utc(parsed)
    raise ValueError("kickoff_utc must be an ISO timestamp or datetime")


def _fetch_public_copy(
    repository_url: str,
    branch: str,
    filename: str,
    *,
    client: httpx.Client | None,
) -> bytes:
    match = re.fullmatch(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?", repository_url)
    if match is None:
        raise ValueError("public ledger URL must be a GitHub repository URL")
    repository = match.group(1).removesuffix(".git")
    url = (
        f"https://raw.githubusercontent.com/{repository}/"
        f"{quote(branch, safe='')}/{quote(filename, safe='/')}"
    )
    own_client = client is None
    requester = client or httpx.Client(timeout=httpx.Timeout(20.0))
    try:
        response = requester.get(url)
        response.raise_for_status()
        return response.content
    finally:
        if own_client:
            requester.close()

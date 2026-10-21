"""Tests for retry-safe public ledger and timestamp-proof publication."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy.orm import Session

from forecast.config import Settings
from forecast.db import create_schema, get_engine
from forecast.ledger.ledger import create_batch
from forecast.ledger.publisher import GitHubLedgerPublisher
from forecast.models import LedgerBatch, Match, Team

CREATED_AT = datetime(2026, 10, 21, 18, tzinfo=UTC)
KICKOFF_AT = datetime(2026, 10, 22, 19, tzinfo=UTC)


def _settings() -> Settings:
    return replace(
        Settings.from_env(),
        ledger_public_url="https://github.com/example/forecast-ledger",
        ledger_public_branch="main",
        ledger_public_path="ledger",
        github_token="test-token",
    )


def _seed_batch(engine, ledger_dir: Path) -> str:
    with Session(engine) as session:
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
        batch, _ = create_batch(
            session,
            predictions=[
                {
                    "match_id": "match-1",
                    "kickoff_utc": KICKOFF_AT,
                    "model_version": "model-v1",
                    "p_home": 0.6,
                    "p_draw": 0.25,
                    "p_away": 0.15,
                    "features_hash": "a" * 64,
                }
            ],
            ledger_dir=ledger_dir,
            created_at=CREATED_AT,
        )
        return batch.batch_id


def _github_transport():
    files: dict[str, bytes] = {}
    commit_ids: dict[str, str] = {}
    commit_counter = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal commit_counter
        path = request.url.path
        if request.url.host == "api.github.com" and "/contents/" in path:
            relative_path = path.split("/contents/", maxsplit=1)[1]
            if request.method == "GET":
                content = files.get(relative_path)
                if content is None:
                    return httpx.Response(404)
                return httpx.Response(
                    200,
                    json={
                        "encoding": "base64",
                        "content": base64.b64encode(content).decode("ascii"),
                        "sha": hashlib.sha1(content).hexdigest(),
                    },
                )
            if request.method == "PUT":
                payload = json.loads(request.content)
                files[relative_path] = base64.b64decode(payload["content"])
                commit_counter += 1
                commit_ids[relative_path] = f"commit-{commit_counter}"
                return httpx.Response(201, json={"commit": {"sha": commit_ids[relative_path]}})
        if request.url.host == "api.github.com" and path.endswith("/commits"):
            relative_path = request.url.params["path"]
            sha = commit_ids.get(relative_path)
            return httpx.Response(200, json=[{"sha": sha}] if sha else [])
        if request.url.host == "raw.githubusercontent.com":
            relative_path = path.split("/main/", maxsplit=1)[1]
            content = files.get(relative_path)
            if content is not None:
                return httpx.Response(200, content=content)
            return httpx.Response(404)
        return httpx.Response(404)

    return files, httpx.MockTransport(handle)


def test_publish_is_retry_safe_and_proof_upgrade_is_separate(tmp_path: Path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    batch_id = _seed_batch(engine, tmp_path)
    files, transport = _github_transport()
    verify_count = 0

    def run_ots(args, **_kwargs):
        nonlocal verify_count
        operation, target = args[1], Path(args[2])
        if operation == "stamp":
            target.with_name(f"{target.name}.ots").write_bytes(b"pending-proof")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "upgrade":
            sidecar = target if target.suffix == ".ots" else target.with_name(f"{target.name}.ots")
            sidecar.write_bytes(sidecar.read_bytes() + b"-anchored")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        verify_count += 1
        message = "Success! Bitcoin block confirmed" if verify_count > 1 else "Pending confirmation"
        return SimpleNamespace(returncode=0, stdout=message, stderr="")

    client = httpx.Client(transport=transport)
    publisher = GitHubLedgerPublisher(
        _settings(), client=client, command_runner=run_ots, sleep=lambda _seconds: None
    )
    assert publisher.publish_pending(engine) == {
        "published": 1,
        "already_published": 0,
        "failed": 0,
    }
    ledger_path = f"ledger/{CREATED_AT.date()}_{batch_id}.json"
    with Session(engine) as session:
        expected_hash = session.get(LedgerBatch, batch_id).sha256
    assert json.loads(files[ledger_path])["sha256"] == expected_hash
    assert files[f"{ledger_path}.ots"] == b"pending-proof"

    with Session(engine) as session:
        saved = session.get(LedgerBatch, batch_id)
        assert saved is not None
        assert saved.public_commit == "commit-1"
        assert saved.timestamp_proof_status == "pending"
        assert saved.timestamp_proof_commit == "commit-2"

    assert publisher.publish_pending(engine) == {
        "published": 0,
        "already_published": 0,
        "failed": 0,
    }
    assert publisher.upgrade_pending_proofs(engine) == {
        "verified": 1,
        "pending": 0,
        "failed": 0,
    }
    assert files[f"{ledger_path}.ots"] == b"pending-proof-anchored"
    with Session(engine) as session:
        saved = session.get(LedgerBatch, batch_id)
        assert saved is not None
        assert saved.timestamp_proof_status == "verified"
        assert saved.timestamp_proof_commit == "commit-3"
    publisher.close()


def test_public_ledger_overwrite_is_rejected(tmp_path: Path) -> None:
    files, transport = _github_transport()
    files["ledger/existing.json"] = b"immutable"
    publisher = GitHubLedgerPublisher(
        _settings(),
        client=httpx.Client(transport=transport),
        command_runner=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite immutable"):
        publisher._put_file(
            "ledger/existing.json",
            b"different",
            message="test",
            allow_update=False,
        )
    publisher.close()

"""Append-only GitHub publication and OpenTimestamps proof lifecycle."""

from __future__ import annotations

import base64
import logging
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.config import Settings
from forecast.ledger.ledger import canonical_batch_bytes
from forecast.models import LedgerBatch

logger = logging.getLogger(__name__)
_GITHUB_REPOSITORY = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")


class GitHubLedgerPublisher:
    """Publish immutable ledger JSON and upgradeable OpenTimestamps sidecars."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.ledger_public_url:
            raise RuntimeError("BWB_LEDGER_PUBLIC_URL must be configured before publishing")
        if not settings.github_token:
            raise RuntimeError("GITHUB_TOKEN must be configured before publishing")
        match = _GITHUB_REPOSITORY.fullmatch(settings.ledger_public_url)
        if match is None:
            raise ValueError("BWB_LEDGER_PUBLIC_URL must be a GitHub repository URL")
        self.repository = f"{match.group(1)}/{match.group(2).removesuffix('.git')}"
        self.branch = settings.ledger_public_branch
        self.public_path = settings.ledger_public_path
        self.command = settings.opentimestamps_command
        self._command_runner = command_runner
        self._sleep = sleep
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(20.0),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {settings.github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        self.raw_base = (
            f"https://raw.githubusercontent.com/{self.repository}/{quote(self.branch, safe='')}"
        )
        self.api_base = f"https://api.github.com/repos/{self.repository}"

    def close(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_client:
            self.client.close()

    def publish_pending(self, engine: Any) -> dict[str, int]:
        """Publish each unpublished batch, safely retrying prior successful writes."""
        counts = {"published": 0, "already_published": 0, "failed": 0}
        with Session(engine) as session:
            batch_ids = session.scalars(
                select(LedgerBatch.batch_id)
                .where(LedgerBatch.public_commit.is_(None))
                .order_by(LedgerBatch.created_at, LedgerBatch.batch_id)
            ).all()
            for batch_id in batch_ids:
                try:
                    changed = self._publish_batch(session, batch_id)
                    session.commit()
                    counts["published" if changed else "already_published"] += 1
                except Exception:
                    session.rollback()
                    counts["failed"] += 1
                    logger.exception("Ledger batch publication failed for batch_id=%s", batch_id)
            if counts["failed"]:
                raise RuntimeError(
                    f"Ledger publishing had {counts['failed']} failure(s); "
                    f"{counts['published']} published and "
                    f"{counts['already_published']} already published"
                )
        return counts

    def upgrade_pending_proofs(self, engine: Any) -> dict[str, int]:
        """Upgrade pending sidecars and verify them without rewriting ledger JSON."""
        counts = {"verified": 0, "pending": 0, "failed": 0}
        with Session(engine) as session:
            batches = session.scalars(
                select(LedgerBatch)
                .where(
                    LedgerBatch.public_commit.is_not(None),
                    LedgerBatch.timestamp_proof_status != "verified",
                )
                .order_by(LedgerBatch.created_at, LedgerBatch.batch_id)
            ).all()
            for batch in batches:
                try:
                    ledger = self._rebuild_batch(session, batch)
                    filename = f"{batch.created_at.date().isoformat()}_{batch.batch_id}.json"
                    ledger_path = "/".join(part for part in (self.public_path, filename) if part)
                    proof_path = f"{ledger_path}.ots"
                    existing = self._get_file(proof_path)
                    if existing is None:
                        raise RuntimeError(f"published timestamp proof is missing: {proof_path}")
                    proof = self._upgrade_proof(ledger, filename, existing[0])
                    if proof != existing[0]:
                        commit, _ = self._put_file(
                            proof_path,
                            proof,
                            message=f"Upgrade timestamp proof for batch {batch.batch_id}",
                            allow_update=True,
                        )
                        if commit:
                            batch.timestamp_proof_commit = commit
                    batch.timestamp_proof_status = self._verify_proof(ledger, proof)
                    counts[batch.timestamp_proof_status] += 1
                    session.commit()
                except Exception:
                    session.rollback()
                    counts["failed"] += 1
                    logger.exception(
                        "Timestamp proof upgrade failed for batch_id=%s", batch.batch_id
                    )
            if counts["failed"]:
                raise RuntimeError(f"Timestamp proof upgrade had {counts['failed']} failure(s)")
        return counts

    def _publish_batch(self, session: Session, batch_id: str) -> bool:
        batch = session.get(LedgerBatch, batch_id)
        if batch is None:
            raise ValueError(f"ledger batch disappeared: {batch_id}")
        content = self._rebuild_batch(session, batch)
        filename = f"{batch.created_at.date().isoformat()}_{batch.batch_id}.json"
        relative_path = "/".join(part for part in (self.public_path, filename) if part)

        ledger_commit, ledger_changed = self._put_file(
            relative_path,
            content,
            message=f"Publish forecast ledger batch {batch.batch_id}",
            allow_update=False,
        )
        if ledger_commit:
            batch.public_commit = ledger_commit
            session.flush()
        self._verify_public_copy(relative_path, content)

        proof_path = f"{relative_path}.ots"
        proof = self._create_or_upgrade_proof(relative_path, content, proof_path)
        proof_commit, proof_changed = self._put_file(
            proof_path,
            proof,
            message=f"Publish timestamp proof for batch {batch.batch_id}",
            allow_update=True,
        )
        if proof_commit:
            batch.timestamp_proof_commit = proof_commit
        batch.timestamp_proof_status = self._verify_proof(content, proof)
        session.flush()
        return ledger_changed or proof_changed

    def _rebuild_batch(self, session: Session, batch: LedgerBatch) -> bytes:
        return canonical_batch_bytes(session, batch)

    def _get_file(self, path: str) -> tuple[bytes, str] | None:
        response = self.client.get(
            f"{self.api_base}/contents/{quote(path, safe='/')}",
            params={"ref": self.branch},
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub contents lookup failed with HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("encoding") != "base64":
            raise RuntimeError("GitHub returned an unsupported contents response")
        try:
            content = base64.b64decode(payload["content"], validate=False)
            blob_sha = payload["sha"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("GitHub contents response was malformed") from exc
        if not isinstance(blob_sha, str):
            raise RuntimeError("GitHub contents response omitted its blob SHA")
        return content, blob_sha

    def _put_file(
        self, path: str, content: bytes, *, message: str, allow_update: bool
    ) -> tuple[str | None, bool]:
        existing = self._get_file(path)
        encoded = base64.b64encode(content).decode("ascii")
        if existing is not None:
            existing_content, blob_sha = existing
            if existing_content == content:
                return self._latest_commit_sha(path), False
            if not allow_update:
                raise RuntimeError(f"refusing to overwrite immutable public ledger file: {path}")
            payload = {
                "message": message,
                "content": encoded,
                "sha": blob_sha,
                "branch": self.branch,
            }
        else:
            payload = {"message": message, "content": encoded, "branch": self.branch}
        response = self.client.put(
            f"{self.api_base}/contents/{quote(path, safe='/')}", json=payload
        )
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub file publication failed with HTTP {response.status_code}")
        try:
            commit_sha = response.json()["commit"]["sha"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("GitHub publication response omitted its commit SHA") from exc
        if not isinstance(commit_sha, str):
            raise RuntimeError("GitHub publication response contained an invalid commit SHA")
        changed = existing is None or existing_content != content
        return commit_sha, changed

    def _latest_commit_sha(self, path: str) -> str | None:
        response = self.client.get(
            f"{self.api_base}/commits",
            params={"path": path, "per_page": 1, "sha": self.branch},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub commit lookup failed with HTTP {response.status_code}")
        try:
            commits = response.json()
            sha = commits[0]["sha"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("GitHub commit lookup response was malformed") from exc
        if not isinstance(sha, str):
            raise RuntimeError("GitHub commit lookup returned an invalid SHA")
        return sha

    def _verify_public_copy(self, path: str, expected: bytes) -> None:
        url = f"{self.raw_base}/{quote(path, safe='/')}"
        for attempt in range(3):
            response = self.client.get(url)
            if response.status_code == 200 and response.content == expected:
                return
            if response.status_code >= 400 and response.status_code != 404:
                raise RuntimeError(
                    f"public ledger verification failed with HTTP {response.status_code}"
                )
            if attempt < 2:
                self._sleep(1.0)
        raise RuntimeError(f"public ledger copy did not match uploaded bytes: {path}")

    def _create_or_upgrade_proof(self, path: str, content: bytes, proof_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="bwb-ots-") as directory:
            target = Path(directory) / Path(path).name
            target.write_bytes(content)
            sidecar = target.with_name(f"{target.name}.ots")
            existing = self._get_file(proof_path)
            if existing is None:
                self._run_ots("stamp", str(target))
            else:
                sidecar.write_bytes(existing[0])
                if self._verify_proof(content, existing[0]) != "verified":
                    self._run_ots("upgrade", str(sidecar))
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                raise RuntimeError("OpenTimestamps did not produce a non-empty proof sidecar")
            return sidecar.read_bytes()

    def _upgrade_proof(self, content: bytes, filename: str, proof: bytes) -> bytes:
        with tempfile.TemporaryDirectory(prefix="bwb-ots-") as directory:
            target = Path(directory) / filename
            target.write_bytes(content)
            sidecar = target.with_name(f"{target.name}.ots")
            sidecar.write_bytes(proof)
            self._run_ots("upgrade", str(sidecar))
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                raise RuntimeError("OpenTimestamps upgrade removed the proof sidecar")
            return sidecar.read_bytes()

    def _verify_proof(self, content: bytes, proof: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="bwb-ots-") as directory:
            target = Path(directory) / "ledger.json"
            target.write_bytes(content)
            sidecar = target.with_name(f"{target.name}.ots")
            sidecar.write_bytes(proof)
            result = self._run_ots("verify", str(sidecar), check=False)
            output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
            if result.returncode != 0:
                raise RuntimeError("OpenTimestamps proof verification command failed")
            if "success" in output and ("bitcoin" in output or "block" in output):
                return "verified"
            return "pending"

    def _run_ots(self, operation: str, path: str, *, check: bool = True) -> Any:
        try:
            result = self._command_runner(
                [self.command, operation, path],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"OpenTimestamps {operation} command could not complete") from exc
        if check and result.returncode != 0:
            raise RuntimeError(f"OpenTimestamps {operation} command failed")
        return result

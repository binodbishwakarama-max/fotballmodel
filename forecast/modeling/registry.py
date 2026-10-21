"""Model artifact registry with evidence-gated production promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome and evidence for a production model promotion check."""

    promoted: bool
    reason: str
    previous_version: str | None
    model_version: str


def load_production_model(*, model_dir: Path, league: str) -> dict[str, Any]:
    """Load a league's registered model bundle or fail if no model is promoted."""
    registry_path = model_dir / "production.json"
    if not registry_path.is_file():
        raise FileNotFoundError("No production model registry exists")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get("leagues", {}).get(league)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"No production model is registered for {league}")
    artifact = (model_dir / entry["artifact_path"]).resolve()
    if model_dir.resolve() not in artifact.parents:
        raise ValueError("model artifact path must remain inside the models directory")
    if not artifact.is_file():
        raise FileNotFoundError(f"Registered model artifact is missing: {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != entry.get("artifact_sha256"):
        raise ValueError("registered model artifact hash does not match")
    import joblib

    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict) or bundle.get("model_version") != entry.get("model_version"):
        raise ValueError("registered model bundle metadata does not match")
    return bundle


def promote_if_better(
    *,
    model_dir: Path,
    league: str,
    challenger_version: str,
    artifact_path: Path,
    paired_log_loss_difference: float,
    ci_95_low: float,
    ci_95_high: float,
    sample_count: int,
    minimum_samples: int = 100,
    evaluation_report: str,
) -> PromotionResult:
    """Promote only when paired walk-forward evidence proves improvement."""
    if sample_count < minimum_samples:
        return PromotionResult(
            False,
            f"requires at least {minimum_samples} paired matches; observed {sample_count}",
            _current_version(model_dir, league),
            challenger_version,
        )
    if not ci_95_high < 0 or paired_log_loss_difference >= 0:
        return PromotionResult(
            False,
            "paired 95% confidence interval does not establish lower log loss",
            _current_version(model_dir, league),
            challenger_version,
        )
    artifact = artifact_path.resolve()
    models_root = model_dir.resolve()
    if models_root not in artifact.parents:
        raise ValueError("challenger artifact must be inside the models directory")
    if not artifact.is_file():
        raise FileNotFoundError(f"challenger artifact does not exist: {artifact}")
    report = Path(evaluation_report).resolve()
    if not report.is_file():
        raise FileNotFoundError(f"evaluation report does not exist: {report}")

    registry_path = model_dir / "production.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"schema_version": 1, "leagues": {}}
    )
    leagues = registry.setdefault("leagues", {})
    previous = leagues.get(league)
    previous_version = previous.get("model_version") if isinstance(previous, dict) else None
    relative_artifact = artifact.relative_to(models_root).as_posix()
    leagues[league] = {
        "model_version": challenger_version,
        "artifact_path": relative_artifact,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "promoted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_report": report.relative_to(models_root).as_posix()
        if models_root in report.parents
        else str(report),
        "paired_matches": sample_count,
        "paired_log_loss_difference": paired_log_loss_difference,
        "ci_95": [ci_95_low, ci_95_high],
        "promotion_rule": "paired-95ci-below-zero-v1",
    }
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PromotionResult(
        True,
        "paired 95% confidence interval establishes lower log loss",
        previous_version,
        challenger_version,
    )


def _current_version(model_dir: Path, league: str) -> str | None:
    registry_path = model_dir / "production.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get("leagues", {}).get(league, {})
    return entry.get("model_version") if isinstance(entry, dict) else None

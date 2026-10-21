"""Tests for registry artifact integrity and promotion gating."""

import json

import joblib
import pytest

from forecast.modeling.registry import load_production_model, promote_if_better


def test_registry_promotes_only_with_paired_evidence(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = model_dir / "candidate.joblib"
    joblib.dump({"model_version": "candidate-v1", "model": "artifact"}, artifact)
    report = tmp_path / "reports" / "walk-forward.md"
    report.parent.mkdir()
    report.write_text("Paired evaluation summary.\n", encoding="utf-8")

    insufficient = promote_if_better(
        model_dir=model_dir,
        league="PL",
        challenger_version="candidate-v1",
        artifact_path=artifact,
        paired_log_loss_difference=-0.1,
        ci_95_low=-0.2,
        ci_95_high=-0.01,
        sample_count=99,
        evaluation_report=str(report),
    )
    assert not insufficient.promoted
    assert not (model_dir / "production.json").exists()

    result = promote_if_better(
        model_dir=model_dir,
        league="PL",
        challenger_version="candidate-v1",
        artifact_path=artifact,
        paired_log_loss_difference=-0.1,
        ci_95_low=-0.2,
        ci_95_high=-0.01,
        sample_count=100,
        evaluation_report=str(report),
    )
    assert result.promoted
    assert result.previous_version is None
    assert load_production_model(model_dir=model_dir, league="PL")["model_version"] == (
        "candidate-v1"
    )
    registry = json.loads((model_dir / "production.json").read_text(encoding="utf-8"))
    assert registry["leagues"]["PL"]["promotion_rule"] == "paired-95ci-below-zero-v1"


def test_registry_refuses_non_improving_or_tampered_artifacts(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = model_dir / "model.joblib"
    joblib.dump({"model_version": "candidate", "model": "ok"}, artifact)
    report = tmp_path / "report.md"
    report.write_text("result", encoding="utf-8")
    refused = promote_if_better(
        model_dir=model_dir,
        league="PD",
        challenger_version="candidate",
        artifact_path=artifact,
        paired_log_loss_difference=-0.01,
        ci_95_low=-0.1,
        ci_95_high=0.01,
        sample_count=200,
        evaluation_report=str(report),
    )
    assert not refused.promoted

    promote_if_better(
        model_dir=model_dir,
        league="PL",
        challenger_version="candidate",
        artifact_path=artifact,
        paired_log_loss_difference=-0.01,
        ci_95_low=-0.1,
        ci_95_high=-0.001,
        sample_count=200,
        evaluation_report=str(report),
    )
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        load_production_model(model_dir=model_dir, league="PL")

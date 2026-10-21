"""Offline factual-consistency tests for generated explanations."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.llm.explain import (
    evaluate_explanations,
    explain_prediction,
    generate_and_store_explanation,
    validate_explanation,
)
from forecast.models import LedgerBatch, Match, Prediction, PredictionExplanation, Team

FACTS = {
    "probabilities": {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15},
    "home_team": "North FC",
    "away_team": "South FC",
    "allowed_names": ["North FC", "South FC"],
}


def test_explanation_checks_citations_numbers_and_named_entities() -> None:
    failures = validate_explanation(
        explanation="North FC has a 0.6 home probability. South FC has a 0.15 away probability.",
        cited_fact_keys=["home_team", "probabilities.p_home", "probabilities.p_away"],
        facts=FACTS,
    )
    assert failures == []

    failures = validate_explanation(
        explanation="North FC has a 0.7 home probability. Kylian Mbappe may score.",
        cited_fact_keys=["probabilities.missing"],
        facts=FACTS,
    )
    assert any("unsupported number: 0.7" in failure for failure in failures)
    assert any("unsupported named entity: Kylian Mbappe" in failure for failure in failures)
    assert any("cited fact key does not exist" in failure for failure in failures)


def test_explanation_rejects_four_sentences() -> None:
    failures = validate_explanation(
        explanation="One. Two. Three. Four.",
        cited_fact_keys=["home_team"],
        facts=FACTS,
    )
    assert "explanation exceeds three sentences" in failures


def test_anthropic_json_response_is_validated() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='{"explanation":"North FC has a 0.6 home probability.",'
                '"cited_fact_keys":["home_team","probabilities.p_home"]}',
            )
        ]
    )

    class Messages:
        def create(self, **_kwargs):
            return response

    result = explain_prediction(
        facts=FACTS,
        api_key="test",
        client=SimpleNamespace(messages=Messages()),
    )

    assert result["cited_fact_keys"] == ["home_team", "probabilities.p_home"]


def test_invalid_provider_response_fails_loudly() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='{"explanation":"Probability is 0.99.",'
                '"cited_fact_keys":["probabilities.p_home"]}',
            )
        ]
    )

    class Messages:
        def create(self, **_kwargs):
            return response

    with pytest.raises(ValueError, match="unsupported number"):
        explain_prediction(
            facts=FACTS,
            api_key="test",
            client=SimpleNamespace(messages=Messages()),
        )


def test_evaluation_reports_pass_rate_and_examples() -> None:
    report = evaluate_explanations(
        [
            {
                "explanation": "North FC has a 0.6 home probability.",
                "cited_fact_keys": ["home_team", "probabilities.p_home"],
                "facts": FACTS,
            },
            {
                "explanation": "North FC has a 0.9 home probability.",
                "cited_fact_keys": ["home_team", "probabilities.p_home"],
                "facts": FACTS,
            },
        ]
    )
    assert report["count"] == 2
    assert report["passed"] == 1
    assert report["pass_rate"] == 0.5
    assert report["failures"][0]["index"] == 1


def test_explanation_generated_once_per_prediction() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        home = Team(canonical_name="North FC", league="PL")
        away = Team(canonical_name="South FC", league="PL")
        session.add_all((home, away))
        session.flush()
        now = datetime.now(UTC).replace(microsecond=0)
        match = Match(
            id="match",
            league="PL",
            season="2026-27",
            kickoff_utc=now + timedelta(days=2),
            home_team_id=home.id,
            away_team_id=away.id,
            status="scheduled",
            source="test",
        )
        session.add(match)
        session.add(
            LedgerBatch(
                batch_id="batch",
                created_at=now,
                sha256="b" * 64,
                n_predictions=1,
            )
        )
        session.flush()
        session.add(
            Prediction(
                id="prediction",
                match_id=match.id,
                model_version="model",
                forecast_policy_version="t-minus-24h-v1",
                p_home=0.6,
                p_draw=0.25,
                p_away=0.15,
                created_at=now,
                features_hash="a" * 64,
                batch_id="batch",
            )
        )
        session.commit()
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text='{"explanation":"North FC has a 0.6 home probability.",'
                    '"cited_fact_keys":["home_team","probabilities.p_home"]}',
                )
            ]
        )

        class Messages:
            calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                return response

        messages = Messages()
        client = SimpleNamespace(messages=messages)
        first = generate_and_store_explanation(
            session,
            prediction_id="prediction",
            facts=FACTS,
            api_key="test",
            client=client,
        )
        session.commit()
        second = generate_and_store_explanation(
            session,
            prediction_id="prediction",
            facts=FACTS,
            api_key="test",
            client=client,
        )
        assert messages.calls == 1
        assert first.id == second.id
        assert (
            session.scalar(
                select(PredictionExplanation).where(
                    PredictionExplanation.prediction_id == "prediction"
                )
            )
            is not None
        )

"""Structured forecast explanation generation, factual checks, and SHAP drivers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import shap
from anthropic import Anthropic
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.models import Prediction, PredictionExplanation

SYSTEM_PROMPT = """You explain an already-published football forecast.
Use ONLY the supplied facts JSON. Do not invent injuries, lineups, player names,
statistics, match context, or causes. Do not add any number that is not exactly
present in the facts object. Mention only team names in allowed_names.
Return valid JSON with exactly these keys:
{"explanation": "at most three short sentences", "cited_fact_keys": ["existing.key"]}
Every factual claim must be supported by one or more cited keys.
"""
MAX_SENTENCES = 3
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*\b")
ALLOWED_SENTENCE_STARTS = {
    "After",
    "Although",
    "Away",
    "Based",
    "Both",
    "Given",
    "Home",
    "Its",
    "Recent",
    "The",
    "This",
    "With",
}


def top_shap_drivers(
    *,
    model_bundle: Mapping[str, Any],
    feature_values: Mapping[str, float | int | None],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return strongest signed class-contrast SHAP drivers for a model forecast."""
    if limit < 1:
        raise ValueError("limit must be positive")
    model = model_bundle.get("model")
    feature_names = model_bundle.get("features")
    if model is None or not isinstance(feature_names, list):
        raise ValueError("model bundle is missing its estimator or feature list")
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    imputer = model.named_steps.get("imputer") if hasattr(model, "named_steps") else None
    row = pd.DataFrame(
        [[feature_values.get(name) for name in feature_names]], columns=feature_names
    )
    transformed = imputer.transform(row) if imputer is not None else row.to_numpy()
    explainer = shap.TreeExplainer(classifier)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == len(feature_names):
            contrast = values[0, :, 0] - values[0, :, -1]
        elif values.shape[0] == 3 and values.shape[2] == len(feature_names):
            contrast = values[0, 0] - values[2, 0]
        else:
            raise ValueError("unsupported multiclass SHAP output shape")
    elif values.ndim == 2:
        if values.shape[0] == len(feature_names):
            contrast = values[:, 0] - values[:, 2] if values.shape[1] > 1 else values[:, 0]
        elif values.shape[1] == len(feature_names):
            contrast = values[0]
        else:
            raise ValueError("SHAP output does not align with the feature list")
    else:
        raise ValueError("unsupported SHAP output shape")
    ranked = sorted(
        range(len(feature_names)),
        key=lambda index: abs(float(contrast[index])),
        reverse=True,
    )[:limit]
    return [
        {
            "feature": feature_names[index],
            "value": feature_values.get(feature_names[index]),
            "direction": "supports_home" if contrast[index] > 0 else "supports_away",
            "magnitude": float(abs(contrast[index])),
        }
        for index in ranked
    ]


def validate_explanation(
    *,
    explanation: str,
    cited_fact_keys: Sequence[str],
    facts: Mapping[str, Any],
) -> list[str]:
    """Return explicit validation failures for unsupported values or citations."""
    failures: list[str] = []
    if len(_sentences(explanation)) > MAX_SENTENCES:
        failures.append("explanation exceeds three sentences")
    if not explanation.strip():
        failures.append("explanation is empty")
    fact_paths = set(_flatten_fact_paths(facts))
    for key in cited_fact_keys:
        if key not in fact_paths:
            failures.append(f"cited fact key does not exist: {key}")
    for number in NUMBER_PATTERN.findall(explanation):
        if number.endswith("%"):
            failures.append(f"unsupported number unit: {number}")
            continue
        if _normalize_number(number) not in _fact_numbers(facts):
            failures.append(f"unsupported number: {number}")
    allowed_names = set(facts.get("allowed_names", []))
    allowed_name_tokens = {
        token for name in allowed_names for token in name.split() if isinstance(name, str)
    }
    for name in NAME_PATTERN.findall(explanation):
        if name in ALLOWED_SENTENCE_STARTS:
            continue
        if name not in allowed_names and name not in allowed_name_tokens:
            failures.append(f"unsupported named entity: {name}")
    if explanation.strip() and not cited_fact_keys:
        failures.append("explanation cites no fact keys")
    return failures


def explain_prediction(
    *,
    facts: Mapping[str, Any],
    api_key: str,
    model: str = "claude-3-5-haiku-latest",
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate a short JSON explanation, then reject output that fails validation."""
    if not api_key.strip():
        raise ValueError("Anthropic API key is required to generate explanations")
    if not isinstance(facts.get("allowed_names", []), list):
        raise ValueError("facts.allowed_names must be a list")
    anthropic_client = client or Anthropic(api_key=api_key)
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(facts, ensure_ascii=False, sort_keys=True),
            }
        ],
    )
    text_blocks = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("Anthropic response must contain exactly one text block")
    try:
        parsed = json.loads(text_blocks[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Anthropic returned invalid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"explanation", "cited_fact_keys"}:
        raise ValueError("Anthropic response must contain explanation and cited_fact_keys only")
    explanation = parsed["explanation"]
    citations = parsed["cited_fact_keys"]
    if not isinstance(explanation, str) or not isinstance(citations, list):
        raise ValueError("explanation must be text and cited_fact_keys must be a list")
    if not all(isinstance(key, str) for key in citations):
        raise ValueError("cited_fact_keys entries must be strings")
    failures = validate_explanation(
        explanation=explanation,
        cited_fact_keys=citations,
        facts=facts,
    )
    if failures:
        raise ValueError(f"Generated explanation failed factual checks: {'; '.join(failures)}")
    return {"explanation": explanation, "cited_fact_keys": citations}


def generate_and_store_explanation(
    session: Session,
    *,
    prediction_id: str,
    facts: Mapping[str, Any],
    api_key: str,
    model: str = "claude-3-5-haiku-latest",
    client: Any | None = None,
) -> PredictionExplanation:
    """Generate an explanation once per immutable prediction and persist its facts."""
    existing = session.scalar(
        select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
    )
    if existing is not None:
        return existing
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"Unknown prediction: {prediction_id}")
    generated = explain_prediction(
        facts=facts,
        api_key=api_key,
        model=model,
        client=client,
    )
    row = PredictionExplanation(
        prediction_id=prediction_id,
        explanation=generated["explanation"],
        cited_fact_keys=generated["cited_fact_keys"],
        facts_json=dict(facts),
        evaluation_status="validated",
        created_at=datetime.now(UTC),
    )
    session.add(row)
    session.flush()
    return row


def evaluate_explanations(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate saved explanations without prompt-tuning on the evaluation sample."""
    results = []
    for index, example in enumerate(examples):
        failures = validate_explanation(
            explanation=str(example["explanation"]),
            cited_fact_keys=example.get("cited_fact_keys", []),
            facts=example["facts"],
        )
        results.append({"index": index, "passed": not failures, "failures": failures})
    passed = sum(result["passed"] for result in results)
    total = len(results)
    return {
        "count": total,
        "passed": passed,
        "pass_rate": passed / total if total else None,
        "failures": [result for result in results if not result["passed"]],
    }


def _sentences(value: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", value.strip()) if sentence]


def _flatten_fact_paths(
    value: Mapping[str, Any],
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        paths.append(path)
        if isinstance(child, Mapping):
            paths.extend(_flatten_fact_paths(child, path))
    return paths


def _fact_numbers(value: Any) -> set[float]:
    numbers: set[float] = set()
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float)):
        numbers.add(round(float(value), 2))
    elif isinstance(value, Mapping):
        for child in value.values():
            numbers.update(_fact_numbers(child))
    elif isinstance(value, list):
        for child in value:
            numbers.update(_fact_numbers(child))
    return numbers


def _normalize_number(value: str) -> float:
    return round(float(value.rstrip("%")), 2)

"""Leakage-gated walk-forward backtest orchestration and report generation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.db import Engine
from forecast.eval.devig import devig
from forecast.eval.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
    reliability_curve_data,
)
from forecast.eval.walk_forward import paired_matchweek_bootstrap
from forecast.features.build import build_features
from forecast.modeling.elo import EloBaseline
from forecast.modeling.training import train_walk_forward
from forecast.models import Match, Odds

logger = logging.getLogger(__name__)
MINIMUM_TRAINING_SEASONS = 3
FORECAST_HORIZON = timedelta(hours=24)


def run_backtest(
    engine: Engine,
    *,
    model_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    optuna_trials: int = 20,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Run an expanding walk-forward evaluation only when source timing permits it."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "backtest.md"
    with Session(engine) as session:
        matches = session.scalars(
            select(Match)
            .where(
                Match.status == "finished",
                Match.result.in_(("H", "D", "A")),
                Match.home_goals.is_not(None),
                Match.away_goals.is_not(None),
            )
            .order_by(Match.league, Match.kickoff_utc)
        ).all()
        eligible = _eligible_fold_count(matches)
        if eligible == 0:
            message = _availability_blocker(matches)
            report_path.write_text(
                "# Walk-forward backtest\n\n"
                "## Status: not run\n\n"
                f"{message}\n\n"
                "No backtest metric or claim is produced. Historical source retrieval "
                "timestamps are not moved backward to manufacture a leakage-safe record.\n",
                encoding="utf-8",
            )
            return {
                "status": "blocked",
                "eligible_folds": 0,
                "report": str(report_path),
                "reason": message,
            }

        feature_rows: list[dict[str, Any]] = []
        labels: list[int] = []
        leagues: list[str] = []
        seasons: list[str] = []
        kickoffs: list[datetime] = []
        forecast_cutoffs: list[datetime] = []
        label_available_at: list[datetime | None] = []
        included_matches: list[Match] = []
        for match in matches:
            if match.result_available_at is None or _utc(match.result_available_at) < _utc(
                match.kickoff_utc
            ):
                continue
            kickoff = _utc(match.kickoff_utc)
            forecast_cutoff = kickoff - FORECAST_HORIZON
            features = build_features(
                session,
                match_id=match.id,
                as_of_kickoff=forecast_cutoff,
            )
            feature_rows.append(features.values)
            labels.append({"H": 0, "D": 1, "A": 2}[match.result])
            leagues.append(match.league)
            seasons.append(match.season)
            kickoffs.append(kickoff)
            forecast_cutoffs.append(forecast_cutoff)
            label_available_at.append(_utc(match.result_available_at))
            included_matches.append(match)

        predictions, artifacts = train_walk_forward(
            pd.DataFrame.from_records(feature_rows),
            labels,
            leagues=leagues,
            seasons=seasons,
            model_dir=model_dir,
            seed=seed,
            optuna_trials=optuna_trials,
            min_training_seasons=MINIMUM_TRAINING_SEASONS,
            kickoffs=kickoffs,
            forecast_cutoffs=forecast_cutoffs,
            label_available_at=label_available_at,
        )
        predictions = _add_elo_and_market(
            session,
            predictions,
            included_matches,
            seed=seed,
        )
        ablations = _run_feature_ablations(
            pd.DataFrame.from_records(feature_rows),
            labels,
            leagues=leagues,
            seasons=seasons,
            kickoffs=kickoffs,
            forecast_cutoffs=forecast_cutoffs,
            label_available_at=label_available_at,
            main_predictions=predictions,
            model_dir=reports_dir / "ablation-models",
            seed=seed,
            optuna_trials=optuna_trials,
        )
        calibration_path = _write_calibration_svg(predictions, reports_dir)
        report = _render_report(
            predictions,
            reports_dir=reports_dir,
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            ablations=ablations,
        )
        report += "\nCalibration diagram: [calibration.svg](calibration.svg)\n"
        report_path.write_text(report, encoding="utf-8")
        return {
            "status": "complete",
            "eligible_folds": eligible,
            "predictions": len(predictions),
            "artifacts": [str(item.model_path) for item in artifacts],
            "report": str(report_path),
            "calibration_plot": str(calibration_path),
        }


def _eligible_fold_count(matches: list[Match]) -> int:
    folds = 0
    leagues = sorted({match.league for match in matches})
    for league in leagues:
        league_matches = [match for match in matches if match.league == league]
        seasons = sorted({match.season for match in league_matches})
        for test_season in seasons[MINIMUM_TRAINING_SEASONS:]:
            test_matches = [match for match in league_matches if match.season == test_season]
            training = [
                match
                for match in league_matches
                if match.season < test_season
                and match.result_available_at is not None
                and _utc(match.result_available_at) >= _utc(match.kickoff_utc)
                and _utc(match.result_available_at)
                < min(_utc(item.kickoff_utc) - FORECAST_HORIZON for item in test_matches)
            ]
            if len({match.season for match in training}) >= MINIMUM_TRAINING_SEASONS:
                folds += 1
    return folds


def _availability_blocker(matches: list[Match]) -> str:
    if not matches:
        return (
            "No settled matches are available. Ingest historical results and begin "
            "recording source availability timestamps."
        )
    dated = [
        (_utc(match.kickoff_utc), _utc(match.result_available_at))
        for match in matches
        if match.result_available_at is not None
    ]
    if not dated:
        return (
            "Settled historical matches do not have system-observed result timestamps, "
            "so no training outcome can be proven available before a later test season."
        )
    earliest_observed = min(available for _, available in dated)
    return (
        "No season fold has at least three prior training seasons whose outcomes were "
        "observed before the test-season kickoff. The earliest result observation is "
        f"{earliest_observed.isoformat()}; begin accumulating time-stamped results "
        "across seasons before reporting production-equivalent performance."
    )


def _add_elo_and_market(
    session: Session,
    predictions: pd.DataFrame,
    matches: list[Match],
    *,
    seed: int,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    prediction_rows = predictions.to_dict(orient="records")
    fold_rows = predictions[predictions["forecaster"] == "lightgbm_calibrated"]
    all_matches = session.scalars(select(Match).where(Match.status == "finished")).all()
    matches_by_league: dict[str, list[Match]] = {}
    for match in all_matches:
        matches_by_league.setdefault(match.league, []).append(match)
    extras: dict[str, dict[str, Any]] = {}
    elo_records: list[dict[str, Any]] = []
    for (league, season), group in fold_rows.groupby(["league", "season"]):
        fixtures = [(row, matches[int(row["index"])]) for row in group.to_dict(orient="records")]
        fixtures.sort(key=lambda item: _utc(item[1].kickoff_utc))
        first_cutoff = _utc(fixtures[0][1].kickoff_utc) - FORECAST_HORIZON
        league_matches = matches_by_league.get(league, [])
        historical = [_elo_match_record(match) for match in league_matches]
        try:
            elo = EloBaseline(random_seed=seed).fit(historical, through=first_cutoff)
        except ValueError as exc:
            logger.warning("Elo baseline unavailable for %s %s: %s", league, season, exc)
            elo = None

        processed = {
            item["id"]
            for item in historical
            if item["result_available_at"] is not None
            and _utc(item["result_available_at"]) < first_cutoff
            and _utc(item["kickoff_utc"]) < first_cutoff
        }
        pending = sorted(
            (
                item
                for item in historical
                if item["result_available_at"] is not None
                and item["home_goals"] is not None
                and item["away_goals"] is not None
                and item["id"] not in processed
            ),
            key=lambda item: (
                _utc(item["result_available_at"]),
                _utc(item["kickoff_utc"]),
                item["id"],
            ),
        )
        event_index = 0
        for row, match in fixtures:
            cutoff = _utc(match.kickoff_utc) - FORECAST_HORIZON
            while (
                event_index < len(pending)
                and _utc(pending[event_index]["result_available_at"]) < cutoff
            ):
                event = pending[event_index]
                event_index += 1
                if _utc(event["kickoff_utc"]) < cutoff and elo is not None:
                    elo.update(
                        str(event["home_team_id"]),
                        str(event["away_team_id"]),
                        int(event["home_goals"]),
                        int(event["away_goals"]),
                        season=str(event["season"]),
                    )
            elo_probs: list[float] | None = None
            if elo is not None:
                try:
                    elo_probs = elo.predict_proba(
                        str(match.home_team_id),
                        str(match.away_team_id),
                        season=season,
                    )
                except ValueError as exc:
                    logger.warning("Elo unavailable for fixture %s: %s", match.id, exc)
            market_probs = _closing_market(session, match)
            extras[match.id] = {
                "elo": elo_probs,
                "market": market_probs,
                "kickoff": _utc(match.kickoff_utc),
                "matchweek": match.matchweek or _utc(match.kickoff_utc).isocalendar().week,
            }
            if elo_probs is not None:
                elo_records.append(
                    {
                        "index": int(row["index"]),
                        "league": league,
                        "season": season,
                        "y_true": int(row["y_true"]),
                        "forecaster": "elo",
                        "model_version": "elo-online-v1",
                        "p_home": elo_probs[0],
                        "p_draw": elo_probs[1],
                        "p_away": elo_probs[2],
                        "match_id": match.id,
                        "kickoff_utc": _utc(match.kickoff_utc),
                        "matchweek": extras[match.id]["matchweek"],
                    }
                )
    records = []
    for row in prediction_rows:
        match = matches[int(row["index"])]
        extra = extras[match.id]
        elo_probs = extra["elo"] or [None, None, None]
        market_probs = extra["market"] or [None, None, None]
        row["p_elo_home"], row["p_elo_draw"], row["p_elo_away"] = elo_probs
        row["p_market_home"], row["p_market_draw"], row["p_market_away"] = market_probs
        row["match_id"] = match.id
        row["kickoff_utc"] = extra["kickoff"]
        row["matchweek"] = extra["matchweek"]
        records.append(row)
    return pd.concat(
        [pd.DataFrame.from_records(records), pd.DataFrame.from_records(elo_records)],
        ignore_index=True,
    )


def _elo_match_record(match: Match) -> dict[str, Any]:
    return {
        "id": match.id,
        "season": match.season,
        "kickoff_utc": match.kickoff_utc,
        "result_available_at": match.result_available_at,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
    }


def _closing_market(session: Session, match: Match) -> list[float] | None:
    rows = session.scalars(
        select(Odds)
        .where(Odds.match_id == match.id, Odds.kind == "close")
        .order_by(Odds.retrieved_at.desc())
    ).all()
    for preferred in ("Pinnacle", "Bet365"):
        for odds in rows:
            if odds.source == preferred:
                return devig([odds.home, odds.draw, odds.away], method="power").tolist()
    return None


def _render_report(
    predictions: pd.DataFrame,
    *,
    reports_dir: Path,
    seed: int,
    bootstrap_replicates: int,
    ablations: list[dict[str, Any]] | None = None,
) -> str:
    if predictions.empty:
        raise ValueError("walk-forward evaluator returned no predictions")
    output = [
        "# Walk-forward backtest",
        "",
        "This report uses chronological, expanding season folds and reports only "
        "time-eligible outcomes. No random train/test split is used.",
        "",
        f"Bootstrap seed: {seed}; replicates: {bootstrap_replicates}.",
        "Intervals are the 2.5th and 97.5th percentiles from paired resampling "
        "of matchweeks within each league-season.",
        "",
        "## Scores",
        "",
        (
            "| League | Season | Forecaster | Matches | Log loss | Brier | RPS | "
            "ECE H | ECE D | ECE A | ECE macro |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    probability_columns = ("p_home", "p_draw", "p_away")
    score_groups = [
        (league, season, forecaster, group)
        for (league, season, forecaster), group in predictions.groupby(
            ["league", "season", "forecaster"], dropna=False
        )
    ]
    score_groups.extend(
        ("ALL", "ALL", forecaster, group)
        for forecaster, group in predictions.groupby("forecaster", dropna=False)
    )
    for league, season, forecaster, group in score_groups:
        labels = group["y_true"].to_numpy(dtype=np.int64)
        probabilities = group.loc[:, probability_columns].to_numpy(dtype=np.float64)
        ece = expected_calibration_error(labels, probabilities)
        output.append(
            f"| {league} | {season} | {forecaster} | {len(group)} | "
            f"{multiclass_log_loss(labels, probabilities):.4f} | "
            f"{multiclass_brier_score(labels, probabilities):.4f} | "
            f"{ranked_probability_score(labels, probabilities):.4f} | "
            f"{ece['H']:.4f} | {ece['D']:.4f} | {ece['A']:.4f} | {ece['macro']:.4f} |"
        )
    output.extend(
        [
            "",
            "## Reliability-bin counts",
            "",
            (
                "| League | Season | Forecaster | Outcome | Bin | Count | "
                "Mean confidence | Observed frequency |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (league, season, forecaster), group in predictions.groupby(
        ["league", "season", "forecaster"], dropna=False
    ):
        labels = group["y_true"].to_numpy(dtype=np.int64)
        probabilities = group.loc[:, probability_columns].to_numpy(dtype=np.float64)
        reliability = reliability_curve_data(labels, probabilities, bins=10)
        for outcome, bins in reliability.items():
            for bin_index, item in enumerate(bins):
                confidence = (
                    f"{item.mean_confidence:.4f}" if item.mean_confidence is not None else ""
                )
                observed = (
                    f"{item.observed_frequency:.4f}" if item.observed_frequency is not None else ""
                )
                output.append(
                    f"| {league} | {season} | {forecaster} | {outcome} | {bin_index} | "
                    f"{item.count} | {confidence} | {observed} |"
                )
    output.extend(
        [
            "",
            "## Error analysis by actual result",
            "",
            (
                "| League | Season | Forecaster | Actual result | Matches | "
                "Mean assigned probability | Mean log loss | Incorrect at >=80% confidence |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    outcome_names = ("H", "D", "A")
    for (league, season, forecaster, label), group in predictions.groupby(
        ["league", "season", "forecaster", "y_true"], dropna=False
    ):
        probabilities = group.loc[:, probability_columns].to_numpy(dtype=np.float64)
        actual_probabilities = probabilities[:, int(label)]
        maximum = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        incorrect_high_confidence = int(((predicted != int(label)) & (maximum >= 0.8)).sum())
        output.append(
            f"| {league} | {season} | {forecaster} | {outcome_names[int(label)]} | "
            f"{len(group)} | {actual_probabilities.mean():.4f} | "
            f"{-np.log(np.maximum(1e-15, actual_probabilities)).mean():.4f} | "
            f"{incorrect_high_confidence} |"
        )
    model_name = "lightgbm_calibrated"
    output.extend(
        [
            "",
            "## Closing-market coverage",
            "",
            "| League | Season | Model matches | Closing-market matches | Coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    calibrated = predictions[predictions["forecaster"] == model_name]
    for (league, season), group in calibrated.groupby(["league", "season"]):
        covered = group.dropna(subset=["p_market_home", "p_market_draw", "p_market_away"])[
            "match_id"
        ].nunique()
        count = group["match_id"].nunique()
        output.append(f"| {league} | {season} | {count} | {covered} | {covered / count:.1%} |")
    total_matches = calibrated["match_id"].nunique()
    total_covered = calibrated.dropna(subset=["p_market_home", "p_market_draw", "p_market_away"])[
        "match_id"
    ].nunique()
    if total_matches:
        output.append(
            f"| ALL | ALL | {total_matches} | {total_covered} | "
            f"{total_covered / total_matches:.1%} |"
        )
    output.extend(
        [
            "",
        ]
    )
    for baseline, baseline_name, baseline_columns in (
        ("market", None, ("p_market_home", "p_market_draw", "p_market_away")),
        ("Elo", "elo", probability_columns),
        ("logistic", "logistic", probability_columns),
    ):
        output.extend(
            [
                "",
                f"## Paired model-versus-{baseline} uncertainty",
                "",
                "| League | Season | Matches | Model − baseline log loss | Paired 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        comparison_rows = _paired_baseline_rows(
            predictions,
            baseline_name=baseline_name,
            baseline_columns=baseline_columns,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        if comparison_rows:
            output.extend(
                f"| {league} | {season} | {count} | {difference:.4f} | [{lower:.4f}, {upper:.4f}] |"
                for league, season, count, difference, lower, upper in comparison_rows
            )
        else:
            output.append("| No paired eligible fixtures | | | | |")
    output.extend(
        [
            "",
            "## Feature-group ablations",
            "",
            "Each ablation removes one feature group and reruns the same expanding "
            "walk-forward training and calibration. A positive log-loss delta means "
            "the removed group helped the full model.",
            "",
            (
                "| League | Season | Removed feature group | Matches | Full log loss | "
                "Ablated log loss | Ablated − full |"
            ),
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if ablations:
        output.extend(
            f"| {item['league']} | {item['season']} | {item['group']} | "
            f"{item['matches']} | {item['full_log_loss']:.4f} | "
            f"{item['ablated_log_loss']:.4f} | {item['delta']:.4f} |"
            for item in ablations
        )
    else:
        output.append("| No ablation folds available | | | | | | |")
    output.extend(
        [
            "",
            "## Limitations",
            "",
            "Missing closing prices and unavailable source timestamps are never filled "
            "from opening odds or future data.",
            "",
            "A historical close from football-data.co.uk without a verified "
            "observation timestamp is a retrospective benchmark, not an executable "
            "real-time price.",
            "",
        ]
    )
    return "\n".join(output)


def _run_feature_ablations(
    features: pd.DataFrame,
    labels: list[int],
    *,
    leagues: list[str],
    seasons: list[str],
    kickoffs: list[datetime],
    forecast_cutoffs: list[datetime],
    label_available_at: list[datetime | None],
    main_predictions: pd.DataFrame,
    model_dir: Path,
    seed: int,
    optuna_trials: int,
) -> list[dict[str, Any]]:
    groups = {
        "xg": [column for column in features if "_xg_" in column],
        "form": [
            column
            for column in features
            if any(
                marker in column
                for marker in (
                    "_points_per_game_",
                    "_goals_for_",
                    "_goals_against_",
                    "_goal_difference_",
                )
                or ("_matches_" in column and not column.endswith("_matches_last_14_days"))
            )
        ],
        "rest": [
            column
            for column in features
            if column.endswith("_rest_days") or column.endswith("_matches_last_14_days")
        ],
        "head_to_head": [column for column in features if "h2h_" in column],
        "elo": [
            column for column in features if column in {"home_elo", "away_elo", "elo_difference"}
        ],
    }
    baseline = main_predictions[main_predictions["forecaster"] == "lightgbm_calibrated"]
    results: list[dict[str, Any]] = []
    for group_name, excluded in groups.items():
        if not excluded:
            continue
        remaining = features.drop(columns=excluded)
        ablated, _ = train_walk_forward(
            remaining,
            labels,
            leagues=leagues,
            seasons=seasons,
            model_dir=model_dir / group_name,
            seed=seed,
            optuna_trials=optuna_trials,
            min_training_seasons=MINIMUM_TRAINING_SEASONS,
            kickoffs=kickoffs,
            forecast_cutoffs=forecast_cutoffs,
            label_available_at=label_available_at,
        )
        ablated = ablated[ablated["forecaster"] == "lightgbm_calibrated"]
        for (league, season), group in ablated.groupby(["league", "season"]):
            full_group = baseline[(baseline["league"] == league) & (baseline["season"] == season)]
            full_by_index = full_group.drop_duplicates("index").set_index("index")
            ablated_by_index = group.drop_duplicates("index").set_index("index")
            common = full_by_index.index.intersection(ablated_by_index.index)
            if common.empty:
                continue
            actual = full_by_index.loc[common, "y_true"].to_numpy(dtype=np.int64)
            full_probabilities = full_by_index.loc[common, ["p_home", "p_draw", "p_away"]].to_numpy(
                dtype=np.float64
            )
            ablated_probabilities = ablated_by_index.loc[
                common, ["p_home", "p_draw", "p_away"]
            ].to_numpy(dtype=np.float64)
            full_losses = -np.log(
                np.maximum(1e-15, full_probabilities[np.arange(len(actual)), actual])
            )
            ablated_losses = -np.log(
                np.maximum(1e-15, ablated_probabilities[np.arange(len(actual)), actual])
            )
            full_loss = float(full_losses.mean())
            ablated_loss = float(ablated_losses.mean())
            results.append(
                {
                    "league": league,
                    "season": season,
                    "group": group_name,
                    "matches": len(common),
                    "full_log_loss": full_loss,
                    "ablated_log_loss": ablated_loss,
                    "delta": ablated_loss - full_loss,
                }
            )
    return results


def _write_calibration_svg(predictions: pd.DataFrame, reports_dir: Path) -> Path:
    group = predictions[predictions["forecaster"] == "lightgbm_calibrated"]
    if group.empty:
        raise ValueError("calibration plot requires calibrated model predictions")
    labels = group["y_true"].to_numpy(dtype=np.int64)
    probabilities = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)
    curves = reliability_curve_data(labels, probabilities, bins=10)
    x0, y0, width, height = 70, 30, 500, 310
    colors = {"H": "#2563eb", "D": "#16a34a", "A": "#dc2626"}
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="410" viewBox="0 0 640 410">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0}" '
            'stroke="#9ca3af" stroke-dasharray="5 5"/>'
        ),
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#111827"/>',
        (
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" '
            f'y2="{y0 + height}" stroke="#111827"/>'
        ),
        (
            f'<text x="{x0 + width / 2}" y="390" text-anchor="middle" '
            'font-family="sans-serif">Mean predicted probability</text>'
        ),
        (
            '<text x="18" y="190" transform="rotate(-90 18 190)" '
            'text-anchor="middle" font-family="sans-serif">Observed frequency</text>'
        ),
    ]
    for outcome, bins in curves.items():
        points = [
            (
                x0 + item.mean_confidence * width,
                y0 + (1 - item.observed_frequency) * height,
            )
            for item in bins
            if item.count
            and item.mean_confidence is not None
            and item.observed_frequency is not None
        ]
        if points:
            coordinate_list = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            elements.append(
                f'<polyline points="{coordinate_list}" fill="none" '
                f'stroke="{colors[outcome]}" stroke-width="2"/>'
            )
            elements.append(
                f'<text x="{x0 + width - 65}" y="{y0 + 20 + 20 * ("HDA".index(outcome))}" '
                f'font-family="sans-serif" fill="{colors[outcome]}">{outcome}</text>'
            )
            elements.extend(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[outcome]}"/>'
                for x, y in points
            )
    elements.append("</svg>")
    path = reports_dir / "calibration.svg"
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return path


def _paired_baseline_rows(
    predictions: pd.DataFrame,
    *,
    baseline_name: str | None,
    baseline_columns: tuple[str, str, str],
    bootstrap_replicates: int,
    seed: int,
) -> list[tuple[str, str, int, float, float, float]]:
    model_name = "lightgbm_calibrated"
    model_rows = predictions[predictions["forecaster"] == model_name]
    if baseline_name is None:
        baseline_rows = model_rows.dropna(subset=list(baseline_columns))
    else:
        baseline_rows = predictions[predictions["forecaster"] == baseline_name]
    if model_rows.empty or baseline_rows.empty:
        return []

    groups = [
        (league, season)
        for league, season in model_rows[["league", "season"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]
    results: list[tuple[str, str, int, float, float, float]] = []

    def compare(
        league: str,
        season: str,
        model_group: pd.DataFrame,
        baseline_group: pd.DataFrame,
    ) -> None:
        model_by_id = model_group.drop_duplicates("match_id").set_index("match_id")
        baseline_by_id = baseline_group.drop_duplicates("match_id").set_index("match_id")
        common_ids = model_by_id.index.intersection(baseline_by_id.index)
        if common_ids.empty:
            return
        labels = model_by_id.loc[common_ids, "y_true"].to_numpy(dtype=np.int64)
        model_probabilities = model_by_id.loc[common_ids, ["p_home", "p_draw", "p_away"]].to_numpy(
            dtype=np.float64
        )
        baseline_probabilities = baseline_by_id.loc[common_ids, list(baseline_columns)].to_numpy(
            dtype=np.float64
        )
        model_losses = -np.log(
            np.maximum(1e-15, model_probabilities[np.arange(len(labels)), labels])
        )
        baseline_losses = -np.log(
            np.maximum(1e-15, baseline_probabilities[np.arange(len(labels)), labels])
        )
        rows = model_by_id.loc[common_ids]
        difference, lower, upper = paired_matchweek_bootstrap(
            model_losses,
            baseline_losses,
            leagues=rows["league"].astype(str).tolist(),
            seasons=rows["season"].astype(str).tolist(),
            matchweeks=rows["matchweek"].astype(int).tolist(),
            replicates=bootstrap_replicates,
            seed=seed,
        )
        results.append((league, season, len(common_ids), difference, lower, upper))

    for league, season in groups:
        model_group = model_rows[
            (model_rows["league"] == league) & (model_rows["season"] == season)
        ]
        baseline_group = baseline_rows[
            (baseline_rows["league"] == league) & (baseline_rows["season"] == season)
        ]
        compare(league, season, model_group, baseline_group)
    compare("ALL", "ALL", model_rows, baseline_rows)
    return results


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

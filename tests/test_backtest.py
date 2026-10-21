from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
from sqlalchemy.orm import Session

from forecast.db import create_schema, get_engine
from forecast.eval.backtest import (
    FORECAST_HORIZON,
    _eligible_fold_count,
    _render_report,
    _write_calibration_svg,
    run_backtest,
)
from forecast.models import Match, Team


def test_backtest_writes_explicit_report_when_no_results_exist(tmp_path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    report_dir = tmp_path / "reports"

    result = run_backtest(
        engine,
        model_dir=tmp_path / "models",
        reports_dir=report_dir,
    )

    report = (report_dir / "backtest.md").read_text(encoding="utf-8")
    assert result["status"] == "blocked"
    assert "No settled matches are available" in result["reason"]
    assert "Status: not run" in report
    assert "No backtest metric or claim is produced" in report


def test_fold_requires_labels_observed_before_the_forecast_cutoff() -> None:
    first_kickoff = datetime(2023, 8, 12, 15, tzinfo=UTC)
    matches = []
    for season_number in range(2020, 2024):
        kickoff = datetime(season_number, 8, 12, 15, tzinfo=UTC)
        available_at = (
            first_kickoff - FORECAST_HORIZON - timedelta(hours=1)
            if season_number < 2023
            else kickoff + timedelta(hours=3)
        )
        matches.append(
            SimpleNamespace(
                league="PL",
                season=f"{season_number}-{(season_number + 1) % 100:02d}",
                kickoff_utc=kickoff,
                result_available_at=available_at,
            )
        )

    assert _eligible_fold_count(matches) == 1

    matches[2].result_available_at = first_kickoff - FORECAST_HORIZON + timedelta(minutes=1)
    assert _eligible_fold_count(matches) == 0


def test_report_includes_pooled_scores_reliability_and_market_coverage(tmp_path) -> None:
    rows = []
    for match_id, label, matchweek in (("m1", 0, 1), ("m2", 2, 2)):
        for forecaster in ("lightgbm_calibrated", "logistic", "elo"):
            rows.append(
                {
                    "match_id": match_id,
                    "league": "PL",
                    "season": "2023-24",
                    "matchweek": matchweek,
                    "y_true": label,
                    "forecaster": forecaster,
                    "p_home": 0.7 if label == 0 else 0.2,
                    "p_draw": 0.2,
                    "p_away": 0.1 if label == 0 else 0.6,
                    "p_market_home": 0.6,
                    "p_market_draw": 0.25,
                    "p_market_away": 0.15,
                }
            )

    report = _render_report(
        pd.DataFrame(rows),
        reports_dir=tmp_path,
        seed=7,
        bootstrap_replicates=20,
    )

    assert "| ALL | ALL | lightgbm_calibrated | 2 |" in report
    assert "## Reliability-bin counts" in report
    assert "## Closing-market coverage" in report
    assert "## Paired model-versus-market uncertainty" in report
    assert "## Paired model-versus-Elo uncertainty" in report
    assert "## Paired model-versus-logistic uncertainty" in report
    assert "| ALL | ALL | 2 |" in report


def test_calibration_plot_is_written_as_svg(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "forecaster": ["lightgbm_calibrated"] * 3,
            "y_true": [0, 1, 2],
            "p_home": [0.8, 0.2, 0.1],
            "p_draw": [0.1, 0.7, 0.2],
            "p_away": [0.1, 0.1, 0.7],
        }
    )

    path = _write_calibration_svg(predictions, tmp_path)

    assert path.name == "calibration.svg"
    assert "<svg" in path.read_text(encoding="utf-8")
    assert "<polyline" in path.read_text(encoding="utf-8")


def test_backtest_runs_eligible_fold_and_generates_ablation_report(tmp_path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        home = Team(canonical_name="Home", league="PL")
        away = Team(canonical_name="Away", league="PL")
        session.add_all([home, away])
        session.flush()
        results = ((2, 0), (0, 0), (0, 1), (3, 1), (1, 1), (0, 2))
        for season_year in range(2020, 2024):
            season = f"{season_year}-{(season_year + 1) % 100:02d}"
            for index, (home_goals, away_goals) in enumerate(results):
                kickoff = datetime(season_year, 8, 1, 15, tzinfo=UTC) + timedelta(days=7 * index)
                session.add(
                    Match(
                        id=f"{season}-{index}",
                        league="PL",
                        season=season,
                        kickoff_utc=kickoff,
                        kickoff_time_known=True,
                        matchweek=index + 1,
                        home_team_id=home.id,
                        away_team_id=away.id,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        result="H"
                        if home_goals > away_goals
                        else "A"
                        if home_goals < away_goals
                        else "D",
                        result_available_at=kickoff + timedelta(hours=3),
                        status="finished",
                        source="test",
                    )
                )
        session.commit()

    result = run_backtest(
        engine,
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        optuna_trials=1,
        bootstrap_replicates=20,
    )

    report = (tmp_path / "reports" / "backtest.md").read_text(encoding="utf-8")
    assert result["status"] == "complete"
    assert result["predictions"] > 0
    assert (tmp_path / "reports" / "calibration.svg").exists()
    assert "Feature-group ablations" in report
    assert "head_to_head" in report

"""Tests for command-line behavior."""

from sqlalchemy import inspect

from forecast.cli import build_parser, main
from forecast.db import get_engine


def test_status_command_reports_project_phase(capsys) -> None:
    assert main(["status"]) == 0
    assert "commands are available" in capsys.readouterr().out


def test_predict_command_runs_without_upcoming_fixtures(capsys) -> None:
    assert main(["predict", "--database-url", "sqlite:///:memory:", "--dry-run"]) == 0
    assert '"predictions_created": 0' in capsys.readouterr().out


def test_backtest_and_xg_sync_are_registered_commands() -> None:
    assert build_parser().parse_args(["backtest"]).command == "backtest"
    args = build_parser().parse_args(["sync-xg", "--league", "PL", "--season", "2025-26"])
    assert (args.command, args.league, args.season) == ("sync-xg", "PL", "2025-26")
    assert build_parser().parse_args(["telegram-bot"]).command == "telegram-bot"
    assert build_parser().parse_args(["scheduler"]).command == "scheduler"


def test_migrate_applies_all_revisions(tmp_path, capsys) -> None:
    database_url = f"sqlite:///{tmp_path / 'production.db'}"

    assert main(["migrate", "--database-url", database_url]) == 0
    assert "migrations are current" in capsys.readouterr().out
    columns = {
        column["name"] for column in inspect(get_engine(database_url)).get_columns("ledger_batches")
    }
    assert "timestamp_proof_commit" in columns

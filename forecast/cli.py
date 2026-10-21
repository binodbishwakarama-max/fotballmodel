"""Command-line entry point for BWB Forecast."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from forecast.config import PROJECT_ROOT, Settings, normalize_database_url
from forecast.db import create_schema, get_engine
from forecast.eval.backtest import run_backtest
from forecast.ingestion.historical import ingest_historical
from forecast.ingestion.understat import UnderstatClient, ingest_understat_xg
from forecast.jobs import predict_upcoming, refresh_scorecard, settle_results, sync_live_fixtures
from forecast.logging_config import configure_logging
from forecast.telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(prog="bwb", description="BWB Forecast tools")
    parser.add_argument(
        "command",
        choices=sorted(
            {
                "status",
                "init-db",
                "migrate",
                "ingest",
                "backtest",
                "sync-xg",
                "sync-live",
                "settle",
                "predict",
                "scorecard",
                "serve",
                "telegram-bot",
                "scheduler",
                "publish-ledger",
                "verify-proofs",
            }
        ),
        help="operation to run",
    )
    parser.add_argument("--league", choices=("PL", "PD"), help="league to ingest")
    parser.add_argument("--season", help="season in YYYY-YY form, such as 2025-26")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the requested historical season(s) before ingestion",
    )
    parser.add_argument("--database-url", help="override DATABASE_URL for this command")
    parser.add_argument("--days-back", type=int, default=14, help="settlement/lookback days")
    parser.add_argument("--days-ahead", type=int, default=30, help="live fixture lookahead days")
    parser.add_argument("--date-from", type=date.fromisoformat, help="live sync start date")
    parser.add_argument("--date-to", type=date.fromisoformat, help="live sync end date")
    parser.add_argument(
        "--window-hours",
        type=int,
        default=72,
        help="prediction scan horizon; canonical window remains 24-30 hours",
    )
    parser.add_argument("--dry-run", action="store_true", help="preview predictions without saving")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--host", default="127.0.0.1", help="API bind address")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return a process exit code."""
    configure_logging()
    args = build_parser().parse_args(argv)

    if args.command == "status":
        print(
            "BWB Forecast ingestion, evaluation, prediction, scorecard, "
            "and API commands are available."
        )
        return 0
    settings = Settings.from_env()
    url = args.database_url or settings.database_url or "sqlite:///data/bwb_forecast.db"
    engine = get_engine(url)
    if args.command in {"ingest", "init-db"}:
        create_schema(engine)
        if args.command == "init-db":
            print(f"Database schema ready: {url}")
            return 0
        try:
            result = ingest_historical(
                engine=engine,
                data_dir=settings.data_dir,
                leagues=(args.league,) if args.league else ("PL", "PD"),
                seasons=(args.season,) if args.season else None,
                download=args.download,
            )
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        print(result.summary())
        return 0
    if args.command == "backtest":
        create_schema(engine)
        result = run_backtest(
            engine,
            model_dir=settings.models_dir,
            reports_dir=settings.reports_dir / "generated",
            seed=settings.random_seed,
        )
        print(f"Backtest status: {result['status']}")
        print(f"Report: {result['report']}")
        if result["status"] == "blocked":
            print(f"Reason: {result['reason']}")
            return 1
        print(f"Predictions: {result['predictions']}")
        return 0
    if args.command == "migrate":
        from alembic import command
        from alembic.config import Config

        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        if args.database_url:
            config.set_main_option(
                "sqlalchemy.url", normalize_database_url(args.database_url).replace("%", "%%")
            )
        elif settings.database_url:
            config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
        else:
            logger.error("DATABASE_URL is required for migrations.")
            return 2
        command.upgrade(config, "head")
        print("Database migrations are current.")
        return 0
    if args.command in {
        "sync-live",
        "settle",
        "predict",
        "scorecard",
        "serve",
        "telegram-bot",
        "scheduler",
        "publish-ledger",
        "verify-proofs",
    }:
        if args.command not in {"serve", "scheduler", "publish-ledger", "verify-proofs"}:
            create_schema(engine)
        if args.command in {"scheduler", "publish-ledger", "verify-proofs"}:
            if settings.database_url is None and args.database_url is None:
                logger.error("DATABASE_URL is required for production operations.")
                return 2
            from forecast.ledger.publisher import GitHubLedgerPublisher

            if args.command == "scheduler":
                from forecast.scheduler import run_scheduler

                run_scheduler(engine, settings)
                return 0
            publisher = GitHubLedgerPublisher(settings)
            try:
                result = (
                    publisher.publish_pending(engine)
                    if args.command == "publish-ledger"
                    else publisher.upgrade_pending_proofs(engine)
                )
            finally:
                publisher.close()
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "sync-live":
            if args.days_back < 0 or args.days_ahead < 0:
                logger.error("--days-back and --days-ahead must be non-negative.")
                return 2
            current = datetime.now(UTC)
            date_from = args.date_from or (current - timedelta(days=args.days_back)).date()
            date_to = args.date_to or (current + timedelta(days=args.days_ahead)).date()
            if date_to < date_from:
                logger.error("--date-to must be on or after --date-from.")
                return 2
            result = sync_live_fixtures(
                engine,
                settings=settings,
                date_from=date_from,
                date_to=date_to,
                now=current,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "settle":
            result = settle_results(
                engine,
                settings=settings,
                now=datetime.now(UTC),
                days_back=args.days_back,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "predict":
            result = predict_upcoming(
                engine,
                settings=settings,
                window_hours=args.window_hours,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 0
        if args.command == "scorecard":
            if args.bootstrap_replicates < 1:
                logger.error("--bootstrap-replicates must be positive.")
                return 2
            result = refresh_scorecard(
                engine,
                now=datetime.now(UTC),
                bootstrap_replicates=args.bootstrap_replicates,
                seed=settings.random_seed,
            )
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 0
        if args.command == "telegram-bot":
            bot = TelegramBot(engine, settings=settings)
            try:
                bot.run_forever()
            finally:
                bot.close()
            return 0
        if not 1 <= args.port <= 65535:
            logger.error("--port must be between 1 and 65535.")
            return 2
        import uvicorn

        from forecast.api.app import create_app

        uvicorn.run(create_app(engine=engine, settings=settings), host=args.host, port=args.port)
        return 0
    if args.command == "sync-xg":
        if not args.league or not args.season:
            logger.error("sync-xg requires both --league and --season.")
            return 2
        create_schema(engine)
        client = UnderstatClient()
        try:
            matches = client.get_matches(league=args.league, season=args.season)
        finally:
            client.close()
        with Session(engine) as session:
            counters = ingest_understat_xg(
                session,
                league=args.league,
                season=args.season,
                matches=matches,
            )
            session.commit()
        print(
            f"Understat {args.league} {args.season}: "
            f"{counters['inserted']} xG rows inserted, "
            f"{counters['unchanged']} already present, "
            f"{counters['unmatched']} fixtures unmatched"
        )
        return 1 if counters["unmatched"] else 0

    logger.error("Command '%s' is not implemented yet.", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())

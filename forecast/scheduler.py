"""Single-instance production scheduler for forecast operations."""

from __future__ import annotations

import logging
import signal
import threading
from datetime import UTC, datetime, timedelta

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from forecast.config import Settings
from forecast.ingestion.understat import UnderstatClient, ingest_understat_xg
from forecast.jobs import predict_upcoming, refresh_scorecard, settle_results, sync_live_fixtures
from forecast.ledger.publisher import GitHubLedgerPublisher

logger = logging.getLogger(__name__)


def run_scheduler(engine: Engine, settings: Settings) -> None:
    """Run one serialized scheduler; deploy exactly one instance of this service."""
    scheduler = BackgroundScheduler(
        timezone="UTC",
        executors={"default": ThreadPoolExecutor(max_workers=1)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 900},
    )
    scheduler.add_job(
        _sync_live,
        "interval",
        hours=6,
        id="sync-live",
        args=(engine, settings),
        next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        _settle,
        "interval",
        hours=1,
        id="settle",
        args=(engine, settings),
        next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        _sync_xg,
        "interval",
        hours=24,
        id="sync-xg",
        args=(engine,),
        next_run_time=datetime.now(UTC) + timedelta(minutes=1),
    )
    scheduler.add_job(
        predict_upcoming,
        "interval",
        hours=1,
        id="predict",
        args=(engine,),
        kwargs={"settings": settings, "window_hours": 30},
        next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        refresh_scorecard,
        "interval",
        hours=24,
        id="scorecard",
        args=(engine,),
        kwargs={"bootstrap_replicates": 2000, "seed": settings.random_seed},
        next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        _publish,
        "interval",
        minutes=10,
        id="publish-ledger",
        args=(engine, settings),
        next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        _upgrade_proofs,
        "interval",
        hours=24,
        id="verify-proofs",
        args=(engine, settings),
        next_run_time=datetime.now(UTC) + timedelta(minutes=2),
    )
    logger.info("Production scheduler started with seven serialized jobs")
    scheduler.start()
    stop_event = threading.Event()

    def request_shutdown(signum, _frame) -> None:
        logger.info("Received shutdown signal %s", signum)
        stop_event.set()

    previous_handlers = {
        signum: signal.signal(signum, request_shutdown)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        stop_event.wait()
    finally:
        scheduler.shutdown(wait=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _sync_live(engine: Engine, settings: Settings) -> None:
    now = datetime.now(UTC)
    sync_live_fixtures(
        engine,
        settings=settings,
        date_from=(now - timedelta(days=14)).date(),
        date_to=(now + timedelta(days=30)).date(),
        now=now,
    )


def _settle(engine: Engine, settings: Settings) -> None:
    settle_results(engine, settings=settings, now=datetime.now(UTC), days_back=14)


def _sync_xg(engine: Engine) -> None:
    now = datetime.now(UTC)
    season_start = now.year if now.month >= 7 else now.year - 1
    season = f"{season_start}-{str(season_start + 1)[-2:]}"
    client = UnderstatClient()
    try:
        with Session(engine) as session:
            for league in ("PL", "PD"):
                matches = client.get_matches(league=league, season=season)
                ingest_understat_xg(
                    session,
                    league=league,
                    season=season,
                    matches=matches,
                )
            session.commit()
    finally:
        client.close()


def _publish(engine: Engine, settings: Settings) -> None:
    publisher = GitHubLedgerPublisher(settings)
    try:
        publisher.publish_pending(engine)
    finally:
        publisher.close()


def _upgrade_proofs(engine: Engine, settings: Settings) -> None:
    publisher = GitHubLedgerPublisher(settings)
    try:
        publisher.upgrade_pending_proofs(engine)
    finally:
        publisher.close()

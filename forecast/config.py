"""Validated application settings and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LEAGUES: Final[dict[str, str]] = {
    "PL": "Premier League",
    "PD": "La Liga",
}
HISTORICAL_COMPLETED_SEASONS: Final = 10


def normalize_database_url(value: str) -> str:
    """Use the declared psycopg 3 driver for Render and conventional Postgres URLs."""
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration, sourced from environment variables."""

    database_url: str | None
    football_data_org_token: str | None
    football_data_org_base_url: str
    web_origin: str
    ledger_public_url: str | None
    ledger_public_branch: str
    ledger_public_path: str
    github_token: str | None
    opentimestamps_command: str
    anthropic_api_key: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    random_seed: int
    data_dir: Path
    models_dir: Path
    reports_dir: Path
    ledger_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        """Read settings from the process environment and validate scalar values."""
        seed_value = os.environ.get("BWB_RANDOM_SEED", "42")
        try:
            random_seed = int(seed_value)
        except ValueError as exc:
            raise ValueError("BWB_RANDOM_SEED must be an integer") from exc

        return cls(
            database_url=(
                normalize_database_url(os.environ["DATABASE_URL"])
                if os.environ.get("DATABASE_URL")
                else None
            ),
            football_data_org_token=os.environ.get("FOOTBALL_DATA_ORG_TOKEN") or None,
            football_data_org_base_url=os.environ.get(
                "FOOTBALL_DATA_ORG_BASE_URL", "https://api.football-data.org/v4"
            ),
            web_origin=os.environ.get("WEB_ORIGIN", "http://localhost:3000"),
            ledger_public_url=os.environ.get("BWB_LEDGER_PUBLIC_URL") or None,
            ledger_public_branch=os.environ.get("BWB_LEDGER_PUBLIC_BRANCH", "main"),
            ledger_public_path=os.environ.get("BWB_LEDGER_PUBLIC_PATH", "ledger").strip("/"),
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            opentimestamps_command=os.environ.get("OPENTIMESTAMPS_COMMAND", "ots"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
            random_seed=random_seed,
            data_dir=Path(os.environ.get("BWB_DATA_DIR", PROJECT_ROOT / "data")),
            models_dir=Path(os.environ.get("BWB_MODELS_DIR", PROJECT_ROOT / "models")),
            reports_dir=Path(os.environ.get("BWB_REPORTS_DIR", PROJECT_ROOT / "reports")),
            ledger_dir=Path(os.environ.get("BWB_LEDGER_DIR", PROJECT_ROOT / "ledger")),
        )

    def require_database_url(self) -> str:
        """Return the configured database URL or raise an actionable error."""
        if self.database_url is None:
            raise RuntimeError("DATABASE_URL is required for database operations")
        return self.database_url

"""Tests for environment-backed settings."""

from pathlib import Path

import pytest

from forecast.config import PROJECT_ROOT, SUPPORTED_LEAGUES, Settings, normalize_database_url


def test_settings_use_deterministic_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("BWB_RANDOM_SEED", raising=False)

    settings = Settings.from_env()

    assert settings.random_seed == 42
    assert settings.database_url is None
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.ledger_dir == PROJECT_ROOT / "ledger"


def test_supported_leagues_are_explicit() -> None:
    assert SUPPORTED_LEAGUES == {"PL": "Premier League", "PD": "La Liga"}


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://example")
    monkeypatch.setenv("BWB_RANDOM_SEED", "123")

    settings = Settings.from_env()

    assert settings.database_url == "postgresql+psycopg://example"
    assert settings.random_seed == 123
    assert settings.models_dir == Path(PROJECT_ROOT / "models")


def test_invalid_seed_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BWB_RANDOM_SEED", "not-an-int")

    with pytest.raises(ValueError, match="BWB_RANDOM_SEED must be an integer"):
        Settings.from_env()


def test_database_is_required_only_for_database_operations() -> None:
    settings = Settings.from_env()

    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        settings.require_database_url()


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("postgres://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        ("postgresql://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        ("sqlite:///data/test.db", "sqlite:///data/test.db"),
    ],
)
def test_database_urls_select_supported_driver(database_url: str, expected: str) -> None:
    assert normalize_database_url(database_url) == expected

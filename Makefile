.PHONY: setup lint test ingest sync-xg sync-live settle backtest predict serve scorecard telegram-bot

export UV_PROJECT_ENVIRONMENT := .venv

setup:
	uv sync --all-groups --locked

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

test:
	uv run --locked python -m pytest

ingest:
	uv run --locked bwb ingest --download

sync-xg:
	uv run --locked bwb sync-xg --league "$(LEAGUE)" --season "$(SEASON)"

sync-live:
	uv run --locked bwb sync-live

settle:
	uv run --locked bwb settle

backtest:
	uv run --locked bwb backtest

predict:
	uv run --locked bwb predict

serve:
	uv run --locked bwb serve

scorecard:
	uv run --locked bwb scorecard

telegram-bot:
	uv run --locked bwb telegram-bot

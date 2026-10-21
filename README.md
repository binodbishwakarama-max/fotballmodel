# BWB Forecast

BWB Forecast is a public football forecasting engine for pre-match home/draw/away probabilities in the Premier League and La Liga. Its record is designed to be reproducible and auditable: forecasts are published before kickoff, scored against clearly identified baselines, and never rewritten.

## Project status

The repository contains the Python ingestion, feature, Elo/model-training, evaluation, ledger, API, and Next.js scoreboard foundations. The live xG importer, leakage-gated backtest, live prediction, settlement, scorecard, API, and scheduled operations are available. Production deployment infrastructure is described by `render.yaml`; it still requires external credentials, a qualified production model artifact, and a live deployment check. No model is approved without eligible evidence. The product and data contracts in [`docs/data-and-evaluation-contract.md`](docs/data-and-evaluation-contract.md) are normative.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Quick start

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest
```

Copy `.env.example` to `.env` and set credentials only when the corresponding integration is implemented. Never commit `.env`.

The Makefile exports a workspace-local `.venv` path for its commands. In PowerShell, set `UV_PROJECT_ENVIRONMENT` as shown before invoking `uv` directly so it does not select a globally configured environment.

## Local workflow

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv run --locked bwb init-db
uv run --locked bwb migrate --database-url sqlite:///data/bwb_forecast.db
uv run --locked bwb ingest --download --league PL --season 2025-26
uv run --locked bwb sync-xg --league PL --season 2025-26
uv run --locked bwb backtest
uv run --locked bwb sync-live
uv run --locked bwb settle
uv run --locked bwb predict --dry-run
uv run --locked bwb scorecard
uv run --locked bwb serve
uv run --locked bwb telegram-bot
```

`sync-xg` requires the corresponding fixtures and teams to have been ingested first. Team names that do not have a canonical or explicit Understat alias are rejected. xG retrieved from Understat is marked available at the retrieval time, so it can inform future forecasts but is never retroactively used in historical folds.

The backtest writes `reports/generated/backtest.md` and a calibration SVG. It returns a blocked status rather than reporting performance when source timestamps do not support leakage-safe folds; historical CSV data first ingested today cannot establish that its results were known in earlier seasons. Live prediction requires an approved production model in `models/production.json`; `--dry-run` previews without writing forecasts. Live fixture sync requires `FOOTBALL_DATA_ORG_TOKEN`. The restricted Telegram bot requires both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` and supports `/next` and `/score`. The API binds to `127.0.0.1:8000` by default.

## Local container deployment

Set `POSTGRES_PASSWORD` in `.env` to a URL-safe value, then run:

```powershell
docker compose up --build --wait
```

This compose file is for local evaluation, not a hardened production deployment.

## Render deployment

`render.yaml` defines the Next.js scoreboard, read-only API, managed PostgreSQL database, and one serialized background scheduler. After this application is pushed to a GitHub repository, create the Render Blueprint from that repository and configure `FOOTBALL_DATA_ORG_TOKEN` and a fine-grained `GITHUB_TOKEN` as Render secrets; never put credential values in this file or in chat. The GitHub token needs Contents write access to the selected repository. Confirm the repository owner, branch, and `BWB_LEDGER_PUBLIC_PATH` before allowing writes. The publisher only creates ledger JSON paths and refuses to overwrite them; OpenTimestamps sidecars are upgraded as proofs become independently verifiable.

The API deployment runs Alembic migrations as a pre-deploy step. The scheduler is deliberately a single instance: it serializes live fixture sync, Understat xG refresh, settlement, prediction, scorecard refresh, ledger publication, and proof upgrades. Do not scale it horizontally. It uses a Render persistent disk for model artifacts; provision an eligible artifact at `/var/data/models` before expecting forecasts. A blank disk or absent/unapproved artifact is a hard prediction failure, not a fallback to an unqualified model. Verify `/health`, scheduled-job logs, public ledger bytes, and proof status after deploying.

OpenTimestamps proofs begin as `pending`; the scheduled upgrade checks again daily and only marks a proof `verified` after Bitcoin anchoring is reported by the client. Stamping sends the ledger's cryptographic digest and timing request to public calendar operators; it does not upload the ledger contents, but it does reveal that the digest was submitted from the service's network. This is not a guarantee of continuous availability. Configure Render alerts, database backups, and a tested restore process before serving users. The blueprint is infrastructure-as-code, not proof that production services, credentials, monitoring, model evidence, or external integrations have been activated.

## Build principles

- Use information demonstrably available before each forecast cutoff; timestamps are UTC-aware.
- Use chronological, walk-forward evaluation only.
- Compare all baselines on the same eligible fixtures and report coverage.
- Keep published predictions immutable; corrections are new, explicitly identified records.
- Pin the resolved dependency graph in `uv.lock`.

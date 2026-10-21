FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.12.3 \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/models /app/ledger /app/reports \
    && chown -R app:app /app

COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app forecast ./forecast
COPY --chown=app:app alembic.ini ./alembic.ini
COPY --chown=app:app migrations ./migrations

USER app

RUN uv sync --locked --no-dev --group data --group db --group api --group ops

EXPOSE 8000

CMD ["uv", "run", "--locked", "bwb", "serve", "--host", "0.0.0.0"]

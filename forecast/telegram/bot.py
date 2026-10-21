"""Restricted-chat Telegram bot for the published forecast and scorecard."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from forecast.config import Settings
from forecast.db import Engine
from forecast.models import Match, Prediction, Score, Team

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_NEXT_MATCHES = 5


class TelegramBot:
    """Long-poll a single configured chat and serve read-only public commands."""

    def __init__(
        self,
        engine: Engine,
        *,
        settings: Settings,
        client: httpx.Client | None = None,
        api_base: str = TELEGRAM_API_BASE,
    ) -> None:
        if not settings.telegram_bot_token or not settings.telegram_bot_token.strip():
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the Telegram bot")
        if not settings.telegram_chat_id or not settings.telegram_chat_id.strip():
            raise RuntimeError("TELEGRAM_CHAT_ID is required to restrict Telegram access")
        self.engine = engine
        self.chat_id = settings.telegram_chat_id.strip()
        self.offset_path = settings.data_dir / "cache" / "telegram" / "update-offset"
        self.api_base = api_base.rstrip("/")
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(35.0),
            headers={"User-Agent": "BWB-Forecast/0.1"},
        )
        self._owns_client = client is None
        self._updates_url = f"{self.api_base}/bot{settings.telegram_bot_token}/getUpdates"
        self._send_url = f"{self.api_base}/bot{settings.telegram_bot_token}/sendMessage"
        self.offset = self._read_offset()

    def close(self) -> None:
        """Close the HTTP client when owned by this instance."""
        if self._owns_client:
            self.client.close()

    def run_forever(self) -> None:
        """Poll Telegram until stopped; API and transport errors are surfaced."""
        while True:
            updates = self._api_get(
                self._updates_url,
                {
                    "offset": self.offset,
                    "timeout": 25,
                    "allowed_updates": '["message"]',
                },
                timeout=30.0,
            )
            if not isinstance(updates, list):
                raise ValueError("Telegram getUpdates result must be a list")
            for update in updates:
                self.process_update(update)

    def process_update(self, update: Any) -> bool:
        """Process and persist an update offset after successful command handling."""
        if (
            not isinstance(update, dict)
            or not isinstance(update.get("update_id"), int)
            or isinstance(update.get("update_id"), bool)
        ):
            raise ValueError("Telegram returned a malformed update")
        update_id = update["update_id"]
        if update_id < self.offset:
            return False
        message = update.get("message")
        if isinstance(message, dict):
            self.handle_message(message)
        self.offset = update_id + 1
        self._write_offset()
        return True

    def handle_message(self, message: dict[str, Any]) -> bool:
        """Handle an update only when its chat matches the configured allowlist."""
        chat = message.get("chat")
        if not isinstance(chat, dict) or str(chat.get("id")) != self.chat_id:
            return False
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return False
        command = text.strip().split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()
        if command in {"/start", "/help"}:
            response = (
                "BWB Forecast commands:\n"
                "/next - upcoming published forecasts\n"
                "/score - latest settled log-loss summary"
            )
        elif command == "/next":
            response = self._next_message()
        elif command == "/score":
            response = self._score_message()
        else:
            response = "Unknown command. Use /help."
        self._send_message(self.chat_id, response)
        return True

    def _next_message(self) -> str:
        now = datetime.now(UTC)
        home_alias = aliased(Team)
        away_alias = aliased(Team)
        statement = (
            select(Match, Prediction, home_alias, away_alias)
            .join(Prediction, Prediction.match_id == Match.id)
            .join(home_alias, Match.home_team_id == home_alias.id)
            .join(away_alias, Match.away_team_id == away_alias.id)
            .where(
                Match.status == "scheduled",
                Match.kickoff_time_known.is_(True),
                Match.kickoff_utc > now,
            )
            .order_by(Match.kickoff_utc, Prediction.created_at.desc())
        )
        lines = []
        seen: set[str] = set()
        with Session(self.engine) as session:
            for match, prediction, home, away in session.execute(statement):
                if match.id in seen:
                    continue
                seen.add(match.id)
                kickoff = _utc(match.kickoff_utc).strftime("%Y-%m-%d %H:%M UTC")
                lines.append(
                    f"{match.league}: {home.canonical_name} - {away.canonical_name} "
                    f"({kickoff})\n"
                    f"H {prediction.p_home:.1%} | D {prediction.p_draw:.1%} | "
                    f"A {prediction.p_away:.1%}"
                )
                if len(lines) == MAX_NEXT_MATCHES:
                    break
        return "\n\n".join(lines) if lines else "No upcoming published forecasts."

    def _score_message(self) -> str:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(Score).where(Score.metric == "log_loss").order_by(Score.computed_at.desc())
            ).all()
        if not rows:
            return "No settled scorecard results yet."
        latest_computed_at = _utc(rows[0].computed_at)
        latest = [row for row in rows if _utc(row.computed_at) == latest_computed_at]
        grouped: dict[str, list[float]] = {}
        for row in latest:
            grouped.setdefault(row.forecaster, []).append(row.value)
        lines = [f"Settled log loss ({latest_computed_at:%Y-%m-%d %H:%M UTC}):"]
        lines.extend(
            f"{forecaster}: {sum(values) / len(values):.4f} ({len(values)} matches)"
            for forecaster, values in sorted(grouped.items())
        )
        return "\n".join(lines)

    def _send_message(self, chat_id: str, text: str) -> None:
        payload = self._api_post(self._send_url, {"chat_id": chat_id, "text": text})
        if not isinstance(payload, dict):
            raise ValueError("Telegram sendMessage result must be an object")

    def _read_offset(self) -> int:
        if not self.offset_path.exists():
            return 0
        content = self.offset_path.read_text(encoding="ascii").strip()
        try:
            offset = int(content)
        except ValueError as exc:
            raise ValueError(f"Invalid Telegram update offset file: {self.offset_path}") from exc
        if offset < 0:
            raise ValueError(f"Invalid Telegram update offset file: {self.offset_path}")
        return offset

    def _write_offset(self) -> None:
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="ascii",
                dir=self.offset_path.parent,
                prefix=".telegram-offset-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                file.write(f"{self.offset}\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.offset_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _api_get(self, url: str, params: dict[str, Any], *, timeout: float) -> Any:
        try:
            response = self.client.get(url, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Telegram getUpdates request failed ({type(exc).__name__})"
            ) from None
        if response.is_error:
            raise RuntimeError(f"Telegram getUpdates returned HTTP {response.status_code}")
        return _telegram_result(response.json())

    def _api_post(self, url: str, payload: dict[str, Any]) -> Any:
        try:
            response = self.client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Telegram sendMessage request failed ({type(exc).__name__})"
            ) from None
        if response.is_error:
            raise RuntimeError(f"Telegram sendMessage returned HTTP {response.status_code}")
        return _telegram_result(response.json())


def _telegram_result(payload: Any) -> Any:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ValueError("Telegram API returned an unsuccessful response")
    if "result" not in payload:
        raise ValueError("Telegram API response is missing result")
    return payload["result"]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

import json
from dataclasses import replace

import httpx
import pytest

from forecast.config import Settings
from forecast.db import create_schema, get_engine
from forecast.telegram.bot import TelegramBot


def test_bot_restricts_chat_and_serves_read_only_commands() -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bot = TelegramBot(
        engine,
        settings=replace(
            Settings.from_env(),
            telegram_bot_token="test-token",
            telegram_chat_id="42",
        ),
        client=client,
        api_base="https://telegram.test",
    )

    assert not bot.handle_message({"chat": {"id": 99}, "text": "/next"})
    assert requests == []
    assert bot.handle_message({"chat": {"id": 42}, "text": "/next"})
    assert requests[0].url.path.endswith("/sendMessage")
    payload = json.loads(requests[0].content)
    assert payload["chat_id"] == "42"
    assert payload["text"] == "No upcoming published forecasts."
    client.close()


def test_bot_requires_a_configured_chat_allowlist() -> None:
    with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
        TelegramBot(
            get_engine("sqlite:///:memory:"),
            settings=replace(
                Settings.from_env(),
                telegram_bot_token="test-token",
                telegram_chat_id=None,
            ),
        )


def test_bot_persists_update_offset_to_avoid_replaying_processed_updates(tmp_path) -> None:
    engine = get_engine("sqlite:///:memory:")
    create_schema(engine)
    settings = replace(
        Settings.from_env(),
        telegram_bot_token="test-token",
        telegram_chat_id="42",
        data_dir=tmp_path,
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    bot = TelegramBot(engine, settings=settings, client=client)

    update = {"update_id": 8, "message": {"chat": {"id": 99}, "text": "/next"}}
    assert bot.process_update(update)
    assert bot.offset == 9
    assert not bot.process_update(update)

    restarted = TelegramBot(engine, settings=settings, client=client)
    assert restarted.offset == 9
    client.close()

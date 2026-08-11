"""QQBot C2C 实时流式输出测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.kernel.llm.model_client.base import StreamEvent

from ..humanize.qqbot_streaming import (
    QQBotStreamingContext,
    QQBotStreamingSession,
    create_streaming_session,
    extract_streaming_context,
)


def _message(**overrides):
    values = {
        "platform": "qq",
        "chat_type": "private",
        "sender_id": "sender",
        "message_id": "message-id",
        "extra": {"qq_user_openid": "openid", "qq_event_type": "C2C_MESSAGE_CREATE"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_extract_streaming_context_supports_c2c_and_interaction() -> None:
    context = extract_streaming_context(_message())
    assert context == QQBotStreamingContext("openid", msg_id="message-id")

    interaction = extract_streaming_context(
        _message(
            extra={
                "qq_user_openid": "openid",
                "qq_event_type": "INTERACTION_CREATE",
                "qq_event_id": "event-id",
            }
        )
    )
    assert interaction == QQBotStreamingContext("openid", event_id="event-id")


def test_extract_streaming_context_rejects_non_c2c() -> None:
    assert extract_streaming_context(_message(platform="other")) is None
    assert extract_streaming_context(_message(chat_type="group")) is None
    assert extract_streaming_context(_message(sender_id="", extra={})) is None


def test_create_streaming_session_respects_enable_and_service() -> None:
    service = SimpleNamespace(start_streaming=AsyncMock())

    def resolver(_signature: str):
        return service

    assert create_streaming_session(_message(), enabled=False, service_resolver=resolver) is None
    session = create_streaming_session(_message(), enabled=True, service_resolver=resolver)
    assert session is not None
    assert session.service is service


async def test_streaming_uses_only_visible_text_delta() -> None:
    controller = SimpleNamespace(update=AsyncMock(return_value=True), end=AsyncMock(return_value=True))
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    await session.on_event(StreamEvent(reasoning_delta="不能发送的思考"))
    await session.on_event(StreamEvent(text_delta="你"))
    await session.on_event(StreamEvent(text_delta="好"))

    service.start_streaming.assert_awaited_once_with(
        user_openid="openid", initial_text="你", event_id="", msg_id="msg"
    )
    controller.update.assert_awaited_once_with("你好")
    assert "不能发送" not in session.visible_text


async def test_streaming_filters_thought_tags_across_chunks() -> None:
    controller = SimpleNamespace(update=AsyncMock(return_value=True), end=AsyncMock(return_value=True))
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    for delta in ("<thi", "nk>隐藏", "内容</think>", "回答"):
        await session.on_event(StreamEvent(text_delta=delta))

    assert session.visible_text == "回答"
    assert session.removed_thoughts == ["隐藏内容"]
    service.start_streaming.assert_awaited_once()
    assert service.start_streaming.await_args.kwargs["initial_text"] == "回答"

    result = await session.finalize("<think>隐藏内容</think>回答")
    assert result.text == "回答"
    controller.end.assert_awaited_once_with("回答")


async def test_start_failure_leaves_session_for_normal_fallback() -> None:
    service = SimpleNamespace(
        start_streaming=AsyncMock(return_value={"success": False, "controller": None})
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    await session.on_event(StreamEvent(text_delta="回答"))

    assert not session.started
    assert session.start_failed
    result = await session.finalize("回答")
    assert result.text == "回答"


async def test_update_failure_still_ends_without_restarting() -> None:
    controller = SimpleNamespace(update=AsyncMock(return_value=False), end=AsyncMock(return_value=True))
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    await session.on_event(StreamEvent(text_delta="你"))
    await session.on_event(StreamEvent(text_delta="好"))
    assert session.update_failed



async def test_streaming_filters_thought_tag_attributes_across_chunks() -> None:
    controller = SimpleNamespace(update=AsyncMock(return_value=True), end=AsyncMock(return_value=True))
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    for delta in ("<think class=\"internal\"", ">隐藏</think>", "回答"):
        await session.on_event(StreamEvent(text_delta=delta))

    assert session.visible_text == "回答"
    assert session.removed_thoughts == ["隐藏"]
    assert service.start_streaming.await_args.kwargs["initial_text"] == "回答"


async def test_finalize_preserves_submitted_prefix_when_cleaning_differs() -> None:
    controller = SimpleNamespace(update=AsyncMock(return_value=True), end=AsyncMock(return_value=True))
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(QQBotStreamingContext("openid", msg_id="msg"), service)

    await session.on_event(StreamEvent(text_delta='"回答"'))
    result = await session.finalize('"回答"')

    assert result.text == "回答"
    controller.end.assert_awaited_once_with('"回答"')

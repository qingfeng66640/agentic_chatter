"""SayAction 结构化 mention 发送测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.core.models.message import Message
from src.core.models.stream import ChatStream
from src.core.transport.message_receive.converter import MessageConverter

from ..actions import say as say_module
from ..actions.say import SayAction

OPENID = "0D0C972A218610FF683F81FAEE647A2F"
TEXT = "qf阁下？蝶在的，怎么了？"


def _group_action() -> SayAction:
    stream = ChatStream(
        "stream-group",
        platform="qq",
        chat_type="group",
        bot_id="bot-id",
        bot_nickname="Bot",
    )
    stream.context.history_messages.append(
        Message(
            message_id="incoming-1",
            content="测试",
            processed_plain_text="测试",
            sender_id="user-1",
            platform="qq",
            chat_type="group",
            group_id="group-1",
            group_name="测试群",
        )
    )
    return SayAction(chat_stream=stream, plugin=object())


async def test_group_at_sends_structured_message_and_preserves_target(
    monkeypatch,
) -> None:
    """群聊 @ 必须通过 Message.extra 传递，而非拼接进正文。"""
    sent: list[Message] = []
    send_message = AsyncMock(side_effect=lambda message: sent.append(message) or True)
    monkeypatch.setattr(say_module.send_api, "send_message", send_message)
    monkeypatch.setattr(
        say_module,
        "get_bot_info_by_platform",
        AsyncMock(return_value={"bot_id": "bot-id", "bot_name": "Bot"}),
    )

    ok, detail = await _group_action().execute(TEXT, at=f"  {OPENID}  ")

    assert (ok, detail) == (True, "已发送")
    send_message.assert_awaited_once()
    message = sent[0]
    assert message.content == TEXT
    assert message.processed_plain_text == TEXT
    assert f"@{OPENID}" not in message.content
    assert message.extra["at_user_id"] == OPENID
    assert message.extra["target_group_id"] == "group-1"
    assert message.extra["target_group_name"] == "测试群"
    assert message.chat_type == "group"
    assert message.stream_id == "stream-group"

    message.stream_id = ""
    envelope = await MessageConverter().message_to_envelope(message)
    assert envelope["message_segment"] == [
        {"type": "at", "data": OPENID},
        {"type": "text", "data": TEXT},
    ]


async def test_group_plain_text_keeps_existing_send_text_path(monkeypatch) -> None:
    """无 @ 时保持普通正文发送，不写入 mention 元数据。"""
    send_text = AsyncMock(return_value=True)
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)

    ok, detail = await _group_action().execute("  普通正文  ")

    assert (ok, detail) == (True, "已发送")
    send_text.assert_awaited_once_with(
        content="普通正文", stream_id="stream-group", reply_to=None
    )


async def test_reply_to_and_at_are_mutually_exclusive(monkeypatch) -> None:
    """引用与 mention 同时存在必须失败且不发送。"""
    send_text = AsyncMock()
    send_message = AsyncMock()
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)
    monkeypatch.setattr(say_module.send_api, "send_message", send_message)

    ok, detail = await _group_action().execute(TEXT, reply_to="message-1", at=OPENID)

    assert not ok
    assert detail == "reply_to 与 at 不能同时使用"
    send_text.assert_not_awaited()
    send_message.assert_not_awaited()


async def test_private_at_fails_without_leaking_target(monkeypatch) -> None:
    """私聊不能发送结构化 mention，也不能泄露目标 ID。"""
    stream = ChatStream("stream-private", platform="qq", chat_type="private")
    action = SayAction(chat_stream=stream, plugin=object())
    send_text = AsyncMock()
    send_message = AsyncMock()
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)
    monkeypatch.setattr(say_module.send_api, "send_message", send_message)

    ok, detail = await action.execute(TEXT, at=OPENID)

    assert not ok
    assert "群聊" in detail
    assert OPENID not in detail
    send_text.assert_not_awaited()
    send_message.assert_not_awaited()


async def test_reply_to_without_at_keeps_existing_behavior(monkeypatch) -> None:
    """仅引用回复继续使用 send_text 原有行为。"""
    send_text = AsyncMock(return_value=True)
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)

    ok, _ = await _group_action().execute(TEXT, reply_to="message-1")

    assert ok
    send_text.assert_awaited_once_with(
        content=TEXT, stream_id="stream-group", reply_to="message-1"
    )


async def test_send_failures_keep_existing_result_semantics(monkeypatch) -> None:
    """mention 发送失败或异常应保留失败语义。"""
    monkeypatch.setattr(say_module, "get_bot_info_by_platform", AsyncMock(return_value={}))
    monkeypatch.setattr(say_module.send_api, "send_message", AsyncMock(return_value=False))
    ok, detail = await _group_action().execute(TEXT, at=OPENID)
    assert not ok
    assert detail == "发送失败，适配器返回未成功"

    monkeypatch.setattr(
        say_module.send_api, "send_message", AsyncMock(side_effect=RuntimeError("boom"))
    )
    ok, detail = await _group_action().execute(TEXT, at=OPENID)
    assert not ok
    assert detail == "发送失败: boom"

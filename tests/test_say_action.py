"""SayAction 结构化 mention 发送测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.models.message import Message
from src.core.models.stream import ChatStream
from src.core.transport.message_receive.converter import MessageConverter

from ..actions import say as say_module
from ..actions.say import SayAction


def test_say_action_description_limits_use_to_quote_or_mention() -> None:
    """工具 schema 必须说明 action-say 的特殊能力和必填目标。"""
    description = SayAction.action_description

    assert "引用回复某条消息" in description
    assert "@ 某个人" in description
    assert "target_type" in description
    assert "target_id" in description
    assert "同一响应再输出正文" in description


def test_say_action_schema_requires_mode_and_target_id() -> None:
    """生成给模型的 schema 必须强制模式、目标 ID 和正文。"""
    parameters = SayAction.to_schema()["function"]["parameters"]

    assert parameters["required"] == ["content", "target_type", "target_id"]
    assert parameters["properties"]["target_type"]["enum"] == ["reply", "mention"]


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

    ok, detail = await _group_action().execute(
        TEXT, target_type="mention", target_id=f"  {OPENID}  "
    )

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


async def test_action_requires_target_parameters(monkeypatch) -> None:
    """调用时省略模式或目标 ID 必须在参数绑定阶段失败。"""
    send_text = AsyncMock()
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)

    with pytest.raises(TypeError):
        await _group_action().execute(TEXT)

    send_text.assert_not_awaited()


async def test_empty_target_id_fails_without_sending(monkeypatch) -> None:
    """空白目标不能绕过必填目标约束。"""
    send_text = AsyncMock()
    send_message = AsyncMock()
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)
    monkeypatch.setattr(say_module.send_api, "send_message", send_message)

    ok, detail = await _group_action().execute(
        TEXT, target_type="reply", target_id="  "
    )

    assert not ok
    assert detail == "target_id 不能为空"
    send_text.assert_not_awaited()
    send_message.assert_not_awaited()


async def test_invalid_target_type_fails_without_sending(monkeypatch) -> None:
    """未知发送模式必须失败且不触发发送 API。"""
    send_text = AsyncMock()
    send_message = AsyncMock()
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)
    monkeypatch.setattr(say_module.send_api, "send_message", send_message)

    ok, detail = await _group_action().execute(
        TEXT, target_type="plain", target_id="target-1"  # type: ignore[arg-type]
    )

    assert not ok
    assert detail == "target_type 必须是 reply 或 mention"
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

    ok, detail = await action.execute(
        TEXT, target_type="mention", target_id=OPENID
    )

    assert not ok
    assert "群聊" in detail
    assert OPENID not in detail
    send_text.assert_not_awaited()
    send_message.assert_not_awaited()


async def test_reply_mode_uses_required_message_id(monkeypatch) -> None:
    """引用模式必须将目标消息 ID 传入 reply_to。"""
    send_text = AsyncMock(return_value=True)
    monkeypatch.setattr(say_module.send_api, "send_text", send_text)

    ok, _ = await _group_action().execute(
        TEXT, target_type="reply", target_id="message-1"
    )

    assert ok
    send_text.assert_awaited_once_with(
        content=TEXT, stream_id="stream-group", reply_to="message-1"
    )


async def test_send_failures_keep_existing_result_semantics(monkeypatch) -> None:
    """mention 发送失败或异常应保留失败语义。"""
    monkeypatch.setattr(say_module, "get_bot_info_by_platform", AsyncMock(return_value={}))
    monkeypatch.setattr(say_module.send_api, "send_message", AsyncMock(return_value=False))
    ok, detail = await _group_action().execute(
        TEXT, target_type="mention", target_id=OPENID
    )
    assert not ok
    assert detail == "发送失败，适配器返回未成功"

    monkeypatch.setattr(
        say_module.send_api, "send_message", AsyncMock(side_effect=RuntimeError("boom"))
    )
    ok, detail = await _group_action().execute(
        TEXT, target_type="mention", target_id=OPENID
    )
    assert not ok
    assert detail == "发送失败: boom"

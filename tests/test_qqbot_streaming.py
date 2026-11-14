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


def _message(**overrides: str | dict[str, str]) -> SimpleNamespace:
    """构造默认 QQ 私聊消息，并允许覆盖上下文提取所需字段。

    Args:
        **overrides: 替换默认字段的文本值或额外信息字典。

    Returns:
        SimpleNamespace: 用于流式上下文提取的消息替身。
    """
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
    """私聊消息应使用消息标识，交互事件应使用事件标识构建流式上下文。"""
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
    """非 QQ 平台、群聊及缺少用户标识的消息不得建立流式上下文。"""
    assert extract_streaming_context(_message(platform="other")) is None
    assert extract_streaming_context(_message(chat_type="group")) is None
    assert extract_streaming_context(_message(sender_id="", extra={})) is None


def test_create_streaming_session_respects_enable_and_service() -> None:
    """关闭开关时不建会话，开启时应使用解析得到的服务实例。"""
    service = SimpleNamespace(start_streaming=AsyncMock())

    def resolver(_signature: str) -> SimpleNamespace:
        """忽略服务签名，返回当前测试准备的流式服务替身。"""
        return service

    assert (
        create_streaming_session(_message(), enabled=False, service_resolver=resolver)
        is None
    )
    session = create_streaming_session(
        _message(), enabled=True, service_resolver=resolver
    )
    assert session is not None
    assert session.service is service


async def test_streaming_uses_only_visible_text_delta() -> None:
    """推理增量不得发送，正文增量应依次用于启动及更新流式正文。"""
    controller = SimpleNamespace(
        update=AsyncMock(return_value=True), end=AsyncMock(return_value=True)
    )
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

    await session.on_event(StreamEvent(reasoning_delta="不能发送的思考"))
    await session.on_event(StreamEvent(text_delta="你"))
    await session.on_event(StreamEvent(text_delta="好"))

    service.start_streaming.assert_awaited_once_with(
        user_openid="openid", initial_text="你", event_id="", msg_id="msg"
    )
    controller.update.assert_awaited_once_with("你好")
    assert "不能发送" not in session.visible_text


async def test_streaming_does_not_emit_gemini_filter_message() -> None:
    """跨分片识别 Gemini 过滤提示后，不得启动、更新或结束发送。"""
    controller = SimpleNamespace(update=AsyncMock(), end=AsyncMock())
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )
    text = (
        "This request was blocked by Gemini's filters. They can occasionally trigger "
        "by mistake on safe coding, security, or biology-related queries. Please try "
        "rephrasing your prompt. You can [send feedback](https://ai.google.dev/gemini-api/docs/troubleshooting#file-bug) "
        "or read more about [our policies here](https://policies.google.com/terms/generative-ai/use-policy)."
    )
    parts = ("This req", "uest was blocked by Gemini's filters.")
    for delta in ("<think>内部</think>" + parts[0], parts[1]):
        await session.on_event(StreamEvent(text_delta=delta))
        service.start_streaming.assert_not_awaited()
    await session.on_event(StreamEvent(text_delta=text[len("".join(parts)) :]))

    assert (await session.finalize(text)).text == ""
    service.start_streaming.assert_not_awaited()
    controller.update.assert_not_awaited()
    controller.end.assert_not_awaited()


async def test_streaming_does_not_emit_old_google_filter_message() -> None:
    """旧版 Google 审核错误即使分片到达，也不得启动流式发送。"""
    service = SimpleNamespace(start_streaming=AsyncMock())
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )
    text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]. "
        "Try rephrasing the prompt."
    )
    parts = ("The prom", "pt could not be submitted.")
    for delta in (*parts, text[len("".join(parts)) :]):
        await session.on_event(StreamEvent(text_delta=delta))

    assert (await session.finalize(text)).text == ""
    service.start_streaming.assert_not_awaited()


async def test_streaming_flushes_normal_text_after_false_alarm() -> None:
    """疑似错误前缀补全为正常句子后，应一次性发送缓冲的完整正文。"""
    controller = SimpleNamespace(update=AsyncMock(return_value=True), end=AsyncMock())
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )
    await session.on_event(StreamEvent(text_delta="This request"))
    service.start_streaming.assert_not_awaited()
    await session.on_event(StreamEvent(text_delta=" is ready."))
    service.start_streaming.assert_awaited_once()
    assert (
        service.start_streaming.await_args.kwargs["initial_text"]
        == "This request is ready."
    )


async def test_streaming_filters_thought_tags_across_chunks() -> None:
    """跨分片思考标签应被移除，仅可见答案用于启动和结束流式发送。"""
    controller = SimpleNamespace(
        update=AsyncMock(return_value=True), end=AsyncMock(return_value=True)
    )
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

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
    """流式启动失败应记录失败状态，并保留正文供普通发送路径使用。"""
    service = SimpleNamespace(
        start_streaming=AsyncMock(return_value={"success": False, "controller": None})
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

    await session.on_event(StreamEvent(text_delta="回答"))

    assert not session.started
    assert session.start_failed
    result = await session.finalize("回答")
    assert result.text == "回答"


async def test_update_failure_still_ends_without_restarting() -> None:
    """流式控制器拒绝正文更新时，应记录更新失败状态。"""
    controller = SimpleNamespace(
        update=AsyncMock(return_value=False), end=AsyncMock(return_value=True)
    )
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

    await session.on_event(StreamEvent(text_delta="你"))
    await session.on_event(StreamEvent(text_delta="好"))
    assert session.update_failed


async def test_streaming_filters_thought_tag_attributes_across_chunks() -> None:
    """带属性且跨分片到达的思考标签也应隐藏，启动文本仅含答案。"""
    controller = SimpleNamespace(
        update=AsyncMock(return_value=True), end=AsyncMock(return_value=True)
    )
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

    for delta in ('<think class="internal"', ">隐藏</think>", "回答"):
        await session.on_event(StreamEvent(text_delta=delta))

    assert session.visible_text == "回答"
    assert session.removed_thoughts == ["隐藏"]
    assert service.start_streaming.await_args.kwargs["initial_text"] == "回答"


async def test_finalize_preserves_submitted_prefix_when_cleaning_differs() -> None:
    """清洗结果可去除外层引号，但流式结束文本应保留已提交的带引号前缀。"""
    controller = SimpleNamespace(
        update=AsyncMock(return_value=True), end=AsyncMock(return_value=True)
    )
    service = SimpleNamespace(
        start_streaming=AsyncMock(
            return_value={"success": True, "controller": controller, "error": None}
        )
    )
    session = QQBotStreamingSession(
        QQBotStreamingContext("openid", msg_id="msg"), service
    )

    await session.on_event(StreamEvent(text_delta='"回答"'))
    result = await session.finalize('"回答"')

    assert result.text == "回答"
    controller.end.assert_awaited_once_with('"回答"')

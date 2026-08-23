"""Agent 主循环测试。

重点守护文本与工具同轮并行、工具结果跟进，以及纯文本回复后的
确定性结束行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock

from src.kernel.llm import LLMPayload, ROLE, Text, ToolCall, ToolResult
from src.kernel.llm.context_structure import validate_payload_sequence

from ..actions.control import MAX_STOP_MINUTES
from .. import chatter as chatter_module
from ..chatter import AgenticChatter, _thought_log_line
from ..pipeline.mailbox import StreamMailbox
from ..pipeline.loop import (
    append_control_tool_results,
    append_interrupted_tool_results,
    build_speak_segments,
    build_tool_execution_records,
    classify_calls,
    collect_tool_result_delta,
    decide_iteration,
    deliver_segments,
    is_repeated_reply,
    normalize_reply_text,
    reply_similarity,
    should_continue_loop,
)
from ..pipeline.stages import (
    STAGE_ACT,
    STAGE_PERCEIVE,
    STAGE_PLAN,
    STAGE_REFLECT,
    resolve_stage_order,
)
from ..pipeline.state import TerminationReason, TurnState


@dataclass
class _FakeCall:
    """用于测试的假 tool call。"""

    name: str
    args: dict[str, Any]
    id: str = "call_1"


class _FakeHumanize:
    """用于测试的假拟人化配置。"""

    enable_segmentation = True
    max_segment_chars = 20
    max_segments = 3
    typing_cps = 0.0
    max_typing_delay = 2.0


def test_thought_log_line_is_single_line_and_bounded() -> None:
    line = _thought_log_line("stream", "reasoning_content", "a\n\t" + "长" * 500)

    assert "\n" not in line
    assert "\t" not in line
    assert len(line) <= 300
    assert line.endswith("…")


async def test_deliver_message_logs_structured_and_removed_thoughts(monkeypatch) -> None:
    """主 Agent 应在 INFO 中记录结构化和正文清洗思考。"""
    sent: list[str] = []

    async def send_text(*, content: str, stream_id: str) -> bool:
        sent.append(content)
        return bool(stream_id)

    info = Mock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module.logger, "info", info)
    chatter = AgenticChatter(stream_id="thought-stream", plugin=object())
    response = SimpleNamespace(
        message="<think>正文里的思考</think>最终回复",
        reasoning_content="结构化\n推理",
    )

    spoken, duplicate = await chatter._deliver_message(
        None,
        response,
        TurnState(stream_id="thought-stream"),
    )

    assert spoken
    assert not duplicate
    assert sent == ["最终回复"]
    logs = [str(call.args[0]) for call in info.call_args_list]
    assert len(logs) == 2
    assert any("source=reasoning_content" in line and "结构化 推理" in line for line in logs)
    assert any(
        "source=removed_message_block" in line and "正文里的思考" in line
        for line in logs
    )
    assert all("\n" not in line and len(line) <= 300 for line in logs)


async def test_deliver_message_does_not_log_thought_only_output(monkeypatch) -> None:
    """只有思考而无可见回复时不得发送或记录 INFO。"""
    send_text = AsyncMock(return_value=True)
    info = Mock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module.logger, "info", info)
    chatter = AgenticChatter(stream_id="thought-only", plugin=object())

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(
            message="<analysis>只有分析</analysis>",
            reasoning_content="结构化推理",
        ),
        TurnState(stream_id="thought-only"),
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    info.assert_not_called()


async def test_deliver_message_intercepts_provider_error_text(monkeypatch) -> None:
    """供应商审核错误正文不得进入发送 API。"""
    send_text = AsyncMock(return_value=True)
    warning = Mock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module.logger, "warning", warning)
    chatter = AgenticChatter(stream_id="provider-error", plugin=object())
    state = TurnState(stream_id="provider-error")
    visible_text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]. "
        "Try rephrasing the prompt."
    )

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=f"<think>内部内容</think> {visible_text}"),
        state,
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    warning.assert_called_once()
    log_line = str(warning.call_args.args[0])
    assert "event=provider_error_text_intercepted" in log_line
    assert "rule=google_prompt_policy_block" in log_line
    assert f"chars={len(visible_text)}" in log_line
    assert visible_text not in log_line
    assert not state.spoke
    assert not state.sent_texts
    assert state.visible_text_emissions == 0


async def test_deliver_message_provider_error_recording_is_disabled_by_default(
    monkeypatch,
) -> None:
    """未开启诊断开关时不得序列化或写入请求体。"""
    send_text = AsyncMock(return_value=True)
    build_record = Mock()
    append_record = AsyncMock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module, "build_provider_error_request_record", build_record)
    monkeypatch.setattr(chatter_module, "append_provider_error_request_record", append_record)
    chatter = AgenticChatter(stream_id="provider-error-off", plugin=object())
    state = TurnState(stream_id="provider-error-off")
    config = SimpleNamespace(tools=SimpleNamespace(record_provider_error_request_body=False))
    visible_text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]."
    )

    result = await chatter._deliver_message(
        config,
        SimpleNamespace(message=visible_text, payloads=["payload"]),
        state,
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    build_record.assert_not_called()
    append_record.assert_not_awaited()


async def test_deliver_message_records_provider_error_request_body(monkeypatch) -> None:
    """开启诊断开关时记录响应中的 payload 快照。"""
    build_record = Mock(return_value={"payload_count": 1, "truncated": False})
    append_record = AsyncMock(return_value="data/provider-error.jsonl")
    info = Mock()
    monkeypatch.setattr(chatter_module, "build_provider_error_request_record", build_record)
    monkeypatch.setattr(chatter_module, "append_provider_error_request_record", append_record)
    monkeypatch.setattr(chatter_module.logger, "info", info)
    chatter = AgenticChatter(stream_id="provider-error-on", plugin=object())
    state = TurnState(stream_id="provider-error-on")
    payloads = ["payload"]
    config = SimpleNamespace(tools=SimpleNamespace(record_provider_error_request_body=True))
    visible_text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]."
    )
    response = SimpleNamespace(message=visible_text, payloads=payloads)

    result = await chatter._deliver_message(config, response, state)

    assert result == (False, False)
    build_record.assert_called_once()
    assert build_record.call_args.args[0] is payloads
    assert build_record.call_args.kwargs["rule"] == "google_prompt_policy_block"
    assert build_record.call_args.kwargs["stream_id"] == "provider-error-on"
    append_record.assert_awaited_once_with(build_record.return_value)
    assert any("event=provider_error_request_recorded" in str(call.args[0]) for call in info.call_args_list)


async def test_deliver_message_provider_error_recording_failure_still_intercepts(
    monkeypatch,
) -> None:
    """请求体记录失败时仍不得发送供应商异常正文。"""
    send_text = AsyncMock(return_value=True)
    warning = Mock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(
        chatter_module,
        "build_provider_error_request_record",
        Mock(side_effect=OSError("write failed")),
    )
    monkeypatch.setattr(chatter_module.logger, "warning", warning)
    chatter = AgenticChatter(stream_id="provider-error-fail", plugin=object())
    state = TurnState(stream_id="provider-error-fail")
    config = SimpleNamespace(tools=SimpleNamespace(record_provider_error_request_body=True))
    visible_text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]."
    )

    result = await chatter._deliver_message(
        config,
        SimpleNamespace(message=visible_text, payloads=[]),
        state,
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    assert any("event=provider_error_request_record_failed" in str(call.args[0]) for call in warning.call_args_list)


async def test_deliver_message_allows_provider_error_discussion(monkeypatch) -> None:
    """正常讨论供应商错误的回复不应被误拦截。"""
    send_text = AsyncMock(return_value=True)
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    chatter = AgenticChatter(stream_id="provider-discussion", plugin=object())
    text = "我可以解释 content policy 和 API error 的区别。"

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=text),
        TurnState(stream_id="provider-discussion"),
    )

    assert result == (True, False)
    send_text.assert_awaited_once_with(
        content=text,
        stream_id="provider-discussion",
    )


async def test_deliver_message_intercepts_reply_decision_json(monkeypatch) -> None:
    """子决策 JSON 被主 Agent 输出时不得进入发送 API。"""
    send_text = AsyncMock(return_value=True)
    warning = Mock()
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module.logger, "warning", warning)
    chatter = AgenticChatter(stream_id="decision-json", plugin=object())
    state = TurnState(stream_id="decision-json")
    text = (
        '{"action":"silent","confidence":0.8,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":["not_directed_to_bot"],'
        '"brief_reason":"消息未指向 bot"}'
    )

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=f"<think>内部推理</think>{text}"),
        state,
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    warning.assert_called_once()
    log_line = str(warning.call_args.args[0])
    assert "event=reply_decision_json_intercepted" in log_line
    assert "action=silent" in log_line
    assert f"chars={len(text)}" in log_line
    assert text not in log_line
    assert not state.spoke
    assert not state.sent_texts
    assert state.visible_text_emissions == 0


async def test_deliver_message_allows_normal_decision_json_discussion(monkeypatch) -> None:
    """普通讨论决策 JSON 的正文不得被过度拦截。"""
    send_text = AsyncMock(return_value=True)
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    chatter = AgenticChatter(stream_id="decision-discussion", plugin=object())
    text = "字段 action、confidence 和 brief_reason 分别表示什么？"

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=text),
        TurnState(stream_id="decision-discussion"),
    )

    assert result == (True, False)
    send_text.assert_awaited_once_with(
        content=text,
        stream_id="decision-discussion",
    )


async def test_deliver_message_intercepts_complete_framework_message_line(monkeypatch) -> None:
    """模型完整复述框架消息行时不得再次发送给用户。"""
    send_text = AsyncMock(return_value=True)
    warning = Mock()
    text = "【13:47】<机器人> [6264745991384149877] 我是打工蝶😭 ："
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    monkeypatch.setattr(chatter_module.logger, "warning", warning)
    chatter = AgenticChatter(stream_id="intercept-stream", plugin=object())
    state = TurnState(stream_id="intercept-stream")

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=text),
        state,
    )

    assert result == (False, False)
    send_text.assert_not_awaited()
    warning.assert_called_once()
    log_line = str(warning.call_args.args[0])
    assert "framework_message_line_intercepted" in log_line
    assert f"chars={len(text)}" in log_line
    assert text not in log_line
    assert not state.spoke
    assert state.visible_text_emissions == 0


async def test_deliver_message_allows_normal_text_with_framework_keywords(monkeypatch) -> None:
    """普通正文包含相似关键词时仍应正常发送。"""
    send_text = AsyncMock(return_value=True)
    text = "机器人刚才的回答让我很开心。"
    monkeypatch.setattr(chatter_module.send_api, "send_text", send_text)
    chatter = AgenticChatter(stream_id="normal-framework-keywords", plugin=object())

    result = await chatter._deliver_message(
        None,
        SimpleNamespace(message=text),
        TurnState(stream_id="normal-framework-keywords"),
    )

    assert result == (True, False)
    send_text.assert_awaited_once_with(
        content=text,
        stream_id="normal-framework-keywords",
    )


def test_normalize_reply_text_ignores_spacing_and_punctuation() -> None:
    """空白和标点差异不应绕过精确复读判断。"""
    assert normalize_reply_text("你好， 世界！") == normalize_reply_text("你好世界")


def test_reply_similarity_detects_paraphrased_repetition() -> None:
    """高度重叠的改写应被视为近似复读。"""
    similarity, containment = reply_similarity(
        "建议你先重启一下服务",
        "建议你先重启一下服务看看",
    )

    assert similarity > 0.75
    assert containment == 1.0
    assert is_repeated_reply(
        "建议你先重启一下服务",
        ["建议你先重启一下服务看看"],
        similarity_threshold=0.88,
        containment_threshold=0.90,
    )


def test_reply_similarity_allows_old_text_with_new_information() -> None:
    """旧文本只是新回复前缀时，应允许发送新增信息。"""
    assert not is_repeated_reply(
        "我先查一下日志，日志显示数据库连接超时",
        ["我先查一下日志"],
        similarity_threshold=0.88,
        containment_threshold=0.90,
    )


def test_reply_similarity_keeps_new_tool_information() -> None:
    """同一话题中的新工具结果不应被当作复读。"""
    assert not is_repeated_reply(
        "日志显示数据库连接超时",
        ["我先查一下日志"],
        similarity_threshold=0.88,
        containment_threshold=0.90,
    )


def test_base_loop_does_not_stop_only_because_state_spoke() -> None:
    """基础循环不负责判断本次是否有工具，迭代级结束由 act 阶段处理。"""
    state = TurnState(stream_id="s", spoke=True, iterations=1)
    assert should_continue_loop(state, max_iterations=6)


def test_loop_stops_on_end_turn() -> None:
    state = TurnState(stream_id="s", end_turn_requested=True)
    assert not should_continue_loop(state, max_iterations=6)


def test_loop_stops_on_stop_request() -> None:
    state = TurnState(stream_id="s", stop_requested=True)
    assert not should_continue_loop(state, max_iterations=6)


def test_loop_stops_at_max_iterations() -> None:
    state = TurnState(stream_id="s", iterations=6)
    assert not should_continue_loop(state, max_iterations=6)


def test_loop_unlimited_when_max_is_zero() -> None:
    state = TurnState(stream_id="s", iterations=999)
    assert should_continue_loop(state, max_iterations=0)


# ----------------------------------------------------------------------
# 控制流拆分
# ----------------------------------------------------------------------


def test_classify_separates_control_calls() -> None:
    calls = [
        _FakeCall("tool-search", {"q": "天气"}),
        _FakeCall("action-end_turn", {"seconds": 30}),
    ]

    normal, end_seconds, stop_minutes = classify_calls(calls)

    assert [call.name for call in normal] == ["tool-search"]
    assert end_seconds == 30.0
    assert stop_minutes is None


def test_classify_handles_stop_call() -> None:
    normal, end_seconds, stop_minutes = classify_calls(
        [_FakeCall("action-stop_conversation", {"minutes": 10})]
    )

    assert normal == []
    assert end_seconds is None
    assert stop_minutes == 10.0


def test_classify_uses_defaults_on_bad_args() -> None:
    _, end_seconds, _ = classify_calls([_FakeCall("action-end_turn", {"seconds": "bad"})])
    assert end_seconds == 0.0

    _, _, stop_minutes = classify_calls(
        [_FakeCall("action-stop_conversation", {"minutes": "bad"})]
    )
    assert stop_minutes == 5.0


def test_classify_clamps_stop_minutes() -> None:
    _, _, stop_minutes = classify_calls(
        [_FakeCall("action-stop_conversation", {"minutes": 99999})]
    )
    assert stop_minutes == MAX_STOP_MINUTES


def test_classify_handles_empty_input() -> None:
    assert classify_calls([]) == ([], None, None)


# ----------------------------------------------------------------------
# 分段发送
# ----------------------------------------------------------------------


def test_build_speak_segments_from_message() -> None:
    segments = build_speak_segments("你好呀。今天怎么样？", _FakeHumanize())
    assert segments
    assert all(segment.text for segment in segments)


def test_build_speak_segments_empty_message() -> None:
    assert build_speak_segments("", _FakeHumanize()) == []
    assert build_speak_segments(None, _FakeHumanize()) == []


def test_build_speak_segments_without_config() -> None:
    segments = build_speak_segments("你好", None)
    assert len(segments) == 1


async def test_deliver_segments_sends_all() -> None:
    sent: list[str] = []

    async def speak(text: str) -> bool:
        sent.append(text)
        return True

    async def no_sleep(_seconds: float) -> None:
        return None

    segments = build_speak_segments("第一句。第二句。", _FakeHumanize())
    count = await deliver_segments(segments, speak, sleep=no_sleep)

    assert count == len(segments)
    assert len(sent) == len(segments)


async def test_deliver_segments_counts_only_successes() -> None:
    async def failing_speak(_text: str) -> bool:
        return False

    async def no_sleep(_seconds: float) -> None:
        return None

    segments = build_speak_segments("一句话", _FakeHumanize())
    assert await deliver_segments(segments, failing_speak, sleep=no_sleep) == 0


async def test_deliver_empty_segments() -> None:
    async def speak(_text: str) -> bool:
        return True

    assert await deliver_segments([], speak) == 0


# ----------------------------------------------------------------------
# 阶段编排
# ----------------------------------------------------------------------


def test_resolve_stage_order_filters_disabled() -> None:
    order = resolve_stage_order(
        [STAGE_PERCEIVE, STAGE_ACT, STAGE_REFLECT],
        enable_perceive=False,
        enable_plan=False,
        enable_reflect=True,
    )

    assert STAGE_PERCEIVE not in order
    assert order == [STAGE_ACT, STAGE_REFLECT]


def test_resolve_stage_order_always_includes_act() -> None:
    """act 是管线核心，遗漏时必须自动补上。"""
    order = resolve_stage_order(
        [STAGE_REFLECT],
        enable_perceive=False,
        enable_plan=False,
        enable_reflect=True,
    )

    assert STAGE_ACT in order


def test_resolve_stage_order_preserves_custom_order() -> None:
    order = resolve_stage_order(
        [STAGE_PLAN, STAGE_PERCEIVE, STAGE_ACT],
        enable_perceive=True,
        enable_plan=True,
        enable_reflect=False,
    )

    assert order.index(STAGE_PLAN) < order.index(STAGE_PERCEIVE)


def test_resolve_stage_order_dedupes() -> None:
    order = resolve_stage_order(
        [STAGE_ACT, STAGE_ACT, STAGE_ACT],
        enable_perceive=False,
        enable_plan=False,
        enable_reflect=False,
    )

    assert order.count(STAGE_ACT) == 1


def test_resolve_stage_order_keeps_custom_stage_names() -> None:
    order = resolve_stage_order(
        ["my_custom_stage", STAGE_ACT],
        enable_perceive=False,
        enable_plan=False,
        enable_reflect=False,
    )

    assert "my_custom_stage" in order


# ----------------------------------------------------------------------
# 回合状态
# ----------------------------------------------------------------------


def test_outcome_defaults_to_wait() -> None:
    outcome = TurnState(stream_id="s").to_outcome()
    assert outcome.should_wait
    assert not outcome.should_stop


def test_outcome_reflects_stop_request() -> None:
    state = TurnState(stream_id="s", stop_requested=True, stop_minutes=10)
    outcome = state.to_outcome()

    assert outcome.should_stop
    assert outcome.stop_seconds == 600.0


def test_outcome_carries_wait_seconds() -> None:
    state = TurnState(stream_id="s", end_turn_requested=True, end_turn_seconds=45)
    assert state.to_outcome().wait_seconds == 45.0


class _PayloadResponse:
    """收集 payload，并可在消费时填充响应内容的假响应。"""

    def __init__(
        self,
        payloads: list[LLMPayload],
        calls: list[ToolCall],
        *,
        delayed: bool = False,
        delayed_message: str = "",
    ) -> None:
        self.payloads = payloads
        self._pending_calls = calls if delayed else []
        self._pending_message = delayed_message if delayed else ""
        self.call_list = [] if delayed else calls
        self.message = None if delayed else "这条过时回复不应发送"
        self.consumed = False

    def add_payload(self, payload: LLMPayload) -> None:
        self.payloads.append(payload)

    def __await__(self):
        async def consume() -> str:
            self.consumed = True
            if self._pending_calls or self._pending_message:
                self.call_list = self._pending_calls
                self.message = self._pending_message
                content: list[Any] = []
                if self.message:
                    content.append(Text(self.message))
                content.extend(self.call_list)
                self.payloads.append(LLMPayload(ROLE.ASSISTANT, content))
                self._pending_calls = []
                self._pending_message = ""
            return self.message or ""

        return consume().__await__()


class _RequestReturningResponse:
    """返回预设响应的假请求。"""

    def __init__(self, response: _PayloadResponse) -> None:
        self.response = response
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        self.payloads.append(payload)

    async def send(self, *, stream: bool = False) -> _PayloadResponse:
        return self.response


class _InterruptConfig:
    """关闭探索工具的 act 阶段配置替身。"""

    class Plugin:
        model_task = "actor"

    class Pipeline:
        max_iterations = 1

    class Tools:
        enable_explore_tools = False
        blacklist: list[str] = []

    class Humanize:
        enable_interrupt = True

    plugin = Plugin()
    pipeline = Pipeline()
    tools = Tools()
    humanize = Humanize()


class _ToolConfig:
    """包含工具筛选配置的 act 阶段替身。"""

    class Plugin:
        model_task = "actor"

    class Pipeline:
        max_iterations = 1

    class Tools:
        enable_explore_tools = True
        blacklist: list[str] = []

    class Humanize:
        max_duplicate_streak = 2

    plugin = Plugin()
    pipeline = Pipeline()
    tools = Tools()
    humanize = Humanize()


class _AllowedExpandedTool:
    """允许在探索后注册的假工具。"""

    tool_name = "allowed"

    @classmethod
    def get_signature(cls) -> str:
        """返回测试组件签名。"""
        return "allowed_plugin:tool:allowed"

    @classmethod
    def to_schema(cls) -> dict[str, dict[str, str]]:
        """返回工具 schema。"""
        return {"function": {"name": "tool-allowed"}}


class _BlockedExpandedTool:
    """必须被黑名单阻止的假工具。"""

    tool_name = "blocked"

    @classmethod
    def get_signature(cls) -> str:
        """返回测试组件签名。"""
        return "blocked_plugin:tool:blocked"

    @classmethod
    def to_schema(cls) -> dict[str, dict[str, str]]:
        """返回工具 schema。"""
        return {"function": {"name": "tool-blocked"}}


async def test_record_streamed_duplicate_is_not_new_progress() -> None:
    """已发出的流式复读应标记重复，不得再次计入有效发言。"""
    chatter = AgenticChatter(stream_id="stream-duplicate", plugin=object())
    state = TurnState(
        stream_id="stream-duplicate",
        spoke=True,
        sent_texts=["建议你先重启一下服务"],
        visible_text_emissions=1,
    )
    cleaned = SimpleNamespace(
        text="建议你先重启一下服务！",
        removed_thoughts=(),
    )

    result = chatter._record_streamed_message(
        None,
        SimpleNamespace(reasoning_content=""),
        state,
        cleaned,
    )

    assert result == (False, True)
    assert state.sent_texts == ["建议你先重启一下服务"]
    assert state.visible_text_emissions == 1
    assert state.duplicate_text_streak == 1


async def test_stage_act_ends_after_text_without_tools() -> None:
    """纯文本回复发送后应立即结束，不再生成确认型尾句。"""
    response = _PayloadResponse([], [])
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="text-auto-end", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={},
        collapsed_classes={},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=False)
    chatter._deliver_message = AsyncMock(return_value=(True, False))
    chatter._execute_calls = AsyncMock()

    await chatter._stage_act(
        _InterruptConfig(),
        object(),
        TurnState(stream_id="text-auto-end"),
        [],
    )

    chatter._deliver_message.assert_awaited_once()
    chatter._execute_calls.assert_not_awaited()
    assert not any(payload.role == ROLE.USER for payload in response.payloads)


async def test_stage_act_continues_when_text_has_tool_call() -> None:
    """文本和普通工具同轮出现时应执行工具并保留后续迭代能力。"""
    config = _ToolConfig()
    config.pipeline.max_iterations = 1
    calls = [ToolCall(id="call_tool", name="tool-search", args={"q": "天气"})]
    response = _PayloadResponse([], calls)
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="text-with-tool", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={},
        collapsed_classes={},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=False)
    chatter._deliver_message = AsyncMock(return_value=(True, False))
    chatter._execute_calls = AsyncMock(return_value=1)

    await chatter._stage_act(
        config,
        object(),
        TurnState(stream_id="text-with-tool"),
        [],
    )

    chatter._execute_calls.assert_awaited_once()


async def test_execute_calls_counts_only_successful_tools() -> None:
    """失败工具不得被当作有效进展，并应记录结果账本。"""
    calls = [
        ToolCall(id="success", name="tool-success", args={}),
        ToolCall(id="failure", name="tool-failure", args={}),
    ]
    response = _PayloadResponse([], calls)

    async def run_tool_call(
        batch: list[ToolCall], *_args: Any
    ) -> list[tuple[str, bool]]:
        call = batch[0]
        if call.id == "success":
            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(value="晴天", call_id="success", name="tool-success"),
                )
            )
            return [("ok", True)]
        response.add_payload(
            LLMPayload(
                ROLE.TOOL_RESULT,
                ToolResult(value="查询失败", call_id="failure", name="tool-failure"),
            )
        )
        return [("failed", False)]

    chatter = AgenticChatter(stream_id="tool-progress", plugin=object())
    chatter.run_tool_call = AsyncMock(side_effect=run_tool_call)
    state = TurnState(stream_id="tool-progress")

    count = await chatter._execute_calls(
        calls,
        response,
        state,
        SimpleNamespace(),
        [],
    )

    assert count == 1
    assert [record.outcome for record in state.tool_ledger] == ["success", "failure"]
    assert [record.result_preview for record in state.tool_ledger] == ["晴天", "查询失败"]
    assert "晴天" in state.deduper.check("tool-success", {}).note


async def test_execute_calls_batch_mode_submits_all_calls_once() -> None:
    calls = [
        ToolCall(id="first", name="tool-first", args={}),
        ToolCall(id="second", name="tool-second", args={}),
    ]
    response = _PayloadResponse([], calls)
    chatter = AgenticChatter(stream_id="tool-batch", plugin=object())
    chatter.run_tool_call = AsyncMock(return_value=[("ok", True), ("ok", True)])
    state = TurnState(stream_id="tool-batch")

    count = await chatter._execute_calls(
        calls,
        response,
        state,
        SimpleNamespace(),
        [],
        tool_call_mode="batch",
    )

    chatter.run_tool_call.assert_awaited_once()
    submitted = chatter.run_tool_call.await_args.args[0]
    assert submitted == calls
    assert count == 2


async def test_execute_calls_planning_mode_submits_calls_one_by_one() -> None:
    calls = [
        ToolCall(id="first", name="tool-first", args={}),
        ToolCall(id="second", name="tool-second", args={}),
    ]
    response = _PayloadResponse([], calls)
    chatter = AgenticChatter(stream_id="tool-planning", plugin=object())
    chatter.run_tool_call = AsyncMock(side_effect=[[('ok', True)], [('ok', True)]])
    state = TurnState(stream_id="tool-planning")

    count = await chatter._execute_calls(
        calls,
        response,
        state,
        SimpleNamespace(),
        [],
        tool_call_mode="planning",
    )

    assert chatter.run_tool_call.await_count == 2
    assert [call.args[0] for call in chatter.run_tool_call.await_args_list] == [
        [calls[0]],
        [calls[1]],
    ]
    assert count == 2


async def test_execute_calls_invalid_mode_falls_back_to_planning() -> None:
    call = ToolCall(id="fallback", name="tool-fallback", args={})
    response = _PayloadResponse([], [call])
    chatter = AgenticChatter(stream_id="tool-mode-fallback", plugin=object())
    chatter.run_tool_call = AsyncMock(return_value=[("ok", True)])

    await chatter._execute_calls(
        [call],
        response,
        TurnState(stream_id="tool-mode-fallback"),
        SimpleNamespace(),
        [],
        tool_call_mode="unknown",
    )

    chatter.run_tool_call.assert_awaited_once_with(
        [call], response, ANY, None,
    )


async def test_execute_calls_logs_tool_name_and_safe_args(monkeypatch) -> None:
    calls = [
        ToolCall(
            id="logged",
            name="tool-search",
            args={"query": "天气", "api_key": "secret-value"},
        )
    ]
    response = _PayloadResponse([], calls)
    chatter = AgenticChatter(stream_id="tool-log", plugin=object())
    chatter.run_tool_call = AsyncMock(return_value=[("ok", True)])
    state = TurnState(stream_id="tool-log", iterations=2)
    info = Mock()
    monkeypatch.setattr(chatter_module.logger, "info", info)

    await chatter._execute_calls(
        calls,
        response,
        state,
        SimpleNamespace(),
        [],
        log_tool_calls=True,
    )

    logs = [str(call.args[0]) for call in info.call_args_list]
    tool_logs = [line for line in logs if "event=tool_call" in line]
    assert len(tool_logs) == 1
    assert "调用工具" in tool_logs[0]
    assert "name=tool-search" in tool_logs[0]
    assert '"query":"天气"' in tool_logs[0]
    assert "secret-value" not in tool_logs[0]
    assert "[REDACTED]" in tool_logs[0]


async def test_execute_calls_does_not_log_when_disabled(monkeypatch) -> None:
    call = ToolCall(id="quiet", name="tool-search", args={"query": "天气"})
    response = _PayloadResponse([], [call])
    chatter = AgenticChatter(stream_id="tool-log-off", plugin=object())
    chatter.run_tool_call = AsyncMock(return_value=[("ok", True)])
    info = Mock()
    monkeypatch.setattr(chatter_module.logger, "info", info)

    await chatter._execute_calls(
        [call], response, TurnState(stream_id="tool-log-off"), SimpleNamespace(), [],
    )

    assert not any("event=tool_call" in str(call.args[0]) for call in info.call_args_list)


def test_decide_iteration_preserves_control_and_text_priority() -> None:
    stop = decide_iteration(
        normal_call_count=1,
        end_seconds=2.0,
        stop_minutes=3.0,
        spoke_now=True,
        duplicate_text=False,
        tool_progress=1,
        no_progress_iterations=0,
        post_speech_iterations=0,
        duplicate_text_streak=0,
        max_no_progress=2,
        max_post_speech=3,
        max_duplicate_streak=2,
    )
    assert stop.reason == TerminationReason.STOP_REQUESTED
    assert stop.stop_seconds == 180.0

    text = decide_iteration(
        normal_call_count=0,
        end_seconds=None,
        stop_minutes=None,
        spoke_now=True,
        duplicate_text=False,
        tool_progress=0,
        no_progress_iterations=1,
        post_speech_iterations=0,
        duplicate_text_streak=0,
        max_no_progress=1,
        max_post_speech=1,
        max_duplicate_streak=1,
    )
    assert text.reason == TerminationReason.TEXT_WITHOUT_TOOL


def test_tool_result_delta_matches_calls_by_id() -> None:
    calls = [
        ToolCall(id="first", name="tool-a", args={}),
        ToolCall(id="second", name="tool-b", args={}),
    ]
    response = _PayloadResponse(
        [LLMPayload(ROLE.USER, Text("before"))],
        calls,
    )
    start = len(response.payloads)
    response.add_payload(
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="B", call_id="second", name="tool-b"),
        )
    )
    response.add_payload(
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="A", call_id="first", name="tool-a"),
        )
    )

    captured = collect_tool_result_delta(response, start)
    records = build_tool_execution_records(
        calls,
        [("a", True), ("b", True)],
        captured,
        iteration=2,
    )

    assert [record.result_preview for record in records] == ["A", "B"]
    assert all(record.result_capture == "captured" for record in records)


async def test_stage_act_does_not_inject_blacklisted_explore_tool() -> None:
    """探索工具本身命中黑名单时不得被重新注入。"""
    config = _ToolConfig()
    config.tools.blacklist = ["agentic_chatter:tool:explore_tools"]
    response = _PayloadResponse([], [])
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="blocked-explore", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={"platform": ["get_info"]},
        collapsed_classes={"platform": [_AllowedExpandedTool]},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=False)
    chatter._deliver_message = AsyncMock(return_value=(False, False))

    state = TurnState(stream_id="blocked-explore")
    await chatter._stage_act(config, object(), state, [])

    tool_payload = next(payload for payload in request.payloads if payload.role == ROLE.TOOL)
    assert tool_payload.content == []


async def test_stage_act_filters_blacklisted_expanded_tools(monkeypatch) -> None:
    """展开结果在注册前仍必须遵守黑名单。"""
    config = _ToolConfig()
    config.tools.blacklist = ["blocked_plugin:tool:*"]
    calls = [ToolCall(id="call_expand", name="tool-explore_tools", args={})]
    response = _PayloadResponse([], calls)
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="expanded-filter", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={},
        collapsed_classes={},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=False)
    chatter._deliver_message = AsyncMock(return_value=(False, False))
    chatter._execute_calls = AsyncMock(return_value=1)
    monkeypatch.setattr(
        chatter_module,
        "consume_expansion",
        lambda _stream_id: [_AllowedExpandedTool, _BlockedExpandedTool],
    )

    await chatter._stage_act(config, object(), TurnState(stream_id="expanded-filter"), [])

    tool_payloads = [payload for payload in response.payloads if payload.role == ROLE.TOOL]
    assert tool_payloads[-1].content == [_AllowedExpandedTool]


# ----------------------------------------------------------------------
# 新消息中断时的上下文闭合
# ----------------------------------------------------------------------


def test_interrupted_calls_are_closed_for_strict_context_validation() -> None:
    calls = [
        ToolCall(id="call_tool", name="tool-search", args={"q": "天气"}),
        ToolCall(id="call_end", name="action-end_turn", args={"seconds": 30}),
        ToolCall(id="call_stop", name="action-stop_conversation", args={"minutes": 5}),
    ]
    payloads = [
        LLMPayload(ROLE.USER, Text("查一下天气")),
        LLMPayload(ROLE.ASSISTANT, calls),
    ]
    response = _PayloadResponse(payloads, calls)

    append_interrupted_tool_results(response, calls)

    validate_payload_sequence(response.payloads, allow_incomplete_tail=False)
    results = [
        part
        for payload in response.payloads
        if payload.role == ROLE.TOOL_RESULT
        for part in payload.content
        if isinstance(part, ToolResult)
    ]
    assert [result.call_id for result in results] == [call.id for call in calls]
    assert [result.name for result in results] == [call.name for call in calls]
    assert all("未执行" in result.value for result in results)


def test_control_calls_are_closed_for_strict_context_validation() -> None:
    calls = [
        ToolCall(id="call_end", name="action-end_turn", args={"seconds": 30}),
        ToolCall(id="call_stop", name="action-stop_conversation", args={"minutes": 5}),
    ]
    response = _PayloadResponse(
        [
            LLMPayload(ROLE.USER, Text("结束本轮")),
            LLMPayload(ROLE.ASSISTANT, calls),
        ],
        calls,
    )

    append_control_tool_results(response, calls)

    validate_payload_sequence(response.payloads, allow_incomplete_tail=False)
    results = [
        part
        for payload in response.payloads
        if payload.role == ROLE.TOOL_RESULT
        for part in payload.content
        if isinstance(part, ToolResult)
    ]
    assert [result.call_id for result in results] == ["call_end", "call_stop"]
    assert "覆盖" in results[0].value
    assert "冷却" in results[1].value


async def test_stage_act_consumes_stream_before_reading_calls() -> None:
    calls = [
        ToolCall(id="call_end", name="action-end_turn", args={"seconds": 30}),
    ]
    response = _PayloadResponse(
        [LLMPayload(ROLE.USER, Text("结束本轮"))],
        calls,
        delayed=True,
        delayed_message="流式文本",
    )
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="stream", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={},
        collapsed_classes={},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=False)
    chatter._deliver_message = AsyncMock(return_value=(False, False))
    chatter._execute_calls = AsyncMock()
    state = TurnState(stream_id="stream")

    await chatter._stage_act(_InterruptConfig(), object(), state, [])

    assert response.consumed
    assert response.message == "流式文本"
    assert state.end_turn_requested
    assert state.end_turn_seconds == 30.0
    chatter._execute_calls.assert_not_awaited()
    validate_payload_sequence(response.payloads, allow_incomplete_tail=False)


async def test_flush_claim_unreads_supports_rehydrated_idless_message(
    monkeypatch,
) -> None:
    """无 ID claim 应按稳定键从当前未读快照移入 history。"""
    claimed = SimpleNamespace(
        message_id="",
        stream_id="flush-idless",
        time=123.0,
        sender_id="user",
        sender_name="用户",
        message_type="text",
        reply_to=None,
        content="无 ID 消息",
        processed_plain_text="无 ID 消息",
    )
    current = SimpleNamespace(**vars(claimed))
    context = SimpleNamespace(unread_messages=[current], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="flush-idless", plugin=object())

    flushed = await chatter._flush_claim_unreads([claimed])

    assert flushed == 1
    assert context.history == [current]
    assert context.unread_messages == []


async def test_flush_claim_unreads_missing_messages_has_no_side_effect(monkeypatch) -> None:
    """claim 缺失时不得部分搬运当前上下文。"""
    claimed = SimpleNamespace(message_id="missing")
    current = SimpleNamespace(message_id="present")
    context = SimpleNamespace(unread_messages=[current], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="flush-missing", plugin=object())

    flushed = await chatter._flush_claim_unreads([claimed])

    assert flushed == 0
    assert context.unread_messages == [current]
    assert context.history == []


async def test_flush_claim_unreads_preserves_new_message_added_after_match(
    monkeypatch,
) -> None:
    """预检后新到的 unread 不得被旧 remained 快照覆盖。"""
    claimed = SimpleNamespace(message_id="claimed")
    matched = SimpleNamespace(message_id="claimed")
    new = SimpleNamespace(message_id="new")
    context = SimpleNamespace(unread_messages=[matched], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="flush-new", plugin=object())

    match = await chatter._match_claim_unreads([claimed])
    assert match is not None
    context.unread_messages.append(new)
    chatter._apply_claim_unread_match(match)

    assert context.history == [matched]
    assert context.unread_messages == [new]


async def test_flush_claim_unreads_removes_only_claimed_duplicate_reference(
    monkeypatch,
) -> None:
    """同一对象重复出现时只确认与 claim 数量一致的那一条。"""
    message = SimpleNamespace(message_id="duplicated")
    context = SimpleNamespace(unread_messages=[message, message], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="flush-duplicate-reference", plugin=object())

    flushed = await chatter._flush_claim_unreads([message])

    assert flushed == 1
    assert context.history == [message]
    assert context.unread_messages == [message]


async def test_apply_claim_unread_match_restores_context_when_match_disappears(
    monkeypatch,
) -> None:
    """提交前 claim 对应 unread 消失时恢复完整上下文。"""
    claimed = SimpleNamespace(message_id="claimed")
    current = SimpleNamespace(message_id="claimed")
    context = SimpleNamespace(unread_messages=[current], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="flush-disappeared", plugin=object())
    match = await chatter._match_claim_unreads([claimed])

    assert match is not None
    context.unread_messages.clear()
    try:
        chatter._apply_claim_unread_match(match)
    except RuntimeError as exc:
        assert str(exc) == "未读消息在提交前发生变化"
    else:
        raise AssertionError("提交前缺失匹配消息应中止确认")

    assert context.unread_messages == []
    assert context.history == []


async def test_mailbox_commit_failure_restores_context_match(monkeypatch) -> None:
    """mailbox commit 失败时恢复已应用的上下文确认。"""
    claimed = SimpleNamespace(message_id="claimed")
    current = SimpleNamespace(message_id="claimed")
    context = SimpleNamespace(unread_messages=[current], history=[])
    context.add_history_message = context.history.append
    monkeypatch.setattr(
        chatter_module.stream_api,
        "get_stream",
        AsyncMock(return_value=SimpleNamespace(context=context)),
    )
    chatter = AgenticChatter(stream_id="commit-failure", plugin=object())
    match = await chatter._match_claim_unreads([claimed])

    assert match is not None
    chatter._apply_claim_unread_match(match)
    assert context.unread_messages == []
    assert context.history == [current]

    chatter._restore_claim_unread_match(match)
    assert context.unread_messages == [current]
    assert context.history == []


async def test_mailbox_interrupt_ignores_claim_and_keeps_new_input(monkeypatch) -> None:
    """claim 重现不算新输入，新 ID 应进入 pending 并触发中断。"""
    old = SimpleNamespace(message_id="old")
    repeated_old = SimpleNamespace(message_id="old")
    new = SimpleNamespace(message_id="new")
    mailbox = StreamMailbox("mailbox-interrupt")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([old])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None

    chatter = AgenticChatter(stream_id="mailbox-interrupt", plugin=object())
    chatter.fetch_unreads = AsyncMock(return_value=("", [repeated_old]))
    monkeypatch.setattr(chatter_module, "get_stream_mailbox", Mock(return_value=mailbox))

    assert not await chatter._has_new_unreads(
        _InterruptConfig(),
        [old],
        claim=claim,
    )

    chatter.fetch_unreads = AsyncMock(return_value=("", [repeated_old, new]))
    assert await chatter._has_new_unreads(
        _InterruptConfig(),
        [old],
        claim=claim,
    )
    assert await mailbox.pending_keys() == ("id:new",)


async def test_stage_act_closes_calls_before_interrupting() -> None:
    calls = [
        ToolCall(id="call_tool", name="tool-search", args={"q": "天气"}),
        ToolCall(id="call_end", name="action-end_turn", args={"seconds": 30}),
        ToolCall(id="call_stop", name="action-stop_conversation", args={"minutes": 5}),
    ]
    response = _PayloadResponse(
        [
            LLMPayload(ROLE.USER, Text("旧消息")),
            LLMPayload(ROLE.ASSISTANT, calls),
        ],
        calls,
    )
    request = _RequestReturningResponse(response)
    chatter = AgenticChatter(stream_id="stream", plugin=object())
    chatter.get_llm_usables = AsyncMock(return_value=[])
    chatter.modify_llm_usables = AsyncMock(return_value=[])
    chatter._build_layout = lambda _config, _usables: SimpleNamespace(
        exposed=[],
        collapsed_categories={},
        collapsed_classes={},
    )
    chatter._build_system_prompt = AsyncMock(return_value="system")
    chatter._build_user_prompt = AsyncMock(return_value="user")
    chatter.create_request = lambda **_kwargs: request
    chatter._has_new_unreads = AsyncMock(return_value=True)
    chatter._deliver_message = AsyncMock(return_value=(True, False))
    chatter._execute_calls = AsyncMock()
    state = TurnState(stream_id="stream")

    await chatter._stage_act(_InterruptConfig(), object(), state, [])

    validate_payload_sequence(response.payloads, allow_incomplete_tail=False)
    chatter._deliver_message.assert_not_awaited()
    chatter._execute_calls.assert_not_awaited()
    assert state.failed
    assert state.error == "生成期间收到新消息，重新规划"
    assert not state.end_turn_requested
    assert not state.stop_requested

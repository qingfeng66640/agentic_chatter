"""Agent 主循环测试。

最重要的是 ``test_loop_continues_after_speaking``：它守护本插件
对 DFC 最关键的修复 —— 说完话之后循环不应终止，模型必须还有
机会继续调用工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from src.kernel.llm import LLMPayload, ROLE, Text, ToolCall, ToolResult
from src.kernel.llm.context_structure import validate_payload_sequence

from ..actions.control import MAX_STOP_MINUTES
from .. import chatter as chatter_module
from ..chatter import AgenticChatter, _thought_log_line
from ..pipeline.loop import (
    append_control_tool_results,
    append_interrupted_tool_results,
    build_speak_segments,
    classify_calls,
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
from ..pipeline.state import TurnState


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


def test_loop_continues_after_speaking() -> None:
    """核心回归测试：说完话之后必须还能继续调工具。

    DFC 在这里会直接挂起（session.py:729-750 注入 __SUSPEND__），
    导致模型没有机会在说完话后查证或记录。本插件必须继续循环。
    """
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

    plugin = Plugin()
    pipeline = Pipeline()
    tools = Tools()


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

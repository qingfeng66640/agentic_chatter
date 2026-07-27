"""Agent 主循环测试。

最重要的是 ``test_loop_continues_after_speaking``：它守护本插件
对 DFC 最关键的修复 —— 说完话之后循环不应终止，模型必须还有
机会继续调用工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..actions.control import MAX_STOP_MINUTES
from ..pipeline.loop import (
    build_speak_segments,
    classify_calls,
    deliver_segments,
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


# ----------------------------------------------------------------------
# 循环终止条件
# ----------------------------------------------------------------------


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

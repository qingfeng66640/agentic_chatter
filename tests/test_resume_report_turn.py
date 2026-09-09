"""resume 事件透传与任务汇报轮的测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ..task_runtime import coordinator
from ..task_runtime.models import TaskStatus
from ..task_runtime.templates import build_task_report_prompt
from .test_task_completion_report import _make_runtime  # 复用 runtime 构造

# ---------------------------------------------------------------------------
# build_task_report_prompt
# ---------------------------------------------------------------------------


def test_report_prompt_contains_status_and_summary() -> None:
    events = (
        {
            "task_id": "t1",
            "task_type": "research",
            "status": "succeeded",
            "summary": "已查到三篇资料",
            "error": "",
        },
        {
            "task_id": "t2",
            "task_type": "general",
            "status": "waiting_user",
            "summary": "",
            "error": "",
        },
    )
    prompt = build_task_report_prompt(events)
    assert "research" in prompt
    assert "succeeded" in prompt
    assert "已查到三篇资料" in prompt
    assert "waiting_user" in prompt
    assert "补充任务" in prompt


def test_report_prompt_empty_events_renders_header_only() -> None:
    prompt = build_task_report_prompt(())
    assert "sub-agent" in prompt


# ---------------------------------------------------------------------------
# _is_task_resume_event
# ---------------------------------------------------------------------------


class _Event:
    def __init__(self, source: str) -> None:
        self.source = source


def test_is_task_resume_event_matches_sub_agent() -> None:
    from ..chatter import _is_task_resume_event

    assert _is_task_resume_event(_Event("sub_agent")) is True
    assert _is_task_resume_event(_Event("message")) is False
    assert _is_task_resume_event(_Event("timer")) is False
    assert _is_task_resume_event(None) is False


# ---------------------------------------------------------------------------
# execute() 透传 asend 值
# ---------------------------------------------------------------------------


class _ProbeChatter:
    """最小 chatter 替身：记录 _run_turns 收到的 asend 值。"""

    def __init__(self) -> None:
        self.stream_id = "probe-stream"
        self.received: list[Any] = []
        self.enabled = False

    @property
    def cfg(self) -> Any:
        return SimpleNamespace(plugin=SimpleNamespace(enabled=self.enabled))

    async def _run_turns(self, config: Any) -> Any:
        # 首次 yield 后接收 asend 值，共收 3 个后结束
        received = yield "result-0"
        self.received.append(received)
        received = yield "result-1"
        self.received.append(received)
        received = yield "result-2"
        self.received.append(received)


@pytest.mark.asyncio
async def test_execute_forwards_asend_values() -> None:
    from ..chatter import _AgenticChatterBase

    probe = _ProbeChatter()
    probe.enabled = True
    event_a = _Event("message")
    event_b = _Event("sub_agent")

    gen = _AgenticChatterBase.execute(probe)
    first = await gen.asend(None)
    assert first == "result-0"
    second = await gen.asend(event_a)
    assert second == "result-1"
    third = await gen.asend(event_b)
    assert third == "result-2"
    await gen.aclose()

    # 框架 asend 进来的事件必须逐次透传到 _run_turns
    # （第三次 asend 的接收发生在 yield "result-2" 挂起期间，
    # aclose 之前替身已记录 event_b；第三个值随 GeneratorExit 丢失）
    assert probe.received == [event_a, event_b]


# ---------------------------------------------------------------------------
# 汇报轮消费（源码级验证 _run_turns 的分支与 _run_report_turn 行为）
# ---------------------------------------------------------------------------


def test_run_turns_has_report_branch() -> None:
    from ..chatter import _AgenticChatterBase

    import inspect

    source = inspect.getsource(_AgenticChatterBase._run_turns)
    assert "_is_task_resume_event(resume_event)" in source
    assert "drain_completed_events" in source
    assert "_run_report_turn" in source
    assert "resume_event = yield turn_result" in source


def test_run_report_turn_no_claim_no_commit() -> None:
    import inspect

    from ..chatter import _AgenticChatterBase

    source = inspect.getsource(_AgenticChatterBase._run_report_turn)
    # 不取 claim、不 commit：不应出现 claim_pending/commit_claim 调用
    assert "claim_pending" not in source
    assert "commit_claim" not in source
    assert "task_report=build_task_report_prompt(events)" in source


def test_input_confirmed_moved_after_commit() -> None:
    """P4：任务路由分支不再提前置位 input_confirmed。"""
    import inspect

    from ..chatter import _AgenticChatterBase

    run_turns = inspect.getsource(_AgenticChatterBase._run_turns)
    # 路由成功后到 commit 之前不得有 input_confirmed 置位
    route_index = run_turns.index("_route_task_message")
    commit_index = run_turns.index("_commit_task_turn")
    segment = run_turns[route_index:commit_index]
    assert "state.input_confirmed = True" not in segment


# ---------------------------------------------------------------------------
# task_report 注入 user prompt extra_parts
# ---------------------------------------------------------------------------


def test_turn_state_has_task_report_field() -> None:
    from ..pipeline.state import TurnState

    state = TurnState(stream_id="s", unread_texts="", deduper=None)
    assert state.task_report == ""


def test_user_prompt_appends_task_report(monkeypatch) -> None:
    from .. import chatter as chatter_module
    from ..chatter import _AgenticChatterBase
    from ..pipeline.state import TurnState

    captured: dict[str, str] = {}

    class _Template:
        def set(self, key: str, value: str) -> "_Template":
            captured[key] = value
            return self

        async def build(self) -> str:
            return captured.get("extra", "")

    class _PM:
        def get_template(self, name: str) -> Any:
            return _Template() if name == "agentic_chatter_user" else None

    chatter = object.__new__(_AgenticChatterBase)
    chatter.stream_id = "extra-stream"
    state = TurnState(
        stream_id="extra-stream",
        unread_texts="用户消息",
        deduper=None,
        task_report="汇报内容",
    )

    class SimpleStream:
        stream_name = "s"
        platform = "p"
        bot_nickname = "n"
        bot_id = "b"
        chat_type = "private"
        context = type("C", (), {"history_messages": []})()

    monkeypatch.setattr(
        chatter_module,
        "get_prompt_manager",
        lambda: _PM(),
    )
    result = asyncio.run(chatter._build_user_prompt(None, SimpleStream(), state, []))
    assert "汇报内容" in result


# ---------------------------------------------------------------------------
# 防多重回复：push 幂等 + drain 单次消费（组合验证）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_push_and_single_drain(monkeypatch) -> None:
    # 手动打桩 resume_chatter（fixture 不跨测试文件共享）
    import src.core.managers.chatter_manager as cm

    calls: list[tuple[str, str]] = []

    class _Manager:
        async def resume_chatter(self, stream_id: str, *, source: str = "") -> None:
            calls.append((stream_id, source))

    monkeypatch.setattr(cm, "get_chatter_manager", lambda: _Manager())
    runtime = _make_runtime("stream-m", TaskStatus.WAITING_USER)
    request = _make_request()
    await coordinator._notify_completion(object(), runtime, request, runtime.state.result)
    # 模拟重复 push（幂等）
    from ..task_runtime.coordinator import push_completed_event

    push_completed_event(
        "stream-m",
        {
            "task_id": runtime.state.task_id,
            "task_type": "general",
            "status": "waiting_user",
            "summary": "",
            "error": "",
        },
    )
    events = coordinator.drain_completed_events("stream-m")
    assert len(events) == 1
    assert calls == [("stream-m", "sub_agent")]


def _make_request() -> Any:
    from ..task_runtime.executor import TaskRequest

    return TaskRequest(objective="目标", deliver_final_text=True, report_events=False)

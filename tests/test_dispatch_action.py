"""dispatch_task 动作测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ..actions import dispatch as dispatch_module
from ..actions.dispatch import DispatchTaskAction, _resolve_task_type
from ..task_runtime import coordinator as coordinator_module
from ..task_runtime import (
    TaskResult,
    TaskRuntime,
    TaskState,
    TaskStatus,
    TaskType,
    get_task_runtime_manager,
)


def _make_action(stream_id: str = "dispatch-stream") -> tuple[DispatchTaskAction, object]:
    stream = SimpleNamespace(stream_id=stream_id, context=None)
    plugin = SimpleNamespace(config=None)
    action = DispatchTaskAction(stream, plugin)
    action._get_last_context_message = lambda: None  # type: ignore[method-assign]
    return action, plugin


def test_resolve_task_type_falls_back_to_general() -> None:
    assert _resolve_task_type("research") == TaskType.RESEARCH
    assert _resolve_task_type("code_change") == TaskType.CODE_CHANGE
    assert _resolve_task_type("未知类型") == TaskType.GENERAL
    assert _resolve_task_type("") == TaskType.GENERAL


@pytest.mark.asyncio
async def test_execute_rejects_empty_objective() -> None:
    action, _plugin = _make_action()
    ok, text = await action.execute(objective="   ")
    assert ok is False
    assert "不能为空" in text


@pytest.mark.asyncio
async def test_execute_rejects_when_disabled(monkeypatch) -> None:
    action, plugin = _make_action()
    plugin.config = SimpleNamespace(
        tasks=SimpleNamespace(action_enabled=False, denied_tools=(), default_allowed_tools=())
    )
    ok, text = await action.execute(objective="整理资料")
    assert ok is False
    assert "未启用" in text


@pytest.mark.asyncio
async def test_execute_rejects_conflicting_active_task() -> None:
    action, _plugin = _make_action()
    manager = get_task_runtime_manager()
    conflict = TaskRuntime(TaskState(stream_id="dispatch-stream", user_goal="已有任务"))
    conflict.start()
    manager.register(conflict)
    try:
        ok, text = await action.execute(objective="新任务")
        assert ok is False
        assert "已有活动任务" in text
    finally:
        manager.remove(conflict.state.task_id)


@pytest.mark.asyncio
async def test_execute_returns_success_with_summary(monkeypatch) -> None:
    action, _plugin = _make_action()
    captured: dict[str, object] = {}
    created_runtimes: list[TaskRuntime] = []

    class _FakeExecutor:
        def __init__(self, chatter: object, runtime: TaskRuntime, store: object = None) -> None:
            captured["runtime"] = runtime

        async def run(self, request: object) -> TaskResult:
            captured["request"] = request
            return TaskResult(TaskStatus.SUCCEEDED, "调研完成：共 3 条结论")

    monkeypatch.setattr(coordinator_module, "TaskExecutor", _FakeExecutor)
    monkeypatch.setattr(
        dispatch_module,
        "AgenticChatter",
        lambda stream_id, plugin: object(),
    )

    manager = get_task_runtime_manager()
    ok, text = await action.execute(objective="调研某主题", task_type="research")
    runtime: TaskRuntime = captured["runtime"]
    created_runtimes.append(runtime)
    try:
        assert ok is True
        assert "成功" in text
        assert "调研完成" in text
        request = captured["request"]
        assert getattr(request, "deliver_final_text") is False
        assert runtime.state.task_type == TaskType.RESEARCH
    finally:
        manager.remove(runtime.state.task_id)


@pytest.mark.asyncio
async def test_execute_reports_timeout(monkeypatch) -> None:
    action, _plugin = _make_action()

    class _HangingExecutor:
        def __init__(self, chatter: object, runtime: TaskRuntime, store: object = None) -> None:
            self.runtime = runtime

        async def run(self, request: object) -> TaskResult:
            await asyncio.sleep(999)

    monkeypatch.setattr(coordinator_module, "TaskExecutor", _HangingExecutor)
    monkeypatch.setattr(
        dispatch_module,
        "AgenticChatter",
        lambda stream_id, plugin: object(),
    )
    monkeypatch.setattr(
        "plugins.agentic_chatter.actions.dispatch._TIMEOUT_GRACE_SECONDS",
        0.05,
    )

    manager = get_task_runtime_manager()
    ok, text = await action.execute(objective="会挂起的任务")
    assert ok is False
    assert "超时" in text
    for task_id in list(manager._tasks):
        if manager._tasks[task_id].state.stream_id == "dispatch-stream":
            manager.remove(task_id)


@pytest.mark.asyncio
async def test_execute_returns_paused_result_with_active_runtime(monkeypatch) -> None:
    action, _plugin = _make_action()

    class _PausedExecutor:
        def __init__(self, chatter: object, runtime: TaskRuntime, store: object = None) -> None:
            self.runtime = runtime

        async def run(self, request: object) -> TaskResult:
            return self.runtime.pause()

    monkeypatch.setattr(coordinator_module, "TaskExecutor", _PausedExecutor)
    monkeypatch.setattr(
        dispatch_module,
        "AgenticChatter",
        lambda stream_id, plugin: object(),
    )

    manager = get_task_runtime_manager()
    ok, text = await action.execute(objective="预算耗尽的任务")
    assert ok is True
    assert "已暂停" in text
    active = manager.get_active("dispatch-stream")
    assert active is not None
    assert active.state.status == TaskStatus.PAUSED
    manager.remove(active.state.task_id)

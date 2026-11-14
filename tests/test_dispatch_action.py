"""dispatch_task 动作测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ..actions import dispatch as dispatch_module
from ..actions.dispatch import DispatchTaskAction, _resolve_task_type
from ..task_runtime import (
    TaskBudget,
    TaskResult,
    TaskRuntime,
    TaskState,
    TaskStatus,
    TaskType,
    get_task_runtime_manager,
)
from ..task_runtime import coordinator as coordinator_module


def _make_action(
    stream_id: str = "dispatch-stream",
) -> tuple[DispatchTaskAction, SimpleNamespace]:
    """使用最小聊天流和插件替身构造无上下文消息的派发动作。

    Args:
        stream_id: 派发动作所属的聊天流标识。

    Returns:
        派发动作及可用于调整配置的插件替身。
    """
    stream = SimpleNamespace(stream_id=stream_id, context=None)
    plugin = SimpleNamespace(config=None)
    action = DispatchTaskAction(stream, plugin)
    action._get_last_context_message = lambda: None  # type: ignore[method-assign]
    return action, plugin


def test_resolve_task_type_falls_back_to_general() -> None:
    """已知类型应正确解析，未知或空类型应回退为通用任务。"""
    assert _resolve_task_type("research") == TaskType.RESEARCH
    assert _resolve_task_type("code_change") == TaskType.CODE_CHANGE
    assert _resolve_task_type("未知类型") == TaskType.GENERAL
    assert _resolve_task_type("") == TaskType.GENERAL


@pytest.mark.asyncio
async def test_execute_rejects_empty_objective() -> None:
    """只有空白字符的目标应被拒绝并提示目标不能为空。"""
    action, _plugin = _make_action()
    ok, text = await action.execute(objective="   ")
    assert ok is False
    assert "不能为空" in text


@pytest.mark.asyncio
async def test_execute_rejects_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置禁用派发动作时，应拒绝任务并提示尚未启用。"""
    action, plugin = _make_action()
    plugin.config = SimpleNamespace(
        tasks=SimpleNamespace(
            action_enabled=False, denied_tools=(), default_allowed_tools=()
        )
    )
    ok, text = await action.execute(objective="整理资料")
    assert ok is False
    assert "未启用" in text


@pytest.mark.asyncio
async def test_execute_rejects_conflicting_active_task() -> None:
    """同一聊天流的默认并发额度已满时，应拒绝新任务。"""
    action, _plugin = _make_action()
    manager = get_task_runtime_manager()
    conflicts = []
    try:
        # 默认并发上限为 2，占满后 dispatch 拒绝新建
        for index in range(2):
            conflict = TaskRuntime(
                TaskState(stream_id="dispatch-stream", user_goal=f"已有任务{index}")
            )
            conflict.start()
            manager.register(conflict)
            conflicts.append(conflict)
        ok, text = await action.execute(objective="新任务")
        assert ok is False
        assert "任务在运行" in text
    finally:
        for conflict in conflicts:
            manager.remove(conflict.state.task_id)


@pytest.mark.asyncio
async def test_execute_returns_success_with_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功派发应返回结果摘要、保留任务类型并禁止执行器重复发送最终文本。"""
    action, _plugin = _make_action()
    captured: dict[str, object] = {}
    created_runtimes: list[TaskRuntime] = []

    class _FakeExecutor:
        """捕获运行时与请求，并返回固定的成功调研结果。"""

        def __init__(
            self, chatter: object, runtime: TaskRuntime, store: object = None
        ) -> None:
            """捕获派发创建的运行时，供用例检查任务类型。

            Args:
                chatter: 为兼容执行器构造接口而接收的聊天器。
                runtime: 待记录的任务运行时。
                store: 为兼容执行器构造接口而接收的存储对象。
            """
            captured["runtime"] = runtime

        async def run(self, request: object) -> TaskResult:
            """记录请求并返回固定的成功结果。

            Args:
                request: 供用例核对最终文本投递设置的执行请求。

            Returns:
                含调研摘要且状态为成功的任务结果。
            """
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
async def test_execute_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """执行器未在预算内完成时，派发动作应返回失败及超时提示。"""
    action, _plugin = _make_action()
    monkeypatch.setattr(
        dispatch_module, "TaskBudget", lambda: TaskBudget(timeout_seconds=0.01)
    )

    class _HangingExecutor:
        """用长时间等待模拟无法在任务预算内完成的执行器。"""

        def __init__(
            self, chatter: object, runtime: TaskRuntime, store: object = None
        ) -> None:
            """保存任务运行时，并接受协调器传入的兼容参数。

            Args:
                chatter: 为兼容执行器构造接口而接收的聊天器。
                runtime: 当前执行器关联的任务运行时。
                store: 为兼容执行器构造接口而接收的存储对象。
            """
            self.runtime = runtime

        async def run(self, request: object) -> None:
            """等待远超任务预算的时间，让派发超时逻辑取消执行。

            Args:
                request: 为兼容执行器接口而接收的请求，替身不处理。

            Raises:
                asyncio.CancelledError: 调度器在超时后取消等待时传播此异常。
            """
            await asyncio.sleep(999)

    monkeypatch.setattr(coordinator_module, "TaskExecutor", _HangingExecutor)
    monkeypatch.setattr(
        dispatch_module,
        "AgenticChatter",
        lambda stream_id, plugin: object(),
    )
    monkeypatch.setattr(
        dispatch_module,
        "_TIMEOUT_GRACE_SECONDS",
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
async def test_execute_returns_paused_result_with_active_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """执行器暂停后应返回成功的暂停提示，并保留可恢复的活跃任务。"""
    action, _plugin = _make_action()

    class _PausedExecutor:
        """将运行时切换为暂停状态，以模拟可恢复的执行结果。"""

        def __init__(
            self, chatter: object, runtime: TaskRuntime, store: object = None
        ) -> None:
            """保存任务运行时，并接受协调器传入的兼容参数。

            Args:
                chatter: 为兼容执行器构造接口而接收的聊天器。
                runtime: 当前执行器关联的任务运行时。
                store: 为兼容执行器构造接口而接收的存储对象。
            """
            self.runtime = runtime

        async def run(self, request: object) -> TaskResult:
            """暂停关联运行时并返回可恢复的任务结果。

            Args:
                request: 为兼容执行器接口而接收的请求，替身不处理。

            Returns:
                运行时暂停操作产生的任务结果。
            """
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

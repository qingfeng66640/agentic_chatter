"""runner 超时兜底（_run_guarded）的测试。"""

from __future__ import annotations

import asyncio

import pytest

from ..task_runtime import coordinator
from ..task_runtime.coordinator import _run_guarded
from ..task_runtime.executor import TaskRequest
from ..task_runtime.models import TaskResult, TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime


class _HangingExecutor:
    """永久等待且可记录通知的执行器，用于触发 runner 超时保护。"""

    def __init__(self, runtime: TaskRuntime) -> None:
        """绑定需要接受超时处理的运行时，并准备通知记录。"""
        self.runtime = runtime
        self.notices: list[str] = []

    async def run(self, request: TaskRequest) -> TaskResult:
        """在永不置位的事件上等待，使保护器必须通过超时终止执行。

        Args:
            request: 为兼容执行器协议接收的请求，不参与挂起行为。

        Returns:
            不会正常返回；结果类型保持与执行器接口一致。

        Raises:
            asyncio.CancelledError: 超时保护取消等待中的协程时向外传播。
            AssertionError: 永不置位的事件意外结束等待时触发不可达断言。
        """
        await asyncio.Event().wait()  # 永久挂起
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_notice(self, text: str) -> str:
        """记录并原样返回通知，用于核对超时提示是否发送。"""
        self.notices.append(text)
        return text


class _InstantExecutor:
    """立即返回预设结果且不改变运行时状态的正常执行器替身。"""

    def __init__(self, runtime: TaskRuntime, result: TaskResult) -> None:
        """绑定运行时及预设结果，供正常路径检查结果透传与通知行为。

        Args:
            runtime: 超时保护器监测的运行时，替身不会主动修改其状态。
            result: run 应原样返回的任务结果对象。
        """
        self.runtime = runtime
        self.result = result
        self.notices: list[str] = []

    async def run(self, request: TaskRequest) -> TaskResult:
        """忽略请求并立即返回同一结果对象，不更新运行时状态。"""
        return self.result

    async def send_notice(self, text: str) -> str:
        """记录并原样返回通知，用于核对超时提示是否发送。"""
        self.notices.append(text)
        return text


def _make_runtime(goal: str) -> TaskRuntime:
    """按给定目标创建并启动超时会话中的通用任务。"""
    runtime = TaskRuntime(
        TaskState(
            stream_id="timeout-stream",
            user_goal=goal,
            task_type=TaskType.GENERAL,
        )
    )
    runtime.start()
    return runtime


@pytest.mark.asyncio
async def test_hanging_run_times_out_to_paused() -> None:
    """挂起任务超时后应转为暂停，记录超时原因并提示用户继续任务。"""
    runtime = _make_runtime("悬挂任务")
    executor = _HangingExecutor(runtime)
    result = await _run_guarded(
        executor, runtime, TaskRequest(objective="目标"), timeout=0.05
    )
    assert result.status == TaskStatus.PAUSED
    assert runtime.state.status == TaskStatus.PAUSED
    assert "超时" in runtime.state.error or "超时" in result.summary
    assert executor.notices and "继续任务" in executor.notices[0]


@pytest.mark.asyncio
async def test_normal_run_passes_through_result() -> None:
    """正常完成时保护器应原样返回结果，不修改运行时状态或额外通知。"""
    runtime = _make_runtime("正常任务")
    expected = TaskResult(TaskStatus.SUCCEEDED, "完成")
    executor = _InstantExecutor(runtime, expected)
    result = await _run_guarded(
        executor, runtime, TaskRequest(objective="目标"), timeout=5.0
    )
    assert result is expected
    assert runtime.state.status == TaskStatus.RUNNING
    assert executor.notices == []


@pytest.mark.asyncio
async def test_timeout_on_terminal_status_keeps_result() -> None:
    """已成功的任务即使执行器随后超时，也应保留成功状态及原结果摘要。"""
    runtime = _make_runtime("终态任务")
    runtime.complete("已完成")
    executor = _HangingExecutor(runtime)
    result = await _run_guarded(
        executor, runtime, TaskRequest(objective="目标"), timeout=0.05
    )
    assert result.status == TaskStatus.SUCCEEDED
    assert result.summary == "已完成"


@pytest.mark.asyncio
async def test_start_task_runner_enforces_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start_task 集成：悬挂任务在预算+宽限后转 PAUSED。"""
    import dataclasses

    runtime = _make_runtime("集成超时任务")
    runtime.state.budget = dataclasses.replace(
        runtime.state.budget, timeout_seconds=1.0
    )
    executor = _HangingExecutor(runtime)
    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: executor)
    monkeypatch.setattr(coordinator, "_RUNNER_TIMEOUT_GRACE_SECONDS", 0.05)

    task = coordinator.start_task(
        object(), runtime, TaskRequest(objective="目标"), None
    )
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    coordinator.forget_task(runtime.state.task_id)

    assert runtime.state.status == TaskStatus.PAUSED
    assert executor.notices and "继续任务" in executor.notices[0]

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
    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self.notices: list[str] = []

    async def run(self, request: TaskRequest) -> TaskResult:
        await asyncio.Event().wait()  # 永久挂起
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_notice(self, text: str) -> str:
        self.notices.append(text)
        return text


class _InstantExecutor:
    def __init__(self, runtime: TaskRuntime, result: TaskResult) -> None:
        self.runtime = runtime
        self.result = result
        self.notices: list[str] = []

    async def run(self, request: TaskRequest) -> TaskResult:
        return self.result

    async def send_notice(self, text: str) -> str:
        self.notices.append(text)
        return text


def _make_runtime(goal: str) -> TaskRuntime:
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
    runtime = _make_runtime("终态任务")
    runtime.complete("已完成")
    executor = _HangingExecutor(runtime)
    result = await _run_guarded(
        executor, runtime, TaskRequest(objective="目标"), timeout=0.05
    )
    assert result.status == TaskStatus.SUCCEEDED
    assert result.summary == "已完成"


@pytest.mark.asyncio
async def test_start_task_runner_enforces_timeout(monkeypatch) -> None:
    """start_task 集成：悬挂任务在预算+宽限后转 PAUSED。"""
    import dataclasses

    runtime = _make_runtime("集成超时任务")
    runtime.state.budget = dataclasses.replace(
        runtime.state.budget, timeout_seconds=1.0
    )
    executor = _HangingExecutor(runtime)
    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: executor)
    monkeypatch.setattr(coordinator, "_RUNNER_TIMEOUT_GRACE_SECONDS", 0.05)

    task = coordinator.start_task(object(), runtime, TaskRequest(objective="目标"), None)
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    coordinator.forget_task(runtime.state.task_id)

    assert runtime.state.status == TaskStatus.PAUSED
    assert executor.notices and "继续任务" in executor.notices[0]

"""任务完成回灌主 Agent 的测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from ..task_runtime import coordinator
from ..task_runtime.coordinator import (
    drain_completed_events,
    has_completed_events,
    push_completed_event,
)
from ..task_runtime.executor import TaskRequest
from ..task_runtime.models import (
    TaskBudget,
    TaskResult,
    TaskState,
    TaskStatus,
    TaskType,
)
from ..task_runtime.runtime import TaskRuntime


def _make_runtime(stream_id: str, status: TaskStatus) -> TaskRuntime:
    """创建已启动的任务，并按完成事件场景设置其状态。

    Args:
        stream_id: 完成事件所属的会话流标识。
        status: 成功、等待用户、失败、暂停或取消状态；其他值保留运行状态。

    Returns:
        设置了结果长度预算及场景所需状态的任务运行时。
    """
    runtime = TaskRuntime(
        TaskState(
            stream_id=stream_id,
            user_goal="目标",
            task_type=TaskType.GENERAL,
            budget=TaskBudget(max_result_size=100),
        )
    )
    runtime.start()
    if status == TaskStatus.SUCCEEDED:
        runtime.complete("任务完成")
    elif status == TaskStatus.WAITING_USER:
        runtime.set_waiting_user("需要确认")
    elif status == TaskStatus.FAILED:
        runtime.fail("失败摘要", "失败原因")
    elif status == TaskStatus.PAUSED:
        runtime.pause()
    elif status == TaskStatus.CANCELLED:
        runtime.cancel()
    return runtime


def _make_result(runtime: TaskRuntime) -> TaskResult:
    """提取状态迁移已经生成的结果，防止用例误用没有结果的运行时。

    Args:
        runtime: 预期已经具有任务结果的运行时。

    Returns:
        运行时保存的原始任务结果。

    Raises:
        AssertionError: 运行时尚未生成结果。
    """
    result = runtime.state.result
    assert result is not None
    return result


@pytest.fixture(autouse=True)
def _clean_events() -> Iterator[None]:
    """在每个用例前后清空完成事件队列，避免跨用例消费残留事件。"""
    coordinator._COMPLETED_EVENTS.clear()
    yield
    coordinator._COMPLETED_EVENTS.clear()


@pytest.fixture
def resume_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """截获主会话恢复请求，供事件通知用例核对唤醒目标及来源。

    Args:
        monkeypatch: 用记录型替身替换会话管理器获取入口的补丁工具。

    Returns:
        按调用顺序保存会话流标识和唤醒来源的可变列表。
    """
    calls: list[tuple[str, str]] = []

    class _Manager:
        """仅收集主会话恢复调用，不实际唤醒 Chatter 的管理器替身。"""

        async def resume_chatter(self, stream_id: str, *, source: str = "") -> None:
            """按顺序记录待恢复会话和触发来源，不启动主 Agent。

            Args:
                stream_id: 待唤醒的会话流标识。
                source: 完成通知传入的唤醒来源。
            """
            calls.append((stream_id, source))

    import src.core.managers.chatter_manager as cm

    monkeypatch.setattr(cm, "get_chatter_manager", lambda: _Manager())
    return calls


@pytest.mark.asyncio
async def test_non_direct_success_pushes_event(
    resume_calls: list[tuple[str, str]],
) -> None:
    """非直送成功任务应产生包含状态和摘要的事件，并以子 Agent 来源唤醒会话。"""
    runtime = _make_runtime("stream-a", TaskStatus.SUCCEEDED)
    request = TaskRequest(
        objective="目标", deliver_final_text=False, report_events=True
    )
    await coordinator._notify_completion(
        object(), runtime, request, _make_result(runtime)
    )
    events = drain_completed_events("stream-a")
    assert len(events) == 1
    assert events[0]["status"] == TaskStatus.SUCCEEDED.value
    assert events[0]["summary"] == "任务完成"
    assert resume_calls == [("stream-a", "sub_agent")]


@pytest.mark.asyncio
async def test_direct_success_does_not_push_by_default(
    resume_calls: list[tuple[str, str]],
) -> None:
    """直送成功任务即使开启事件选项，也不应重复回灌事件或唤醒会话。"""
    runtime = _make_runtime("stream-b", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=True, report_events=True)
    await coordinator._notify_completion(
        object(), runtime, request, _make_result(runtime)
    )
    assert has_completed_events("stream-b") is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_waiting_user_pushes_even_for_direct_delivery(
    resume_calls: list[tuple[str, str]],
) -> None:
    """等待用户状态应越过直送及关闭事件选项的限制，回灌事件并唤醒会话。"""
    runtime = _make_runtime("stream-c", TaskStatus.WAITING_USER)
    request = TaskRequest(
        objective="目标", deliver_final_text=True, report_events=False
    )
    await coordinator._notify_completion(
        object(), runtime, request, _make_result(runtime)
    )
    events = drain_completed_events("stream-c")
    assert len(events) == 1
    assert events[0]["status"] == TaskStatus.WAITING_USER.value
    assert resume_calls == [("stream-c", "sub_agent")]


@pytest.mark.asyncio
async def test_failed_does_not_push_by_default(
    resume_calls: list[tuple[str, str]],
) -> None:
    """失败任务默认不回灌完成事件，也不唤醒主会话。"""
    runtime = _make_runtime("stream-d", TaskStatus.FAILED)
    request = TaskRequest(
        objective="目标", deliver_final_text=False, report_events=True
    )
    await coordinator._notify_completion(
        object(), runtime, request, _make_result(runtime)
    )
    assert has_completed_events("stream-d") is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_paused_and_cancelled_do_not_push(
    resume_calls: list[tuple[str, str]],
) -> None:
    """暂停和取消任务均不应产生完成事件或唤醒主会话。"""
    for index, status in enumerate((TaskStatus.PAUSED, TaskStatus.CANCELLED)):
        stream_id = f"stream-e{index}"
        runtime = _make_runtime(stream_id, status)
        request = TaskRequest(
            objective="目标", deliver_final_text=False, report_events=True
        )
        await coordinator._notify_completion(
            object(), runtime, request, _make_result(runtime)
        )
        assert has_completed_events(stream_id) is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_report_events_disabled_disables_success_push(
    resume_calls: list[tuple[str, str]],
) -> None:
    """非直送成功任务关闭事件报告后，不应回灌事件或唤醒主会话。"""
    runtime = _make_runtime("stream-f", TaskStatus.SUCCEEDED)
    request = TaskRequest(
        objective="目标", deliver_final_text=False, report_events=False
    )
    await coordinator._notify_completion(
        object(), runtime, request, _make_result(runtime)
    )
    assert has_completed_events("stream-f") is False
    assert resume_calls == []


def test_push_is_idempotent_per_task_id() -> None:
    """同一任务标识的完成事件重复入队时应只保留一条。"""
    payload: dict[str, Any] = {
        "task_id": "task-1",
        "task_type": "general",
        "status": "succeeded",
        "summary": "结果",
        "error": "",
    }
    push_completed_event("stream-g", payload)
    push_completed_event("stream-g", dict(payload))
    assert len(coordinator._COMPLETED_EVENTS["stream-g"]) == 1


def test_acknowledge_only_snapshot_preserves_new_events() -> None:
    """确认已取快照后应只移除快照内事件，保留随后到达的新事件。"""
    first = {"task_id": "first", "summary": "第一条"}
    second = {"task_id": "second", "summary": "第二条"}
    push_completed_event("stream-snapshot", first)
    snapshot = coordinator.pending_completed_events("stream-snapshot")
    push_completed_event("stream-snapshot", second)

    coordinator.acknowledge_completed_events("stream-snapshot", snapshot)
    assert drain_completed_events("stream-snapshot") == [second]


def test_drain_is_single_consumption() -> None:
    """完成事件只能被取出一次，重复读取后队列应保持为空。"""
    push_completed_event(
        "stream-h",
        {
            "task_id": "task-2",
            "task_type": "general",
            "status": "succeeded",
            "summary": "",
            "error": "",
        },
    )
    first = drain_completed_events("stream-h")
    second = drain_completed_events("stream-h")
    assert len(first) == 1
    assert second == []
    assert has_completed_events("stream-h") is False


@pytest.mark.asyncio
async def test_runner_pushes_completion_after_run(
    monkeypatch: pytest.MonkeyPatch, resume_calls: list[tuple[str, str]]
) -> None:
    """后台 runner 完成后应保存状态、回灌一次事件并唤醒对应会话。"""
    runtime = _make_runtime("stream-i", TaskStatus.SUCCEEDED)
    request = TaskRequest(
        objective="目标", deliver_final_text=False, report_events=True
    )

    class _Executor:
        """直接返回已完成任务结果，使 runner 的通知行为可独立验证。"""

        def __init__(self) -> None:
            """绑定当前用例已完成的运行时，供 runner 保存状态。"""
            self.runtime = runtime

        async def run(self, req: TaskRequest) -> TaskResult:
            """忽略传入请求，返回运行时已有结果，不再进行模型调用。"""
            return _make_result(runtime)

    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: _Executor())
    saved: list[Any] = []

    class _Store:
        """收集状态保存请求，用于断言 runner 已触发持久化。"""

        def save(self, runtime: TaskRuntime) -> None:
            """记录传入的运行时对象，不执行实际的磁盘写入。"""
            saved.append(runtime)

    task = coordinator.start_task(object(), runtime, request, _Store())
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)

    events = drain_completed_events("stream-i")
    assert len(events) == 1
    assert resume_calls == [("stream-i", "sub_agent")]
    assert saved
    coordinator.forget_task(runtime.state.task_id)


@pytest.mark.asyncio
async def test_runner_does_not_push_when_disabled(
    monkeypatch: pytest.MonkeyPatch, resume_calls: list[tuple[str, str]]
) -> None:
    """后台 runner 的直送请求结束后不应额外回灌事件或唤醒会话。"""
    runtime = _make_runtime("stream-j", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=True, report_events=True)

    class _Executor:
        """直接返回已完成任务结果，使 runner 的通知行为可独立验证。"""

        def __init__(self) -> None:
            """绑定当前用例已完成的运行时，供 runner 保存状态。"""
            self.runtime = runtime

        async def run(self, req: TaskRequest) -> TaskResult:
            """忽略传入请求，返回运行时已有结果，不再进行模型调用。"""
            return _make_result(runtime)

    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: _Executor())
    task = coordinator.start_task(object(), runtime, request, None)
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    coordinator.forget_task(runtime.state.task_id)
    assert has_completed_events("stream-j") is False
    assert resume_calls == []


def test_dispatch_path_never_produces_events(
    resume_calls: list[tuple[str, str]],
) -> None:
    """dispatch 同步等待不经 runner()，天然不产生完成事件。"""
    push_completed_event("stream-k", {"task_id": "task-3"})
    drain_completed_events("stream-k")
    assert has_completed_events("stream-k") is False


def test_payload_summary_truncated_to_budget() -> None:
    """完成载荷构建应按结果预算限制摘要和错误文本长度。"""
    runtime = TaskRuntime(
        TaskState(
            stream_id="stream-l",
            user_goal="目标",
            budget=TaskBudget(max_result_size=5),
        )
    )
    payload = coordinator._build_completion_payload(
        runtime,
        TaskResult(TaskStatus.SUCCEEDED, "超出预算的长摘要文本", error="错误内容"),
    )
    assert payload["summary"] == "超出预算的"
    assert payload["error"] == "错误内容"[:5]

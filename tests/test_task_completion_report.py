"""任务完成回灌主 Agent 的测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ..task_runtime import coordinator
from ..task_runtime.coordinator import (
    drain_completed_events,
    has_completed_events,
    push_completed_event,
)
from ..task_runtime.executor import TaskRequest
from ..task_runtime.models import TaskBudget, TaskResult, TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime


def _make_runtime(stream_id: str, status: TaskStatus) -> TaskRuntime:
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
    result = runtime.state.result
    assert result is not None
    return result


@pytest.fixture(autouse=True)
def _clean_events():
    coordinator._COMPLETED_EVENTS.clear()
    yield
    coordinator._COMPLETED_EVENTS.clear()


@pytest.fixture
def resume_calls(monkeypatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    class _Manager:
        async def resume_chatter(self, stream_id: str, *, source: str = "") -> None:
            calls.append((stream_id, source))

    import src.core.managers.chatter_manager as cm

    monkeypatch.setattr(cm, "get_chatter_manager", lambda: _Manager())
    return calls


@pytest.mark.asyncio
async def test_non_direct_success_pushes_event(resume_calls) -> None:
    runtime = _make_runtime("stream-a", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=False, report_events=True)
    await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
    events = drain_completed_events("stream-a")
    assert len(events) == 1
    assert events[0]["status"] == TaskStatus.SUCCEEDED.value
    assert events[0]["summary"] == "任务完成"
    assert resume_calls == [("stream-a", "sub_agent")]


@pytest.mark.asyncio
async def test_direct_success_does_not_push_by_default(resume_calls) -> None:
    runtime = _make_runtime("stream-b", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=True, report_events=True)
    await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
    assert has_completed_events("stream-b") is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_waiting_user_pushes_even_for_direct_delivery(resume_calls) -> None:
    runtime = _make_runtime("stream-c", TaskStatus.WAITING_USER)
    request = TaskRequest(objective="目标", deliver_final_text=True, report_events=False)
    await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
    events = drain_completed_events("stream-c")
    assert len(events) == 1
    assert events[0]["status"] == TaskStatus.WAITING_USER.value
    assert resume_calls == [("stream-c", "sub_agent")]


@pytest.mark.asyncio
async def test_failed_does_not_push_by_default(resume_calls) -> None:
    runtime = _make_runtime("stream-d", TaskStatus.FAILED)
    request = TaskRequest(objective="目标", deliver_final_text=False, report_events=True)
    await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
    assert has_completed_events("stream-d") is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_paused_and_cancelled_do_not_push(resume_calls) -> None:
    for index, status in enumerate((TaskStatus.PAUSED, TaskStatus.CANCELLED)):
        stream_id = f"stream-e{index}"
        runtime = _make_runtime(stream_id, status)
        request = TaskRequest(objective="目标", deliver_final_text=False, report_events=True)
        await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
        assert has_completed_events(stream_id) is False
    assert resume_calls == []


@pytest.mark.asyncio
async def test_report_events_disabled_disables_success_push(resume_calls) -> None:
    runtime = _make_runtime("stream-f", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=False, report_events=False)
    await coordinator._notify_completion(object(), runtime, request, _make_result(runtime))
    assert has_completed_events("stream-f") is False
    assert resume_calls == []


def test_push_is_idempotent_per_task_id() -> None:
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


def test_drain_is_single_consumption() -> None:
    push_completed_event(
        "stream-h",
        {"task_id": "task-2", "task_type": "general", "status": "succeeded",
         "summary": "", "error": ""},
    )
    first = drain_completed_events("stream-h")
    second = drain_completed_events("stream-h")
    assert len(first) == 1
    assert second == []
    assert has_completed_events("stream-h") is False


@pytest.mark.asyncio
async def test_runner_pushes_completion_after_run(monkeypatch, resume_calls) -> None:
    runtime = _make_runtime("stream-i", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=False, report_events=True)

    class _Executor:
        def __init__(self) -> None:
            self.runtime = runtime

        async def run(self, req: TaskRequest) -> TaskResult:
            return _make_result(runtime)

    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: _Executor())
    saved: list[Any] = []

    class _Store:
        def save(self, runtime: TaskRuntime) -> None:
            saved.append(runtime)

    task = coordinator.start_task(object(), runtime, request, _Store())
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)

    events = drain_completed_events("stream-i")
    assert len(events) == 1
    assert resume_calls == [("stream-i", "sub_agent")]
    assert saved
    coordinator.forget_task(runtime.state.task_id)


@pytest.mark.asyncio
async def test_runner_does_not_push_when_disabled(monkeypatch, resume_calls) -> None:
    runtime = _make_runtime("stream-j", TaskStatus.SUCCEEDED)
    request = TaskRequest(objective="目标", deliver_final_text=True, report_events=True)

    class _Executor:
        def __init__(self) -> None:
            self.runtime = runtime

        async def run(self, req: TaskRequest) -> TaskResult:
            return _make_result(runtime)

    monkeypatch.setattr(coordinator, "TaskExecutor", lambda *a, **k: _Executor())
    task = coordinator.start_task(object(), runtime, request, None)
    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    coordinator.forget_task(runtime.state.task_id)
    assert has_completed_events("stream-j") is False
    assert resume_calls == []


def test_dispatch_path_never_produces_events(resume_calls) -> None:
    """dispatch 同步等待不经 runner()，天然不产生完成事件。"""
    push_completed_event("stream-k", {"task_id": "task-3"})
    drain_completed_events("stream-k")
    assert has_completed_events("stream-k") is False


def test_payload_summary_truncated_to_budget() -> None:
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

"""任务取消路径的测试：运行/暂停占位/等待占位/已完成幂等。"""

from __future__ import annotations


import pytest

from ..task_runtime import coordinator
from ..task_runtime.coordinator import (
    _cancel_active,
    cancel_task_runtime,
    register_task,
)
from ..task_runtime.executor import TaskRequest
from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.persistence import TaskStateStore
from ..task_runtime.runtime import TaskRuntime, get_task_runtime_manager


class _FakeExecutor:
    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self.notices: list[str] = []

    async def send_notice(self, text: str) -> str:
        self.notices.append(text)
        return text


class _FakeTask:
    def __init__(self, done: bool = True) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done

    def cancelled(self) -> bool:
        return False


def _make_active(
    runtime: TaskRuntime,
    *,
    done: bool = True,
    store: TaskStateStore | None = None,
) -> coordinator.ActiveTask:
    return coordinator.ActiveTask(
        _FakeExecutor(runtime),
        _FakeTask(done=done),
        "manager-task-1",
        object(),
        TaskRequest(objective=runtime.state.user_goal),
        store,
    )


def _make_runtime(stream_id: str, goal: str, status: TaskStatus) -> TaskRuntime:
    runtime = TaskRuntime(
        TaskState(
            stream_id=stream_id,
            user_goal=goal,
            task_type=TaskType.GENERAL,
        )
    )
    runtime.start()
    if status == TaskStatus.PAUSED:
        runtime.pause()
    elif status == TaskStatus.WAITING_USER:
        runtime.set_waiting_user("需要确认")
    elif status == TaskStatus.SUCCEEDED:
        runtime.complete("已完成")
    return runtime


@pytest.fixture(autouse=True)
def _clean_registries():
    manager = get_task_runtime_manager()
    saved_tasks = dict(manager._tasks)
    saved_active = dict(coordinator._ACTIVE)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved_tasks)
    coordinator._ACTIVE.clear()
    coordinator._ACTIVE.update(saved_active)


@pytest.mark.asyncio
async def test_cancel_running_task_notifies_and_cleans(monkeypatch) -> None:
    runtime = _make_runtime("stream-cancel", "运行中任务", TaskStatus.RUNNING)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    active = _make_active(runtime, done=False)
    coordinator._ACTIVE[runtime.state.task_id] = active

    cancelled: list[str] = []

    def _fake_cancel(task_id: str) -> None:
        cancelled.append(task_id)

    # get_task_manager 返回的是单例管理器，直接 patch 其方法
    manager_instance = coordinator.get_task_manager()
    monkeypatch.setattr(manager_instance, "cancel_task", _fake_cancel)

    result = await _cancel_active(active, notify=True)
    assert result.status == TaskStatus.CANCELLED
    assert runtime.state.status == TaskStatus.CANCELLED
    assert cancelled == ["manager-task-1"]
    assert active.executor.notices and "已取消" in active.executor.notices[0]


@pytest.mark.asyncio
async def test_cancel_paused_placeholder_cleans_registries(tmp_path) -> None:
    """核心回归：PAUSED 占位任务此前取消被吞，现在必须真正清理。"""
    store = TaskStateStore(str(tmp_path))
    runtime = _make_runtime("stream-paused", "暂停任务", TaskStatus.PAUSED)
    store.save(runtime)
    active = _make_active(runtime, done=True, store=store)
    coordinator._ACTIVE[runtime.state.task_id] = active
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime

    await _cancel_active(active, notify=True)

    assert runtime.state.status == TaskStatus.CANCELLED
    assert coordinator._ACTIVE.get(runtime.state.task_id) is None
    assert get_task_runtime_manager().get(runtime.state.task_id) is None


@pytest.mark.asyncio
async def test_cancel_waiting_placeholder_cleans_registries(tmp_path) -> None:
    store = TaskStateStore(str(tmp_path))
    runtime = _make_runtime("stream-waiting", "等待任务", TaskStatus.WAITING_USER)
    store.save(runtime)
    active = _make_active(runtime, done=True, store=store)
    coordinator._ACTIVE[runtime.state.task_id] = active
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime

    await _cancel_active(active, notify=False)

    assert runtime.state.status == TaskStatus.CANCELLED
    assert coordinator._ACTIVE.get(runtime.state.task_id) is None
    assert get_task_runtime_manager().get(runtime.state.task_id) is None
    assert active.executor.notices == []


@pytest.mark.asyncio
async def test_cancel_succeeded_task_is_idempotent() -> None:
    runtime = _make_runtime("stream-done", "完成任务", TaskStatus.SUCCEEDED)
    active = _make_active(runtime, done=True)
    result = await _cancel_active(active, notify=False)
    assert result.status == TaskStatus.SUCCEEDED
    assert runtime.state.status == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_task_runtime_without_active_entry() -> None:
    runtime = _make_runtime("stream-orphan", "孤儿任务", TaskStatus.RUNNING)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    message = await cancel_task_runtime(runtime, notify=False)
    assert "已取消" in message
    assert runtime.state.status == TaskStatus.CANCELLED
    assert get_task_runtime_manager().get(runtime.state.task_id) is None


@pytest.mark.asyncio
async def test_register_task_then_cancel_via_route_message(monkeypatch) -> None:
    """占位任务经 route_message('取消任务') 被真正取消（修复吞消息）。"""
    runtime = _make_runtime("stream-route", "路由取消任务", TaskStatus.PAUSED)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    register_task(object(), runtime, TaskRequest(objective="目标"), None)

    manager_instance = coordinator.get_task_manager()
    monkeypatch.setattr(
        manager_instance, "cancel_task", lambda task_id: None
    )

    routed = await coordinator.route_message(runtime.state.task_id, "取消任务")
    assert routed is True
    assert runtime.state.status == TaskStatus.CANCELLED
    assert coordinator._ACTIVE.get(runtime.state.task_id) is None

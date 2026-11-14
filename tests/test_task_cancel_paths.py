"""任务取消路径的测试：运行/暂停占位/等待占位/已完成幂等。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
    """绑定任务状态并记录取消通知的执行器替身。"""

    def __init__(self, runtime: TaskRuntime) -> None:
        """保留待取消任务的运行时，并初始化通知记录。"""
        self.runtime = runtime
        self.notices: list[str] = []

    async def send_notice(self, text: str) -> str:
        """收集取消通知并原样返回，以检查通知开关是否生效。"""
        self.notices.append(text)
        return text


class _FakeTask:
    """用可配置完成状态区分运行中后台任务和已完成的占位句柄。"""

    def __init__(self, done: bool = True) -> None:
        """保存后台句柄是否已完成的状态，默认模拟占位任务。"""
        self._done = done

    def done(self) -> bool:
        """返回预设完成状态，供取消逻辑区分后台任务与占位任务。"""
        return self._done

    def cancelled(self) -> bool:
        """固定返回未取消，避免句柄状态代替运行时的取消路径。"""
        return False


def _make_active(
    runtime: TaskRuntime,
    *,
    done: bool = True,
    store: TaskStateStore | None = None,
) -> coordinator.ActiveTask:
    """为取消路径组装带执行器、任务句柄和可选存储的活动任务。

    Args:
        runtime: 取消操作要修改的任务运行时。
        done: 后台句柄是否已完成；为真时模拟暂停或等待占位任务。
        store: 取消过程使用的状态存储，不需要持久化时为空。

    Returns:
        使用固定管理器任务标识、并按运行时目标构造请求的活动任务。
    """
    return coordinator.ActiveTask(
        _FakeExecutor(runtime),
        _FakeTask(done=done),
        "manager-task-1",
        object(),
        TaskRequest(objective=runtime.state.user_goal),
        store,
    )


def _make_runtime(stream_id: str, goal: str, status: TaskStatus) -> TaskRuntime:
    """创建已启动的任务，并设置取消用例需要的暂停、等待或成功状态。

    Args:
        stream_id: 任务所属的会话流标识。
        goal: 运行时保存的用户目标。
        status: 暂停、等待用户或成功状态；其他值保留运行状态。

    Returns:
        已完成场景状态初始化的通用任务运行时。
    """
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
def _clean_registries() -> Iterator[None]:
    """快照任务管理器与活动任务注册表，并在用例结束后恢复原有条目。"""
    manager = get_task_runtime_manager()
    saved_tasks = dict(manager._tasks)
    saved_active = dict(coordinator._ACTIVE)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved_tasks)
    coordinator._ACTIVE.clear()
    coordinator._ACTIVE.update(saved_active)


@pytest.mark.asyncio
async def test_cancel_running_task_notifies_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行中任务取消后应更新状态、取消管理器后台任务并发送取消通知。"""
    runtime = _make_runtime("stream-cancel", "运行中任务", TaskStatus.RUNNING)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    active = _make_active(runtime, done=False)
    coordinator._ACTIVE[runtime.state.task_id] = active

    cancelled: list[str] = []

    def _fake_cancel(task_id: str) -> None:
        """记录取消请求中的管理器任务标识，不触碰实际后台任务。"""
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
async def test_cancel_paused_placeholder_cleans_registries(tmp_path: Path) -> None:
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
async def test_cancel_waiting_placeholder_cleans_registries(tmp_path: Path) -> None:
    """等待用户的占位任务取消后应清理两个注册表，关闭通知时不发送消息。"""
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
    """取消已成功的任务应保持运行时及返回结果的成功状态。"""
    runtime = _make_runtime("stream-done", "完成任务", TaskStatus.SUCCEEDED)
    active = _make_active(runtime, done=True)
    result = await _cancel_active(active, notify=False)
    assert result.status == TaskStatus.SUCCEEDED
    assert runtime.state.status == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_task_runtime_without_active_entry() -> None:
    """没有活动注册条目的任务仍应被取消，并从运行时管理器中移除。"""
    runtime = _make_runtime("stream-orphan", "孤儿任务", TaskStatus.RUNNING)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    message = await cancel_task_runtime(runtime, notify=False)
    assert "已取消" in message
    assert runtime.state.status == TaskStatus.CANCELLED
    assert get_task_runtime_manager().get(runtime.state.task_id) is None


@pytest.mark.asyncio
async def test_register_task_then_cancel_via_route_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """占位任务经 route_message('取消任务') 被真正取消（修复吞消息）。"""
    runtime = _make_runtime("stream-route", "路由取消任务", TaskStatus.PAUSED)
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    register_task(object(), runtime, TaskRequest(objective="目标"), None)

    manager_instance = coordinator.get_task_manager()
    monkeypatch.setattr(manager_instance, "cancel_task", lambda task_id: None)

    routed = await coordinator.route_message(runtime.state.task_id, "取消任务")
    assert routed is True
    assert runtime.state.status == TaskStatus.CANCELLED
    assert coordinator._ACTIVE.get(runtime.state.task_id) is None

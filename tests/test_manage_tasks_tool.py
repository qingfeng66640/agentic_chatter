"""manage_tasks 工具的测试。"""

from __future__ import annotations

import pytest

from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime, get_task_runtime_manager
from ..tooling.manage_tasks import ManageTasksTool, _resolve_task_ref


def _make_tool(stream_id: str) -> ManageTasksTool:
    tool = ManageTasksTool.__new__(ManageTasksTool)
    tool.stream_id = stream_id
    tool.trigger_message = None
    return tool


def _make_task(stream_id: str, goal: str, status: TaskStatus = TaskStatus.RUNNING) -> TaskRuntime:
    runtime = TaskRuntime(
        TaskState(stream_id=stream_id, user_goal=goal, task_type=TaskType.GENERAL)
    )
    runtime.start()
    if status == TaskStatus.PAUSED:
        runtime.pause()
    return runtime


@pytest.fixture(autouse=True)
def _clean_registries():
    manager = get_task_runtime_manager()
    saved = dict(manager._tasks)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved)


@pytest.mark.asyncio
async def test_list_empty_stream() -> None:
    ok, message = await _make_tool("empty-stream").execute("list")
    assert ok is True
    assert "没有进行中的任务" in message


@pytest.mark.asyncio
async def test_list_renders_tasks() -> None:
    runtime = _make_task("stream-l", "调研竞品")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-l").execute("list")
    assert ok is True
    assert "共 1 个任务" in message
    assert "调研竞品" in message


@pytest.mark.asyncio
async def test_status_by_ordinal_and_prefix() -> None:
    first = _make_task("stream-s", "任务一")
    second = _make_task("stream-s", "任务二", TaskStatus.PAUSED)
    manager = get_task_runtime_manager()
    manager._tasks[first.state.task_id] = first
    manager._tasks[second.state.task_id] = second
    tool = _make_tool("stream-s")

    ok, message = await tool.execute("status", "1")
    assert ok is True
    assert "任务一" in message

    ok, message = await tool.execute("status", second.state.task_id[:8])
    assert ok is True
    assert "任务二" in message


@pytest.mark.asyncio
async def test_status_requires_ref() -> None:
    runtime = _make_task("stream-r", "任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-r").execute("status", "")
    assert ok is False
    assert "序号" in message or "ID" in message


@pytest.mark.asyncio
async def test_status_ambiguous_prefix() -> None:
    tasks = [_make_task("stream-a", "任务一"), _make_task("stream-a", "任务二")]
    manager = get_task_runtime_manager()
    for runtime in tasks:
        manager._tasks[runtime.state.task_id] = runtime
    # 构造同前缀：把第二个的 ID 换成第一个的 6 位前缀 + 不同后缀
    manager._tasks.pop(tasks[1].state.task_id)
    tasks[1].state.task_id = tasks[0].state.task_id[:6] + "ffffff"
    manager._tasks[tasks[1].state.task_id] = tasks[1]

    ok, message = await _make_tool("stream-a").execute("status", tasks[0].state.task_id[:6])
    assert ok is False
    assert "多个任务" in message


@pytest.mark.asyncio
async def test_cancel_by_ordinal_removes_task() -> None:
    runtime = _make_task("stream-c", "待取消任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-c").execute("cancel", "1")
    assert ok is True
    assert "已取消" in message
    assert runtime.state.status == TaskStatus.CANCELLED
    assert get_task_runtime_manager().get(runtime.state.task_id) is None


@pytest.mark.asyncio
async def test_missing_stream_id_fails() -> None:
    tool = _make_tool("")
    ok, message = await tool.execute("list")
    assert ok is False
    assert "聊天流" in message


def test_resolve_task_ref_out_of_range() -> None:
    tasks = [_make_task("s", "唯一")]
    target, error = _resolve_task_ref("5", tasks)
    assert target is None
    assert "超出范围" in error


def test_resolve_task_ref_short_prefix_rejected() -> None:
    tasks = [_make_task("s", "唯一")]
    target, error = _resolve_task_ref(tasks[0].state.task_id[:3], tasks)
    assert target is None
    assert "至少" in error

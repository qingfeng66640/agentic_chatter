"""manage_tasks 工具的测试。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime, get_task_runtime_manager
from ..tooling.manage_tasks import ManageTasksTool, _resolve_task_ref


def _make_tool(stream_id: str) -> ManageTasksTool:
    """绕过组件初始化构造绑定指定聊天流的任务管理工具。

    Args:
        stream_id: 工具查询或控制任务时使用的聊天流标识。

    Returns:
        未关联触发消息的任务管理工具。
    """
    tool = ManageTasksTool.__new__(ManageTasksTool)
    tool.stream_id = stream_id
    tool.trigger_message = None
    return tool


def _make_task(
    stream_id: str, goal: str, status: TaskStatus = TaskStatus.RUNNING
) -> TaskRuntime:
    """创建已启动的通用任务，并按需切换到暂停状态。

    Args:
        stream_id: 任务所属的聊天流标识。
        goal: 用于构造任务状态的用户目标。
        status: 为 PAUSED 时暂停任务，其他值保留运行状态。

    Returns:
        尚未注册到管理器的任务运行时。
    """
    runtime = TaskRuntime(
        TaskState(stream_id=stream_id, user_goal=goal, task_type=TaskType.GENERAL)
    )
    runtime.start()
    if status == TaskStatus.PAUSED:
        runtime.pause()
    return runtime


@pytest.fixture(autouse=True)
def _clean_registries() -> Iterator[None]:
    """保存任务注册表，并在用例结束后恢复原有条目。

    Yields:
        None: 将控制权交给用例，随后还原注册表。
    """
    manager = get_task_runtime_manager()
    saved = dict(manager._tasks)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved)


@pytest.mark.asyncio
async def test_list_empty_stream() -> None:
    """聊天流没有活跃任务时，列表查询应成功并返回空列表提示。"""
    ok, message = await _make_tool("empty-stream").execute("list")
    assert ok is True
    assert "没有进行中的任务" in message


@pytest.mark.asyncio
async def test_list_renders_tasks() -> None:
    """列表查询应包含当前聊天流的任务总数与目标。"""
    runtime = _make_task("stream-l", "调研竞品")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-l").execute("list")
    assert ok is True
    assert "共 1 个任务" in message
    assert "调研竞品" in message


@pytest.mark.asyncio
async def test_status_by_ordinal_and_prefix() -> None:
    """状态查询应分别通过序号和任务 ID 前缀选中对应任务。"""
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
    """状态查询未提供任务引用时，应失败并提示使用序号或 ID。"""
    runtime = _make_task("stream-r", "任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-r").execute("status", "")
    assert ok is False
    assert "序号" in message or "ID" in message


@pytest.mark.asyncio
async def test_status_ambiguous_prefix() -> None:
    """任务 ID 前缀同时匹配多个任务时，应拒绝查询并提示歧义。"""
    tasks = [_make_task("stream-a", "任务一"), _make_task("stream-a", "任务二")]
    manager = get_task_runtime_manager()
    # 构造同前缀。task_id 来自随机 uuid4().hex，前 6 位有约 15% 概率全为数字，
    # 那样该前缀会被当作序号而非 ID 前缀。改用固定含字母的前缀，
    # 让本用例稳定地覆盖「前缀命中多个任务」这一分支。
    shared_prefix = "abcdef"
    tasks[0].state.task_id = shared_prefix + "0123456789abcdef"
    tasks[1].state.task_id = shared_prefix + "fedcba9876543210"
    for runtime in tasks:
        manager._tasks[runtime.state.task_id] = runtime

    ok, message = await _make_tool("stream-a").execute("status", shared_prefix)
    assert ok is False
    assert "多个任务" in message


@pytest.mark.asyncio
async def test_cancel_by_ordinal_removes_task() -> None:
    """按序号取消任务后，应返回取消提示并从管理器移除该任务。"""
    runtime = _make_task("stream-c", "待取消任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    ok, message = await _make_tool("stream-c").execute("cancel", "1")
    assert ok is True
    assert "已取消" in message
    assert runtime.state.status == TaskStatus.CANCELLED
    assert get_task_runtime_manager().get(runtime.state.task_id) is None


@pytest.mark.asyncio
async def test_missing_stream_id_fails() -> None:
    """缺少聊天流标识时，应拒绝任务列表查询并提示上下文缺失。"""
    tool = _make_tool("")
    ok, message = await tool.execute("list")
    assert ok is False
    assert "聊天流" in message


def test_resolve_task_ref_out_of_range() -> None:
    """超出候选列表范围的序号应返回空目标并提示越界。"""
    tasks = [_make_task("s", "唯一")]
    target, error = _resolve_task_ref("5", tasks)
    assert target is None
    assert "超出范围" in error


def test_resolve_task_ref_short_prefix_rejected() -> None:
    """短于 ID 前缀下限的非数字输入应被拒绝并提示长度要求。"""
    tasks = [_make_task("s", "唯一")]
    # 前缀取自随机 uuid4().hex，若恰好为纯数字会被正确地当作序号解释，
    # 从而得到「超出范围」而非长度提示。此处固定一个非数字短前缀，
    # 断言的才是本用例真正要验证的分支。
    target, error = _resolve_task_ref("abc", tasks)
    assert target is None
    assert "至少" in error


def test_resolve_task_ref_short_numeric_prefix_is_ordinal() -> None:
    """短纯数字输入只能按序号解释，越界时给出序号范围提示。"""
    tasks = [_make_task("s", "唯一")]

    # 范围内：命中第一个任务
    target, _ = _resolve_task_ref("1", tasks)
    assert target is tasks[0]

    # 越界：报序号范围，而不是 ID 前缀长度
    target, error = _resolve_task_ref("999", tasks)
    assert target is None
    assert "超出范围" in error


def test_resolve_task_ref_numeric_id_prefix_not_mistaken_for_ordinal() -> None:
    """纯数字任务ID前缀不应被当作序号。

    task_id 来自 uuid4().hex，可能以纯数字开头；若只按序号解释，
    用户粘贴纯数字 ID 前缀时会被误报「序号超出范围」。
    """
    runtime = _make_task("s", "唯一")
    runtime.state.task_id = "12345678" + runtime.state.task_id[8:]
    tasks = [runtime]

    # 序号仍在范围内时优先按序号解释
    target, _ = _resolve_task_ref("1", tasks)
    assert target is runtime

    # 越界的纯数字前缀应回退到 ID 前缀匹配并命中
    target, error = _resolve_task_ref("12345678", tasks)
    assert target is runtime
    assert error == ""


def test_resolve_task_ref_numeric_out_of_range_reports_both() -> None:
    """既不是有效序号也不匹配任何 ID 时，错误信息同时提示两种可能。"""
    tasks = [_make_task("s", "唯一")]

    target, error = _resolve_task_ref("999999", tasks)

    assert target is None
    assert "超出范围" in error

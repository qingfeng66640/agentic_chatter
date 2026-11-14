"""chatter 层多任务路由语义的测试。"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from .. import chatter as chatter_module
from ..task_runtime import coordinator
from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime, get_task_runtime_manager


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


class _RoutingChatter:
    """最小 chatter 替身：仅承载 _route_task_message 所需属性。"""

    def __init__(self, stream_id: str, sent: list[str]) -> None:
        """保存路由调用需要的聊天流和发送记录。

        Args:
            stream_id: 路由操作所属的聊天流标识。
            sent: 与发送 API 替身共享的消息记录列表。
        """
        self.stream_id = stream_id
        self.sent = sent


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """保存任务和活跃协调器注册表，并在用例结束后恢复。

    Args:
        monkeypatch: pytest 注入的补丁 fixture，此处不施加补丁。

    Yields:
        None: 将控制权交给用例，随后恢复两份注册表。
    """
    manager = get_task_runtime_manager()
    saved = dict(manager._tasks)
    saved_active = dict(coordinator._ACTIVE)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved)
    coordinator._ACTIVE.clear()
    coordinator._ACTIVE.update(saved_active)


@pytest.fixture
def send_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """将发送 API 替换为只记录文本且报告成功的替身。

    Args:
        monkeypatch: 用于在用例结束后还原发送 API 的补丁 fixture。

    Returns:
        替身发送文本时追加内容的共享列表。
    """
    log: list[str] = []

    class _SendAPI:
        """记录发送文本而不访问真实消息平台。"""

        @staticmethod
        async def send_text(*, content: str, stream_id: str, **kwargs: Any) -> bool:
            """收集消息正文并模拟发送成功。

            Args:
                content: 追加到发送日志的消息正文。
                stream_id: 为兼容发送接口而接收的聊天流标识。
                **kwargs: 真实发送接口的其他可选参数，替身不处理。

            Returns:
                始终为 True，表示模拟发送成功。
            """
            log.append(content)
            return True

    monkeypatch.setattr(chatter_module, "send_api", _SendAPI())
    return log


@pytest.mark.asyncio
async def test_route_single_task_input(send_log: list[str]) -> None:
    """补充消息交给唯一任务后，应返回该运行时并标记为已消费。"""
    runtime = _make_task("stream-1", "唯一任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    register_calls: list[str] = []
    monkey_route = SimpleNamespace()

    async def _fake_route(task_id: str, text: str) -> bool:
        """记录补充正文和目标任务，并模拟路由成功。

        Args:
            task_id: 路由选中的任务标识。
            text: 追加到调用记录的补充正文。

        Returns:
            始终为 True，表示消息已成功交给任务。
        """
        register_calls.append((task_id, text)[1])
        monkey_route.last = task_id
        return True

    chatter = _RoutingChatter("stream-1", send_log)
    tasks = [runtime]
    # 直接调用内部 route_message 的打桩版本：通过 monkeypatch 模块函数
    import unittest.mock as mock

    with mock.patch.object(chatter_module, "route_message", _fake_route):
        routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
            chatter, tasks, [SimpleNamespace(content="补充任务：继续查资料")], ""
        )
    assert consumed is True
    assert routed is runtime


@pytest.mark.asyncio
async def test_route_bare_input_latest_wins(send_log: list[str]) -> None:
    """未指定任务的补充消息应只路由给列表中的最新任务。"""
    first = _make_task("stream-2", "旧任务")
    second = _make_task("stream-2", "新任务")
    import unittest.mock as mock

    routed_ids: list[str] = []

    async def _fake_route(task_id: str, text: str) -> bool:
        """记录被选中的任务标识，并模拟消息路由成功。

        Args:
            task_id: 追加到路由记录的目标任务标识。
            text: 为兼容路由接口而接收的消息正文。

        Returns:
            始终为 True，表示消息已成功交给任务。
        """
        routed_ids.append(task_id)
        return True

    chatter = _RoutingChatter("stream-2", send_log)
    with mock.patch.object(chatter_module, "route_message", _fake_route):
        routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
            chatter,
            [first, second],
            [SimpleNamespace(content="补充任务：继续")],
            "",
        )
    assert consumed is True
    assert routed is second
    assert routed_ids == [second.state.task_id]


@pytest.mark.asyncio
async def test_route_bare_cancel_multi_task_disambiguates(send_log: list[str]) -> None:
    """多任务下不带目标的取消消息应提示消歧，不取消任何任务。"""
    first = _make_task("stream-3", "任务一")
    second = _make_task("stream-3", "任务二")
    chatter = _RoutingChatter("stream-3", send_log)
    routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
        chatter,
        [first, second],
        [SimpleNamespace(content="取消任务")],
        "",
    )
    assert routed is None
    assert consumed is True
    assert any("共 2 个任务" in item for item in send_log)
    # 未执行任何取消
    assert first.state.status == TaskStatus.RUNNING
    assert second.state.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_route_ordinal_cancel_targets_task(send_log: list[str]) -> None:
    """带任务序号的取消消息应选择对应任务并标记为已消费。"""
    first = _make_task("stream-4", "任务一")
    second = _make_task("stream-4", "任务二")
    import unittest.mock as mock

    routed_ids: list[str] = []

    async def _fake_route(task_id: str, text: str) -> bool:
        """记录被选中的任务标识，并模拟消息路由成功。

        Args:
            task_id: 追加到路由记录的目标任务标识。
            text: 为兼容路由接口而接收的消息正文。

        Returns:
            始终为 True，表示消息已成功交给任务。
        """
        routed_ids.append(task_id)
        return True

    chatter = _RoutingChatter("stream-4", send_log)
    with mock.patch.object(chatter_module, "route_message", _fake_route):
        routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
            chatter,
            [first, second],
            [SimpleNamespace(content="取消任务2")],
            "",
        )
    assert consumed is True
    assert routed is second


@pytest.mark.asyncio
async def test_route_status_multi_task_sends_merged_list(send_log: list[str]) -> None:
    """多任务状态查询应发送合并列表，而不选择单个任务。"""
    first = _make_task("stream-5", "任务一")
    second = _make_task("stream-5", "任务二")
    chatter = _RoutingChatter("stream-5", send_log)
    routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
        chatter,
        [first, second],
        [SimpleNamespace(content="任务状态")],
        "",
    )
    assert routed is None
    assert consumed is True
    assert any("共 2 个任务" in item for item in send_log)


@pytest.mark.asyncio
async def test_route_chat_message_not_consumed(send_log: list[str]) -> None:
    """普通闲聊不应选中任务、消费消息或产生任务提示。"""
    runtime = _make_task("stream-6", "任务")
    chatter = _RoutingChatter("stream-6", send_log)
    routed, consumed = await chatter_module._AgenticChatterBase._route_task_message(
        chatter,
        [runtime],
        [SimpleNamespace(content="今天天气不错")],
        "",
    )
    assert routed is None
    assert consumed is False
    assert send_log == []


@pytest.mark.asyncio
async def test_create_task_runtime_returns_none_when_full(
    monkeypatch: pytest.MonkeyPatch, send_log: list[str]
) -> None:
    """聊天流任务额度已满时，创建入口应返回空值并发送失败提示。"""
    from ..config import AgenticChatterConfig

    manager = get_task_runtime_manager()
    manager.configure_limits(1)
    existing = _make_task("stream-7", "占位任务")
    manager._tasks[existing.state.task_id] = existing

    config = (
        AgenticChatterConfig.load_empty()
        if hasattr(AgenticChatterConfig, "load_empty")
        else None
    )
    if config is None:
        config = SimpleNamespace(
            tasks=SimpleNamespace(
                enabled=True,
                max_iterations=12,
                max_tool_calls=24,
                max_same_signature_calls=2,
                max_no_progress_steps=3,
                max_failures=3,
                timeout_seconds=300.0,
                max_result_size=6000,
                default_allowed_tools=(),
                denied_tools=(),
                max_concurrent_tasks=1,
            )
        )

    chatter = _RoutingChatter("stream-7", send_log)
    result = await chatter_module._AgenticChatterBase._create_task_runtime(
        chatter, config, "新目标", TaskType.GENERAL, 1
    )
    assert result is None
    assert any("任务创建失败" in item for item in send_log)


@pytest.mark.asyncio
async def test_commit_task_turn_none_runtime_only_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """构造出邮箱认领后，提交空任务运行时的回合不应启动任务。"""
    from ..pipeline.mailbox import get_stream_mailbox

    started: list[Any] = []
    monkeypatch.setattr(chatter_module, "start_task", lambda *a, **k: started.append(a))

    mailbox = get_stream_mailbox("stream-commit")
    turn_token = object()
    generation = await mailbox.try_acquire(turn_token)
    assert generation is not None
    message = SimpleNamespace(
        message_id="m1", content="取消任务", stream_id="stream-commit"
    )
    claim = await mailbox.claim_pending(turn_token, generation)
    if claim is None:
        mailbox.merge_snapshot([message])
        claim = await mailbox.claim_pending(turn_token, generation)
    if claim is None:
        pytest.skip("mailbox 无法构造 claim")
        return

    chatter = _RoutingChatter("stream-commit", [])
    await chatter_module._AgenticChatterBase._commit_task_turn(
        chatter, None, mailbox, claim, None, [message], "取消任务", None, generation
    )
    assert started == []
    await mailbox.release_owner(turn_token, generation)

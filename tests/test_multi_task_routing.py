"""chatter 层多任务路由语义的测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .. import chatter as chatter_module
from ..task_runtime import coordinator
from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime, get_task_runtime_manager


def _make_task(stream_id: str, goal: str, status: TaskStatus = TaskStatus.RUNNING) -> TaskRuntime:
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
        self.stream_id = stream_id
        self.sent = sent


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch):
    manager = get_task_runtime_manager()
    saved = dict(manager._tasks)
    saved_active = dict(coordinator._ACTIVE)
    yield
    manager._tasks.clear()
    manager._tasks.update(saved)
    coordinator._ACTIVE.clear()
    coordinator._ACTIVE.update(saved_active)


@pytest.fixture
def send_log(monkeypatch) -> list[str]:
    log: list[str] = []

    class _SendAPI:
        @staticmethod
        async def send_text(*, content: str, stream_id: str, **kwargs: Any) -> bool:
            log.append(content)
            return True

    monkeypatch.setattr(chatter_module, "send_api", _SendAPI())
    return log


@pytest.mark.asyncio
async def test_route_single_task_input(send_log) -> None:
    runtime = _make_task("stream-1", "唯一任务")
    get_task_runtime_manager()._tasks[runtime.state.task_id] = runtime
    register_calls: list[str] = []
    monkey_route = SimpleNamespace()

    async def _fake_route(task_id: str, text: str) -> bool:
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
async def test_route_bare_input_latest_wins(send_log) -> None:
    first = _make_task("stream-2", "旧任务")
    second = _make_task("stream-2", "新任务")
    import unittest.mock as mock

    routed_ids: list[str] = []

    async def _fake_route(task_id: str, text: str) -> bool:
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
async def test_route_bare_cancel_multi_task_disambiguates(send_log) -> None:
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
async def test_route_ordinal_cancel_targets_task(send_log) -> None:
    first = _make_task("stream-4", "任务一")
    second = _make_task("stream-4", "任务二")
    import unittest.mock as mock

    routed_ids: list[str] = []

    async def _fake_route(task_id: str, text: str) -> bool:
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
async def test_route_status_multi_task_sends_merged_list(send_log) -> None:
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
async def test_route_chat_message_not_consumed(send_log) -> None:
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
async def test_create_task_runtime_returns_none_when_full(monkeypatch, send_log) -> None:
    from ..config import AgenticChatterConfig

    manager = get_task_runtime_manager()
    manager.configure_limits(1)
    existing = _make_task("stream-7", "占位任务")
    manager._tasks[existing.state.task_id] = existing

    config = AgenticChatterConfig.load_empty() if hasattr(
        AgenticChatterConfig, "load_empty"
    ) else None
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
async def test_commit_task_turn_none_runtime_only_commits(monkeypatch) -> None:
    from ..pipeline.mailbox import get_stream_mailbox

    started: list[Any] = []
    monkeypatch.setattr(
        chatter_module, "start_task", lambda *a, **k: started.append(a)
    )

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

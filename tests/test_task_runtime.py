"""复杂任务运行时测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.kernel.llm import LLMPayload, ToolCall

from .. import chatter as chatter_module
from ..task_runtime import (
    JsonCheckpointStore,
    SubtaskSpec,
    TaskBudget,
    TaskCollaboration,
    TaskExecutorFactory,
    TaskMessageKind,
    TaskMessageMailbox,
    TaskResult,
    TaskRuntime,
    TaskRuntimeManager,
    TaskState,
    TaskStatus,
    TaskType,
    ToolPermission,
    ToolPolicy,
    classify_task_message,
)
from ..task_runtime.coordinator import register_task, route_message
from ..task_runtime.executor import TaskCommand, TaskExecutor, TaskRequest


class _StatusExecutor:
    """持有运行中状态并收集路由通知与命令的执行器替身。"""

    def __init__(self) -> None:
        """启动状态查询用的运行时，并初始化通知与命令记录。"""
        self.runtime = TaskRuntime(
            TaskState(stream_id="status-stream", user_goal="状态")
        )
        self.runtime.start()
        self.sent: list[str] = []

    async def send_notice(self, text: str) -> str:
        """记录状态通知并原样返回，供路由结果断言检查。"""
        self.sent.append(text)
        return text

    async def send_command(self, command: TaskCommand) -> None:
        """记录命令种类，以确认补充输入已送达执行器。"""
        self.sent.append(command.kind)


class _StatusTask:
    """模拟尚未完成且已向后台管理器登记的活动任务。"""

    def __init__(self) -> None:
        """组装路由所需的执行器、后台任务句柄和任务请求。"""
        self.executor = _StatusExecutor()
        self.task = SimpleNamespace(done=lambda: False)
        self.task_info_id = "manager-task"
        self.chatter = object()
        self.request = TaskRequest("状态")
        self.store = None


@pytest.fixture
def status_task(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StatusTask]:
    """临时注册状态任务，并在用例结束后移除该任务条目。

    Args:
        monkeypatch: 为活动任务注册表安装可恢复补丁的工具。

    Yields:
        已启动且可记录通知和命令的状态任务替身。
    """
    from ..task_runtime import coordinator

    task = _StatusTask()
    monkeypatch.setitem(coordinator._ACTIVE, "status-task", task)
    yield task
    coordinator._ACTIVE.pop("status-task", None)


@pytest.mark.asyncio
async def test_route_status_sends_current_state(status_task: _StatusTask) -> None:
    """状态查询应被路由接收，并发送包含 running 状态的通知。"""
    routed = await route_message("status-task", "查看任务状态")
    assert routed is True
    assert any("running" in item for item in status_task.executor.sent)


@pytest.mark.asyncio
async def test_route_plain_chat_does_not_enter_task(status_task: _StatusTask) -> None:
    """普通闲聊不进入任务路由，也不向执行器发送通知或命令。"""
    routed = await route_message("status-task", "今天吃什么")
    assert routed is False
    assert status_task.executor.sent == []


@pytest.mark.asyncio
async def test_route_pause_cancels_active_background_task(
    status_task: _StatusTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """暂停消息应暂停运行时，并按管理器任务标识取消后台执行。"""
    from ..task_runtime import coordinator

    cancelled: list[str] = []
    monkeypatch.setattr(
        coordinator,
        "get_task_manager",
        lambda: SimpleNamespace(cancel_task=lambda task_id: cancelled.append(task_id)),
    )

    routed = await route_message("status-task", "暂停任务")

    assert routed is True
    assert status_task.executor.runtime.state.status == TaskStatus.PAUSED
    assert cancelled == ["manager-task"]


@pytest.mark.asyncio
async def test_route_input_reaches_executor(status_task: _StatusTask) -> None:
    """补充资料应进入任务路由，并向执行器传递 input 命令。"""
    routed = await route_message("status-task", "补充资料：使用官方文档")
    assert routed is True
    assert "input" in status_task.executor.sent


async def _send_task_text(**_kwargs: object) -> bool:
    """忽略平台发送参数并返回成功，隔离任务用例中的实际通信。"""
    return True


@pytest.fixture
def task_send_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """将文本发送接口替换为固定成功的异步替身。"""
    monkeypatch.setattr(chatter_module.send_api, "send_text", _send_task_text)


@pytest.fixture
def task_runtime_factory() -> Callable[[], TaskRuntime]:
    """提供创建独立任务运行时的工厂，避免用例之间共享任务状态。"""
    return lambda: TaskRuntime(TaskState(stream_id="task-test", user_goal="执行任务"))


class _TaskResponse:
    """用于任务执行器集成测试的可回灌响应。"""

    def __init__(self, responses: list[tuple[list[ToolCall], str]]) -> None:
        """保存按等待次数消费的工具调用与文本响应序列。

        Args:
            responses: 每次等待时依次读取的工具调用列表与消息文本。
        """
        self._responses = responses
        self._index = 0
        self.call_list: list[ToolCall] = []
        self.message = ""
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """保存执行器注入的消息载荷，承接后续轮次的上下文回灌。"""
        self.payloads.append(payload)

    def __await__(self) -> Generator[Any, None, str]:
        """提供等待迭代器，在等待时消费下一组工具调用和消息。

        Returns:
            更新本轮调用与消息、并以消息文本作为完成值的协程等待迭代器。

        Raises:
            IndexError: 等待迭代器推进时，预设的响应序列已经耗尽。
        """

        async def consume() -> str:
            """推进一次响应游标，并同步本轮工具调用和消息文本。"""
            self.call_list, self.message = self._responses[self._index]
            self._index += 1
            return self.message

        return consume().__await__()

    async def send(self, *, stream: bool = False) -> "_TaskResponse":
        """保持响应链，返回自身以供执行器继续等待下一轮结果。

        Args:
            stream: 兼容请求接口的流式选项；此替身不区分流式模式。

        Returns:
            尚未消费下一项预设结果的当前响应对象。
        """
        return self


class _TaskRequest:
    """返回任务响应序列的假 LLM 请求。"""

    def __init__(self, response: _TaskResponse) -> None:
        """绑定预设响应，并准备收集首轮请求的消息载荷。"""
        self.response = response
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """保存执行器注入的消息载荷，承接后续轮次的上下文回灌。"""
        self.payloads.append(payload)

    async def send(self, *, stream: bool = False) -> _TaskResponse:
        """返回预设响应，代替真实的模型请求。

        Args:
            stream: 兼容请求接口的流式选项；此替身不据此改变响应。

        Returns:
            构造请求替身时绑定的响应对象。
        """
        return self.response


class _TaskChatter:
    """任务执行器所需的最小 Chatter 协议替身。"""

    stream_id = "task-test"

    def __init__(self, response: _TaskResponse) -> None:
        """以预设响应构造请求，并准备记录执行器提交的工具调用。"""
        self.request = _TaskRequest(response)
        self.tool_calls: list[list[ToolCall]] = []

    def create_request(self, **_kwargs: object) -> _TaskRequest:
        """返回预置请求，忽略执行器传入的请求构造选项。"""
        return self.request

    async def get_llm_usables(self) -> list[object]:
        """提供空的工具候选列表，使场景只依赖预设的模型响应。"""
        return []

    async def modify_llm_usables(self, usables: list[object]) -> list[object]:
        """原样保留工具候选列表，不施加额外的筛选或修改。"""
        return usables

    async def run_tool_call(
        self, calls: list[ToolCall], *_args: object
    ) -> list[tuple[str, bool]]:
        """记录本轮工具调用，并为每个调用返回固定成功结果。

        Args:
            calls: 执行器选出的本轮工具调用列表。
            *_args: 为兼容 Chatter 工具调用接口接收的其余参数。

        Returns:
            与调用列表等长的结果列表，每项包含输出文本与成功标记。
        """
        self.tool_calls.append(calls)
        return [("ok", True) for _ in calls]


class _TaskResponse:
    """用于任务执行器集成测试的可回灌响应。"""

    def __init__(self, responses: list[tuple[list[ToolCall], str]]) -> None:
        """保存按等待次数消费的工具调用与文本响应序列。

        Args:
            responses: 每次等待时依次读取的工具调用列表与消息文本。
        """
        self._responses = responses
        self._index = 0
        self.call_list: list[ToolCall] = []
        self.message = ""
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """保存执行器注入的消息载荷，承接后续轮次的上下文回灌。"""
        self.payloads.append(payload)

    def __await__(self) -> Generator[Any, None, str]:
        """提供等待迭代器，在等待时消费下一组工具调用和消息。

        Returns:
            更新本轮调用与消息、并以消息文本作为完成值的协程等待迭代器。

        Raises:
            IndexError: 等待迭代器推进时，预设的响应序列已经耗尽。
        """

        async def consume() -> str:
            """推进一次响应游标，并同步本轮工具调用和消息文本。"""
            self.call_list, self.message = self._responses[self._index]
            self._index += 1
            return self.message

        return consume().__await__()

    async def send(self, *, stream: bool = False) -> "_TaskResponse":
        """保持响应链，返回自身以供执行器继续等待下一轮结果。

        Args:
            stream: 兼容请求接口的流式选项；此替身不区分流式模式。

        Returns:
            尚未消费下一项预设结果的当前响应对象。
        """
        return self


class _TaskRequest:
    """返回任务响应序列的假 LLM 请求。"""

    def __init__(self, response: _TaskResponse) -> None:
        """绑定预设响应，并准备收集首轮请求的消息载荷。"""
        self.response = response
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """保存执行器注入的消息载荷，承接后续轮次的上下文回灌。"""
        self.payloads.append(payload)

    async def send(self, *, stream: bool = False) -> _TaskResponse:
        """返回预设响应，代替真实的模型请求。

        Args:
            stream: 兼容请求接口的流式选项；此替身不据此改变响应。

        Returns:
            构造请求替身时绑定的响应对象。
        """
        return self.response


class _TaskChatter:
    """任务执行器所需的最小 Chatter 协议替身。"""

    stream_id = "task-test"

    def __init__(self, response: _TaskResponse) -> None:
        """以预设响应构造请求，并准备记录执行器提交的工具调用。"""
        self.request = _TaskRequest(response)
        self.tool_calls: list[list[ToolCall]] = []

    def create_request(self, **_kwargs: object) -> _TaskRequest:
        """返回预置请求，忽略执行器传入的请求构造选项。"""
        return self.request

    async def get_llm_usables(self) -> list[object]:
        """提供空的工具候选列表，使场景只依赖预设的模型响应。"""
        return []

    async def modify_llm_usables(self, usables: list[object]) -> list[object]:
        """原样保留工具候选列表，不施加额外的筛选或修改。"""
        return usables

    async def run_tool_call(
        self, calls: list[ToolCall], *_args: object
    ) -> list[tuple[str, bool]]:
        """记录本轮工具调用，并为每个调用返回固定成功结果。

        Args:
            calls: 执行器选出的本轮工具调用列表。
            *_args: 为兼容 Chatter 工具调用接口接收的其余参数。

        Returns:
            与调用列表等长的结果列表，每项包含输出文本与成功标记。
        """
        self.tool_calls.append(calls)
        return [("ok", True) for _ in calls]


def test_executor_runs_tool_loop_and_returns_result(
    task_send_patch: None, task_runtime_factory: Callable[[], TaskRuntime]
) -> None:
    """工具调用回灌后应在第二轮完成，并保留结果摘要和验证证据。"""
    response = _TaskResponse(
        [
            ([ToolCall(id="one", name="test_query", args={})], ""),
            ([], "任务已完成"),
        ]
    )
    chatter = _TaskChatter(response)
    runtime = task_runtime_factory()

    result = asyncio.run(
        TaskExecutor(chatter, runtime).run(
            TaskRequest(objective="执行任务", validation_steps=("运行相关测试",))
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert result.summary == "任务已完成"
    assert result.validation == ("已验证：运行相关测试",)
    assert [call[0].name for call in chatter.tool_calls] == ["test_query"]
    assert runtime.state.iterations == 2


def test_executor_rejects_schema_result_without_marking_success(
    task_send_patch: None, task_runtime_factory: Callable[[], TaskRuntime]
) -> None:
    """不符合结构化结果要求的文本应使任务失败，且不触发工具调用。"""
    response = _TaskResponse([([], "不是 JSON")])
    chatter = _TaskChatter(response)
    runtime = task_runtime_factory()

    result = asyncio.run(
        TaskExecutor(chatter, runtime).run(
            TaskRequest(
                objective="返回结构化结果",
                result_schema={"type": "object", "required": ["answer"]},
            )
        )
    )

    assert result.status == TaskStatus.FAILED
    assert runtime.state.status == TaskStatus.FAILED
    assert chatter.tool_calls == []


def test_executor_validates_nested_result_schema() -> None:
    """嵌套结果校验应拒绝错误的数组元素类型及额外的顶层字段。"""
    schema = {
        "type": "object",
        "required": ["answer"],
        "additionalProperties": False,
        "properties": {
            "answer": {
                "type": "object",
                "required": ["items"],
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            }
        },
    }
    assert TaskExecutor._validate_result('{"answer":{"items":["ok"]}}', schema)
    assert not TaskExecutor._validate_result('{"answer":{"items":[1]}}', schema)
    assert not TaskExecutor._validate_result(
        '{"answer":{"items":[]},"extra":true}', schema
    )


def test_executor_suppresses_delivery_when_deliver_final_text_false(
    task_send_patch: None,
    task_runtime_factory: Callable[[], TaskRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭最终文本投递时仍返回成功摘要，但不向平台发送结果。"""
    sent: list[str] = []

    async def _record_send(**kwargs: object) -> bool:
        """记录发送接口收到的正文并返回成功，以核对最终文本投递次数。"""
        sent.append(str(kwargs.get("content", "")))
        return True

    monkeypatch.setattr(chatter_module.send_api, "send_text", _record_send)
    response = _TaskResponse([([], "最终结果文本")])
    chatter = _TaskChatter(response)
    runtime = task_runtime_factory()

    result = asyncio.run(
        TaskExecutor(chatter, runtime).run(
            TaskRequest(objective="回灌模式", deliver_final_text=False)
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert result.summary == "最终结果文本"
    assert sent == []


def test_executor_delivers_final_text_by_default(
    task_runtime_factory: Callable[[], TaskRuntime], monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认后台执行应将最终结果文本恰好投递一次。"""
    sent: list[str] = []

    async def _record_send(**kwargs: object) -> bool:
        """记录发送接口收到的正文并返回成功，以核对最终文本投递次数。"""
        sent.append(str(kwargs.get("content", "")))
        return True

    monkeypatch.setattr(chatter_module.send_api, "send_text", _record_send)
    response = _TaskResponse([([], "后台任务结果")])
    chatter = _TaskChatter(response)
    runtime = task_runtime_factory()

    result = asyncio.run(
        TaskExecutor(chatter, runtime).run(TaskRequest(objective="后台模式"))
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert sent == ["后台任务结果"]


def test_executor_pause_command_stops_before_next_model_call(
    task_send_patch: None, task_runtime_factory: Callable[[], TaskRuntime]
) -> None:
    """预先收到暂停命令后应返回暂停状态，且不消费任何模型响应。"""
    response = _TaskResponse([([], "不应发送")])
    chatter = _TaskChatter(response)
    runtime = task_runtime_factory()
    executor = TaskExecutor(chatter, runtime)
    runtime.start()
    asyncio.run(executor.send_command(TaskCommand("pause")))

    result = asyncio.run(executor.run(TaskRequest(objective="暂停任务")))

    assert result.status == TaskStatus.PAUSED
    assert runtime.state.status == TaskStatus.PAUSED
    assert response._index == 0


def test_executor_restored_runtime_continues_from_persisted_state(
    tmp_path: Path, task_send_patch: None
) -> None:
    """恢复的运行时应继续完成任务，同时保留已完成步骤和原始开始时间。"""
    from ..task_runtime import TaskRuntime, TaskState, TaskStateStore

    store = TaskStateStore(tmp_path)
    runtime = TaskRuntime(TaskState(stream_id="task-test", user_goal="恢复任务"))
    runtime.start()
    runtime.checkpoint("step-1", "已完成")
    store.save(runtime)
    restored = store.load(runtime.state.task_id)
    assert restored is not None

    response = _TaskResponse([([], "恢复完成")])
    chatter = _TaskChatter(response)
    result = asyncio.run(
        TaskExecutor(chatter, restored, store).run(
            TaskRequest(objective=restored.state.user_goal)
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert restored.state.completed_steps == ["step-1"]
    assert restored.state.started_at == runtime.state.started_at


def test_task_state_store_roundtrip(tmp_path: Path) -> None:
    """状态存储往返应保留任务标识、检查点步骤和开始时间。"""
    from ..task_runtime import TaskStateStore

    store = TaskStateStore(tmp_path)
    runtime = TaskRuntime(TaskState(stream_id="s", user_goal="保存"))
    runtime.start()
    runtime.checkpoint("step", "完成")
    store.save(runtime)
    restored = store.load(runtime.state.task_id)
    assert restored is not None
    assert restored.state.task_id == runtime.state.task_id
    assert restored.state.completed_steps == ["step"]
    assert restored.state.started_at == runtime.state.started_at


def test_task_state_store_restores_result(tmp_path: Path) -> None:
    """已完成任务从存储恢复后应保留原结果及成功状态。"""
    from ..task_runtime import TaskStateStore

    store = TaskStateStore(tmp_path)
    runtime = TaskRuntime(TaskState(stream_id="s", user_goal="完成"))
    runtime.start()
    expected = runtime.complete("已完成")
    store.save(runtime)

    restored = store.load(runtime.state.task_id)

    assert restored is not None
    assert restored.state.result == expected
    assert restored.state.status == TaskStatus.SUCCEEDED


def test_restored_runtime_uses_original_timeout_start(tmp_path: Path) -> None:
    """恢复任务应沿用原始超时起点，已超预算的任务仍须暂停。"""
    from datetime import datetime, timedelta, timezone

    from ..task_runtime import TaskStateStore

    store = TaskStateStore(tmp_path)
    started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    runtime = TaskRuntime(
        TaskState(
            stream_id="s",
            user_goal="超时",
            budget=TaskBudget(timeout_seconds=5),
            started_at=started_at,
        )
    )
    runtime.start()
    store.save(runtime)

    restored = store.load(runtime.state.task_id)

    assert restored is not None
    assert restored.state.started_at == started_at
    assert restored.check_budget() is not None
    assert restored.state.status == TaskStatus.PAUSED


def test_new_manager_restores_active_task(tmp_path: Path) -> None:
    """新建管理器应从存储找回活动任务及其检查点和开始时间。"""
    from ..task_runtime import TaskStateStore

    store = TaskStateStore(tmp_path)
    runtime = TaskRuntime(TaskState(stream_id="restore", user_goal="恢复"))
    runtime.start()
    runtime.checkpoint("step-1", "完成")
    store.save(runtime)

    manager = TaskRuntimeManager(store=store)
    restored = manager.get_active("restore")

    assert restored is not None
    assert restored.state.task_id == runtime.state.task_id
    assert restored.state.completed_steps == ["step-1"]
    assert restored.state.started_at == runtime.state.started_at


@pytest.mark.asyncio
async def test_paused_runtime_rebuilds_request_for_recovery(tmp_path: Path) -> None:
    """暂停任务应按原目标重建请求，并由已完成的占位句柄接收补充输入。"""
    from ..task_runtime import TaskStateStore
    from ..task_runtime.coordinator import get_active_task

    store = TaskStateStore(tmp_path)
    runtime = TaskRuntime(TaskState(stream_id="paused-recover", user_goal="等待恢复"))
    runtime.start()
    runtime.pause()
    store.save(runtime)

    restored = store.load(runtime.state.task_id)
    request = chatter_module._task_request_for_runtime(restored)
    assert request.objective == "等待恢复"

    register_task(object(), restored, request, store)
    active = get_active_task(runtime.state.task_id)

    assert active is not None
    assert active.task.done() is True
    routed = await route_message(runtime.state.task_id, "补充资料：继续")
    assert routed is True

    from ..task_runtime import coordinator

    coordinator.forget_task(runtime.state.task_id)


def test_task_lifecycle_and_checkpoint() -> None:
    """任务成功结束时应把检查点中的已完成步骤与产物写入结果。"""
    runtime = TaskRuntime(TaskState(stream_id="s", user_goal="完成任务"))
    runtime.start()
    runtime.checkpoint("step-1", "已完成", ("a.txt",))
    result = runtime.finish(TaskStatus.SUCCEEDED, "完成")
    assert result.completed_steps == ("step-1",)
    assert result.artifacts == ("a.txt",)
    assert runtime.state.status == TaskStatus.SUCCEEDED


def test_invalid_transition_is_rejected() -> None:
    """未启动的任务不能直接转为成功，非法状态迁移应抛出 ValueError。"""
    state = TaskState(stream_id="s", user_goal="任务")
    with pytest.raises(ValueError, match="不允许"):
        state.transition(TaskStatus.SUCCEEDED)


def test_budget_stops_iterations() -> None:
    """达到迭代预算时应暂停运行时并生成任务结果。"""
    runtime = TaskRuntime(
        TaskState(stream_id="s", user_goal="任务", budget=TaskBudget(max_iterations=1))
    )
    runtime.start()
    assert runtime.record_iteration(True) is not None
    assert runtime.state.status == TaskStatus.PAUSED
    assert runtime.state.result is not None


def test_tool_policy_and_signature_budget() -> None:
    """重复工具签名和禁用工具均被拒绝，只有获准调用计入预算。"""
    runtime = TaskRuntime(
        TaskState(
            stream_id="s",
            user_goal="任务",
            budget=TaskBudget(max_tool_calls=2, max_same_signature_calls=1),
        )
    )
    policy = ToolPolicy(("read*",), denied_tools=("read_secret",))
    assert (
        runtime.check_tool("read_file", "read_file:{}", policy) == ToolPermission.ALLOW
    )
    assert (
        runtime.check_tool("read_file", "read_file:{}", policy) == ToolPermission.DENY
    )
    assert (
        runtime.check_tool("read_secret", "read_secret:{}", policy)
        == ToolPermission.DENY
    )
    assert runtime.state.tool_calls == 1


def test_confirmation_and_default_policy() -> None:
    """策略应区分允许工具、需确认工具及不在白名单内的工具。"""
    policy = ToolPolicy(("edit*",), require_confirmation_tools=("edit_sensitive",))
    assert policy.permission("edit_file") == ToolPermission.ALLOW
    assert policy.permission("edit_sensitive") == ToolPermission.CONFIRM
    assert policy.permission("delete_file") == ToolPermission.DENY


def test_result_is_bounded() -> None:
    """结果长度限制应同时截断摘要与错误文本。"""
    result = TaskResult(TaskStatus.FAILED, "abcdef", error="ghijkl").bounded(3)
    assert result.summary == "abc"
    assert result.error == "ghi"


def test_message_classification_and_mailbox() -> None:
    """消息分类应区分控制、补充和闲聊，邮箱投递应保持任务隔离。"""
    control = classify_task_message("暂停任务", "task-a")
    supplement = classify_task_message("补充资料", "task-b")
    chat = classify_task_message("普通聊天")
    assert control.kind == TaskMessageKind.CONTROL
    assert supplement.kind == TaskMessageKind.INPUT
    assert chat.kind == TaskMessageKind.CHAT

    async def scenario() -> None:
        """分别投递两个任务的消息，核对任务标识隔离和首条消息的代次。"""
        mailbox = TaskMessageMailbox()
        first = await mailbox.put("task-a", "暂停任务")
        second = await mailbox.put("task-b", "补充资料")
        assert first.task_id != second.task_id
        assert mailbox.generation("task-a") == 1

    asyncio.run(scenario())


def test_manager_allows_concurrent_tasks_up_to_limit() -> None:
    """管理器应拒绝超额并发，取消释放名额后可创建任务，调低上限继续生效。"""
    manager = TaskRuntimeManager()
    first = manager.create("stream", "第一个", task_type=TaskType.RESEARCH)
    assert manager.get(first.state.task_id) is first
    second = manager.create("stream", "第二个")
    assert second.state.task_id != first.state.task_id
    with pytest.raises(ValueError, match="已达上限"):
        manager.create("stream", "第三个")
    first.cancel()
    third = manager.create("stream", "第三个")
    assert third.state.task_id != first.state.task_id
    manager.configure_limits(1)
    with pytest.raises(ValueError, match="已达上限"):
        manager.create("stream", "第四个")


def test_completed_task_cannot_be_cancelled() -> None:
    """已成功任务的取消操作应原样返回既有结果并保持成功状态。"""
    runtime = TaskRuntime(TaskState(stream_id="s", user_goal="任务"))
    runtime.start()
    result = runtime.complete("完成")
    assert runtime.cancel() is result
    assert runtime.state.status == TaskStatus.SUCCEEDED


def test_checkpoint_store_roundtrip(tmp_path: Path) -> None:
    """JSON 检查点存储应按任务标识读回与写入值相同的检查点。"""
    from ..task_runtime import TaskCheckpoint

    store = JsonCheckpointStore(tmp_path)
    checkpoint = TaskCheckpoint("task", "step", "completed", "完成", ())
    store.save(checkpoint)
    assert store.load("task") == checkpoint


def test_empty_template_registry_fails_clearly() -> None:
    """空模板注册表查询通用任务模板时应明确抛出 KeyError。"""
    from ..task_runtime import TaskTemplateRegistry

    with pytest.raises(KeyError):
        TaskTemplateRegistry().get(TaskType.GENERAL)


def test_control_status_returns_structured_state() -> None:
    """状态控制命令应返回运行状态、已完成步骤及包含状态名的摘要。"""
    from ..task_runtime import control_task

    manager = TaskRuntimeManager()
    runtime = manager.create("status-stream", "状态任务")
    runtime.start()
    runtime.checkpoint("step-1", "已完成")

    result = control_task(manager, runtime.state.task_id, "status")

    assert result is not None
    assert result.status == TaskStatus.RUNNING
    assert result.completed_steps == ("step-1",)
    assert "running" in result.summary


def test_task_message_plain_text_stays_in_chat() -> None:
    """即使关联任务标识，普通天气问句仍应归类为闲聊。"""
    message = classify_task_message("今天的天气怎么样", "task-a")
    assert message.kind == TaskMessageKind.CHAT


def test_collaboration_stops_when_parent_budget_is_consumed() -> None:
    """父任务已耗尽迭代预算时，子任务执行应返回失败结果。"""
    parent = TaskRuntime(
        TaskState(
            stream_id="s",
            user_goal="父任务",
            budget=TaskBudget(max_iterations=2, max_tool_calls=2, max_failures=2),
        )
    )
    parent.start()
    parent.state.iterations = 2
    factory = TaskExecutorFactory(object(), parent)
    result = asyncio.run(factory.execute(SubtaskSpec("child", "子任务")))
    assert result.status == TaskStatus.FAILED


def test_task_type_marker_and_service_registration() -> None:
    """任务标记应识别为通用任务，插件组件列表应包含运行时服务。"""
    assert chatter_module._detect_task_type("任务: 整理长期资料") == TaskType.GENERAL
    assert chatter_module._detect_task_type("闲聊") is None

    from types import SimpleNamespace

    from ..plugin import AgenticChatterPlugin

    plugin_stub = SimpleNamespace(config=None)
    component_names = {
        getattr(component, "__name__", "")
        for component in AgenticChatterPlugin.get_components(plugin_stub)
    }
    assert "TaskRuntimeService" in component_names


def test_collaboration_passes_dependency_summary() -> None:
    """协作调度应将前置任务摘要传给依赖任务，首个任务的依赖上下文为空。"""
    received: list[str] = []

    async def execute(
        spec: SubtaskSpec,
        *,
        dependency_context: str = "",
    ) -> TaskResult:
        """记录收到的依赖摘要，并以子任务名称构造成功结果。

        Args:
            spec: 当前子任务的描述，其名称用作结果摘要。
            dependency_context: 调度器汇总的前置任务结果文本。

        Returns:
            摘要为当前子任务名称的成功结果。
        """
        received.append(dependency_context)
        return TaskResult(TaskStatus.SUCCEEDED, spec.name)

    result = asyncio.run(
        TaskCollaboration().run(
            (
                SubtaskSpec("research", "查询"),
                SubtaskSpec("review", "复核", ("research",)),
            ),
            execute,
        )
    )
    assert result.status == TaskStatus.SUCCEEDED
    assert received[0] == ""
    assert received[1] == "前置任务 research：research"


def test_collaboration_preserves_structured_results() -> None:
    """协作结果应按任务顺序合并已完成步骤、产物与验证证据。"""

    async def execute(spec: SubtaskSpec) -> TaskResult:
        """为子任务构造同时包含步骤、产物与验证证据的成功结果。

        Args:
            spec: 当前子任务描述，其名称作为各结果字段的标记。

        Returns:
            可用于断言协作结果合并顺序的结构化成功结果。
        """
        return TaskResult(
            TaskStatus.SUCCEEDED,
            spec.name,
            completed_steps=(spec.name,),
            artifacts=(f"{spec.name}.txt",),
            validation=(f"已验证：{spec.name}",),
        )

    result = asyncio.run(
        TaskCollaboration(max_concurrency=2).run(
            (
                SubtaskSpec("research", "查询"),
                SubtaskSpec("review", "复核", ("research",)),
            ),
            execute,
            parent_budget=2,
        )
    )
    assert result.status == TaskStatus.SUCCEEDED
    assert result.completed_steps == ("research", "review")
    assert result.artifacts == ("research.txt", "review.txt")
    assert result.validation == ("已验证：research", "已验证：review")


def test_validation_requires_matching_tool_evidence() -> None:
    """验证步骤应拒绝无关工具证据，并接受 pytest 调用作为运行测试的证据。"""
    from ..task_runtime.executor import TaskExecutor

    executor = TaskExecutor.__new__(TaskExecutor)
    with pytest.raises(ValueError, match="未获得执行证据"):
        asyncio.run(executor._run_validations(("运行相关测试",), ("read_file",)))

    evidence = asyncio.run(executor._run_validations(("运行相关测试",), ("pytest",)))
    assert evidence == ("已验证：运行相关测试",)


def test_schema_prefixed_call_name_passes_policy() -> None:
    """schema 名（tool-xxx）经前缀还原后应通过白名单，与注入侧判定一致。"""
    from ..task_runtime.executor import TaskExecutor, _strip_schema_prefix
    from ..task_runtime.policy import ToolPermission

    assert _strip_schema_prefix("tool-explore_tools") == "explore_tools"
    assert _strip_schema_prefix("action-send_emoji") == "send_emoji"
    assert _strip_schema_prefix("plain_name") == "plain_name"

    executor = TaskExecutor.__new__(TaskExecutor)
    executor.runtime = TaskRuntime(
        TaskState(
            stream_id="s",
            user_goal="权限",
            allowed_tools=("read*", "search*", "list*", "query*", "test*", "explore*"),
        )
    )
    call = ToolCall(id="1", name="tool-explore_tools", args={})
    normal, denied, confirmation = executor._classify_calls([call])
    assert len(normal) == 1
    assert denied == []
    assert confirmation == []

    # 白名单外仍拒绝；黑名单优先级最高
    outside = ToolCall(id="2", name="tool-send_email", args={})
    _, denied, _ = executor._classify_calls([outside])
    assert denied == ["tool-send_email"]
    assert executor.runtime.policy.permission("delete_file") == ToolPermission.DENY


def test_visible_tools_injects_schema_allowed_tools() -> None:
    """注入侧应放行白名单内组件（explore_tools 命中 explore*）。"""
    from ..task_runtime.executor import TaskExecutor

    class _FakeExplore:
        """以探索工具名称和带前缀的 schema 名提供白名单匹配样本。"""

        name = "explore_tools"

        @classmethod
        def get_signature(cls) -> str | None:
            """返回空签名，模拟不提供组件签名的工具。"""
            return None

        @classmethod
        def to_schema(cls) -> dict[str, object]:
            """提供带 tool 前缀的函数 schema，供注入侧按名称注册。"""
            return {"type": "function", "function": {"name": "tool-explore_tools"}}

    class _Registry:
        """仅注册带 schema 前缀的探索工具，供可见工具筛选读取。"""

        _tools = {"tool-explore_tools": _FakeExplore}

    runtime = TaskRuntime(
        TaskState(
            stream_id="s",
            user_goal="注入",
            allowed_tools=("read*", "explore*"),
        )
    )
    executor = TaskExecutor.__new__(TaskExecutor)
    executor.runtime = runtime

    from src.kernel.llm import ToolRegistry

    filtered = executor._visible_tools(_Registry())
    assert isinstance(filtered, ToolRegistry)
    assert filtered.get("tool-explore_tools") is _FakeExplore


def test_visible_tools_excludes_skill_readers_even_when_allowed() -> None:
    """子任务即使配置了 read*，也不注入主回合技能工具。"""
    from ..task_runtime.executor import TaskExecutor
    from ..tooling.skills import ReadSkillReferenceTool, ReadSkillTool

    runtime = TaskRuntime(
        TaskState(stream_id="s", user_goal="读取", allowed_tools=("read*",))
    )
    executor = TaskExecutor.__new__(TaskExecutor)
    executor.runtime = runtime
    registry = type(
        "Registry",
        (),
        {
            "_tools": {
                "read_skill": ReadSkillTool,
                "read_skill_reference": ReadSkillReferenceTool,
            }
        },
    )()
    assert executor._visible_tools(registry).get_all() == []

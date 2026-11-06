"""工具提交超时与真实框架执行任务的取消契约测试。"""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.core.components.base.tool import BaseTool
from src.kernel.concurrency import get_task_manager
from src.kernel.llm import LLMPayload, LLMUsableExecution, ROLE, ToolCall, ToolResult
from src.kernel.llm.usable_execution import execute_usable_instance

from ..chatter import AgenticChatter
from ..config import AgenticChatterConfig
from ..pipeline.state import TurnState
from ..tooling.timed_calls import run_with_timeout, submission_timeout
from ..tooling.timeout import ToolExecutionScope


class _Response:
    """仅提供真实 dispatcher 所用的响应接口。"""

    def __init__(self) -> None:
        """初始化用于记录调度器回写的响应载荷。"""
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload, position: Any = None) -> _Response:
        """记录回写 payload。"""
        self.payloads.append(payload)
        return self

    @property
    def results(self) -> list[ToolResult]:
        """提取结果，便于检测迟到与重复回写。"""
        return [part for payload in self.payloads for part in payload.content]


def _config(timeout: float = 0.04, grace: float = 0.04) -> SimpleNamespace:
    """使用短小但有限的测试预算。"""
    return SimpleNamespace(
        tools=SimpleNamespace(
            tool_timeout_seconds=timeout,
            tool_cancel_grace_seconds=grace,
            tool_timeout_overrides={},
        ),
        tasks=SimpleNamespace(timeout_seconds=300.0),
    )


def _spawn(coroutine: Any) -> asyncio.Task[Any]:
    """通过框架 TaskManager 创建测试任务。"""
    return get_task_manager().create_task(coroutine, daemon=True).task


def _append(response: Any, call: ToolCall, value: str = "ok") -> None:
    """模拟 dispatcher 的唯一结果写入接口。"""
    response.add_payload(
        LLMPayload(ROLE.TOOL_RESULT, ToolResult(value, call.id, call.name))
    )


class _Usable:
    """将测试执行体交给真实 LLMUsableExecution 与 scheduler。"""

    def __init__(self, execution: Any) -> None:
        """保存测试执行体及核心创建的推进任务。"""
        self.execution = execution
        self.tasks: list[asyncio.Task[Any]] = []

    async def execute(self) -> None:
        """提供框架参数检查所需的签名。"""

    def _wrap_execute(self, **kwargs: Any) -> LLMUsableExecution:
        """保留核心执行封装，而非模拟其任务取消行为。"""
        wrapped = LLMUsableExecution(self.execution)
        wrapped.add_task_observer(self.tasks.append)
        return wrapped


@pytest.mark.parametrize("generator", [False, True])
async def test_timeout_cancels_real_execution_and_resumed_tasks(
    generator: bool,
) -> None:
    """真实核心推进任务（含多 yield 换任务）必须在超时后取消。"""
    cancelled = asyncio.Event()

    async def execute() -> tuple[bool, str]:
        """直到被取消才退出。"""
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True, "unreachable"

    async def iterate() -> Any:
        """多次 READY/resume 后挂起。"""
        yield None
        yield (True, "intermediate")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    usable = _Usable(iterate() if generator else execute())
    call = ToolCall("one", "tool-slow", {})
    response = _Response()
    original_factory = asyncio.get_running_loop().get_task_factory()

    async def dispatch(calls: Any, receiver: Any, *args: Any) -> Any:
        """使用 Manager 相同的真实核心入口。"""
        ok, text = await execute_usable_instance(usable)
        _append(receiver, calls[0], str(text))
        return [(True, ok)]

    outcomes = await run_with_timeout(
        dispatch, [call], response, {}, None, config=_config(), stream_id="real"
    )
    assert outcomes == [(True, False)]
    assert cancelled.is_set()
    assert all(task.done() for task in usable.tasks)
    assert len(usable.tasks) == (3 if generator else 1)
    assert len(response.results) == 1
    assert "不能确认回滚" in response.results[0].value
    assert asyncio.get_running_loop().get_task_factory() is original_factory


async def test_batch_preserves_partial_result_only_fills_missing() -> None:
    """已回写结果不重复，未执行或挂起项按 call id 补齐。"""
    calls = [ToolCall(str(i), "tool-same", {}) for i in range(3)]
    response = _Response()
    submissions: list[list[ToolCall]] = []

    async def dispatch(batch: Any, receiver: Any, *args: Any) -> Any:
        """整批只提交一次，第一项返回后挂起。"""
        submissions.append(batch)
        _append(receiver, batch[0], "first completed")
        await asyncio.Event().wait()

    outcomes = await run_with_timeout(
        dispatch, calls, response, {}, None, config=_config(0.015), stream_id="batch"
    )
    assert submissions == [calls]
    assert outcomes == [(True, False)] * 3
    assert [result.call_id for result in response.results] == ["0", "1", "2"]
    assert response.results[0].value == "first completed"
    assert all("已请求取消" in result.value for result in response.results[1:])


async def test_uncooperative_dispatch_is_bounded_and_late_result_is_dropped() -> None:
    """忽略 cancel 的任务不阻塞聊天，释放后也不能写入真实链。"""
    release = asyncio.Event()
    cancellation_seen = asyncio.Event()
    finished = asyncio.Event()
    response = _Response()
    call = ToolCall("late", "tool-late", {})

    async def dispatch(calls: Any, receiver: Any, *args: Any) -> Any:
        """故意拒绝一次取消，随后写迟到结果。"""
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        finally:
            finished.set()
        _append(receiver, calls[0], "late")
        return [(True, True)]

    start = asyncio.get_running_loop().time()
    try:
        assert await run_with_timeout(
            dispatch,
            [call],
            response,
            {},
            None,
            config=_config(0.02, 0.01),
            stream_id="late",
        ) == [(True, False)]
        assert asyncio.get_running_loop().time() - start < 0.5
        assert cancellation_seen.is_set()
        before = list(response.results)
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        await asyncio.sleep(0)
    assert response.results == before


async def test_external_cancel_propagates_and_cancels_real_execution() -> None:
    """调用者取消不是工具超时，不吞掉 CancelledError。"""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def execute() -> None:
        """记录执行任务取消。"""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    usable = _Usable(execute())

    async def dispatch(*args: Any) -> Any:
        """执行真实封装。"""
        return await execute_usable_instance(usable)

    response = _Response()
    task = _spawn(
        run_with_timeout(
            dispatch,
            [ToolCall("cancel", "tool-cancel", {})],
            response,
            {},
            None,
            config=_config(1),
            stream_id="cancel",
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert all(child.done() for child in usable.tasks)
    assert not response.results


async def test_concurrent_scopes_do_not_cancel_other_stream() -> None:
    """先超时的提交不能取消另一个上下文的真实框架执行任务。"""
    release = asyncio.Event()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def execute() -> tuple[bool, str]:
        """独立流等待明确释放。"""
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return True, "other succeeded"

    usable = _Usable(execute())

    async def other_dispatch(calls: Any, receiver: Any, *args: Any) -> Any:
        """独立作用域执行。"""
        ok, text = await execute_usable_instance(usable)
        _append(receiver, calls[0], str(text))
        return [(True, ok)]

    async def slow_dispatch(*args: Any) -> Any:
        """第一条流挂起。"""
        await asyncio.Event().wait()

    response = _Response()
    other = _spawn(
        run_with_timeout(
            other_dispatch,
            [ToolCall("other", "tool-other", {})],
            response,
            {},
            None,
            config=_config(1),
            stream_id="other",
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await run_with_timeout(
            slow_dispatch,
            [ToolCall("slow", "tool-slow", {})],
            _Response(),
            {},
            None,
            config=_config(),
            stream_id="slow",
        )
        assert not cancelled.is_set()
        assert not other.done()
    finally:
        release.set()
        result = await asyncio.wait_for(other, 1)
    assert result == [(True, True)]


async def test_background_inherited_context_is_not_owned_by_parent() -> None:
    """工具创建的独立后台 runner 不因 ContextVar 继承被误杀。"""
    release = asyncio.Event()
    entered = asyncio.Event()
    background: list[asyncio.Task[Any]] = []

    async def execute() -> tuple[bool, str]:
        """后台 runner 中的真实执行任务。"""
        entered.set()
        await release.wait()
        return True, "background"

    usable = _Usable(execute())

    async def dispatch(*args: Any) -> Any:
        """创建普通 TaskManager 后台任务，不属于当前提交执行链。"""
        background.append(_spawn(execute_usable_instance(usable)))
        await entered.wait()
        await asyncio.Event().wait()

    try:
        await run_with_timeout(
            dispatch,
            [ToolCall("parent", "tool-parent", {})],
            _Response(),
            {},
            None,
            config=_config(),
            stream_id="parent",
        )
        assert not background[0].done()
        assert not usable.tasks[0].done()
    finally:
        release.set()
        assert await asyncio.wait_for(background[0], 1) == (True, "background")


@pytest.mark.parametrize("legacy", [False, True])
async def test_factory_chaining_context_and_nested_restore(legacy: bool) -> None:
    """保留已有 factory，正确处理显式空 Context 和嵌套恢复。"""
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()
    marker = contextvars.ContextVar("timeout_test_marker", default="empty")
    marker.set("inherited")
    observed: list[str] = []
    created: list[tuple[Any, asyncio.Task[Any], Any]] = []

    def factory(loop: Any, coroutine: Any, context: Any = None) -> asyncio.Task[Any]:
        """记录原 factory 收到的协程、Context 和返回任务。"""
        task = asyncio.Task(coroutine, loop=loop, context=context)
        created.append((coroutine, task, context))
        return task

    def old_factory(loop: Any, coroutine: Any) -> asyncio.Task[Any]:
        """记录旧式 factory，并通过当前 Context 创建任务。"""
        task = asyncio.Task(coroutine, loop=loop)
        created.append((coroutine, task, contextvars.copy_context()))
        return task

    async def probe() -> None:
        """检查 Task 实际运行上下文。"""
        observed.append(marker.get())

    previous = old_factory if legacy else factory
    loop.set_task_factory(previous)
    try:
        async with ToolExecutionScope():
            installed = loop.get_task_factory()
            async with ToolExecutionScope():
                assert loop.get_task_factory() is installed
                coroutine = probe()
                explicit_context = contextvars.Context()
                task = loop.create_task(coroutine, context=explicit_context)
                await task
                matching = [entry for entry in created if entry[0] is coroutine]
                assert len(matching) == 1
                assert matching[0][1] is task
                if not legacy:
                    assert matching[0][2] is explicit_context
            assert loop.get_task_factory() is installed
        assert loop.get_task_factory() is previous
        assert observed == ["empty"]
    finally:
        loop.set_task_factory(original)


async def test_factory_external_replacement_is_not_overwritten() -> None:
    """scope 结束时不得覆盖第三方刚安装的 factory。"""
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()

    def replacement(loop: Any, coroutine: Any) -> asyncio.Task[Any]:
        """第三方 factory。"""
        return asyncio.Task(coroutine, loop=loop)

    try:
        async with ToolExecutionScope():
            loop.set_task_factory(replacement)
        assert loop.get_task_factory() is replacement
    finally:
        loop.set_task_factory(original)


async def test_success_and_disabled_mode_preserve_outcomes() -> None:
    """正常完成保持状态，关闭保护直接传递原 response。"""
    call = ToolCall("ok", "tool-ok", {})
    response = _Response()

    async def dispatch(calls: Any, receiver: Any, *args: Any) -> Any:
        """正常写回。"""
        _append(receiver, calls[0])
        return [(True, True)]

    assert await run_with_timeout(
        dispatch, [call], response, {}, None, config=_config(), stream_id="ok"
    ) == [(True, True)]
    assert len(response.results) == 1
    disabled = _config(0)
    original = _Response()
    mock = AsyncMock(return_value=[(False, False)])
    assert await run_with_timeout(
        mock, [call], original, {}, None, config=disabled, stream_id="disabled"
    ) == [(False, False)]
    mock.assert_awaited_once_with([call], original, {}, None)


def test_budget_signature_override_batch_and_dispatch_dynamic() -> None:
    """精确覆盖、整批求和和 dispatch_task 动态预算。"""
    config = _config(30)
    normal = SimpleNamespace(get_signature=lambda: "example:tool:slow")
    dispatch = SimpleNamespace(
        get_signature=lambda: "agentic_chatter:action:dispatch_task"
    )
    registry = {"normal": normal, "dispatch": dispatch}
    calls = [ToolCall("a", "normal", {}), ToolCall("b", "dispatch", {})]
    config.tools.tool_timeout_overrides = {"example:tool:slow": 90}
    config.tasks.timeout_seconds = 600
    assert submission_timeout(config, calls, registry) == 720
    config.tools.tool_timeout_overrides["agentic_chatter:action:dispatch_task"] = 12
    assert submission_timeout(config, calls, registry) == 102
    config.tools.tool_timeout_seconds = 0
    assert submission_timeout(config, calls, registry) is None


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_override_config_rejects_invalid_budget(value: float) -> None:
    """覆盖表在配置加载时拒绝无效预算。"""
    with pytest.raises(ValidationError):
        AgenticChatterConfig.ToolsSection(
            tool_timeout_overrides={"example:tool:slow": value}
        )


@pytest.mark.parametrize("field", ["tool_timeout_seconds", "tool_cancel_grace_seconds"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_config_rejects_invalid_global_and_grace(field: str, value: float) -> None:
    """全局预算与宽限不能被 NaN 或无限数悄然禁用。"""
    with pytest.raises(ValidationError):
        AgenticChatterConfig.ToolsSection(**{field: value})


async def test_real_dispatch_manager_contract_and_partial_batch(
    monkeypatch: Any,
) -> None:
    """真实 dispatcher 仍路由 Manager，并按原顺序保留部分结果。"""
    from src.core import managers

    entered: list[str] = []
    cancelled = asyncio.Event()

    class ContractTool(BaseTool):
        """仅用来提供真实组件类型与签名。"""

        name = "timeout_contract"
        description = "timeout contract"

        @classmethod
        def get_signature(cls) -> str:
            """固定组件签名。"""
            return "test:tool:timeout_contract"

        async def execute(self, **kwargs: Any) -> tuple[bool, str]:
            """本测试使用 Manager mock 创建真实执行封装。"""
            return True, "unused"

    async def execute_tool(
        signature: str, plugin: Any, trigger: Any, **kwargs: Any
    ) -> Any:
        """记录真实 dispatcher 的 Manager 路由，并使用其核心执行入口。"""
        entered.append(kwargs["kind"])

        async def execute() -> tuple[bool, str]:
            """首调用完成，第二项挂起，第三项不应启动。"""
            if kwargs["kind"] == "first":
                return True, "completed through manager"
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return True, "unreachable"

        return await execute_usable_instance(_Usable(execute()))

    monkeypatch.setattr(
        managers, "get_tool_use", lambda: SimpleNamespace(execute_tool=execute_tool)
    )
    chatter = AgenticChatter(stream_id="contract", plugin=object())
    calls = [
        ToolCall(kind, "tool-contract", {"kind": kind})
        for kind in ("first", "second", "third")
    ]
    response = _Response()
    result = await run_with_timeout(
        chatter.run_tool_call,
        calls,
        response,
        {"tool-contract": ContractTool},
        SimpleNamespace(stream_id="contract"),
        config=_config(0.015),
        stream_id="contract",
    )
    assert entered == ["first", "second"]
    assert cancelled.is_set()
    assert result == [(True, False)] * 3
    assert len(response.results) == 3
    assert response.results[0].value == "completed through manager"
    assert [part.call_id for part in response.results] == ["first", "second", "third"]


@pytest.mark.parametrize("second_cancel", [False, True])
async def test_uncooperative_real_execution_and_repeated_cancel(
    second_cancel: bool,
) -> None:
    """真实框架子任务拒绝取消时仍有限返回，二次取消不延长宽限。"""
    entered = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    original_factory = asyncio.get_running_loop().get_task_factory()

    async def execute() -> tuple[bool, str]:
        """故意忽略第一次取消，测试结束时明确释放。"""
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        return True, "late execution"

    usable = _Usable(execute())

    async def dispatch(*args: Any) -> Any:
        """与 Manager 相同的核心执行入口。"""
        return await execute_usable_instance(usable)

    task = _spawn(
        run_with_timeout(
            dispatch,
            [ToolCall("stubborn", "tool-stubborn", {})],
            _Response(),
            {},
            None,
            config=_config(0.02, 1 if second_cancel else 0.01),
            stream_id="stubborn",
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if second_cancel:
            task.cancel()
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 0.3)
        else:
            assert await asyncio.wait_for(task, 0.5) == [(True, False)]
        assert cancellation_seen.is_set()
        assert not usable.tasks[-1].done()
        assert asyncio.get_running_loop().get_task_factory() is original_factory
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*usable.tasks), 1)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_dispatch_error_restores_factory() -> None:
    """非超时异常应按原语义抛出且恢复 factory。"""
    original = asyncio.get_running_loop().get_task_factory()

    async def dispatch(*args: Any) -> Any:
        """直接失败。"""
        raise RuntimeError("dispatcher failure")

    with pytest.raises(RuntimeError, match="dispatcher failure"):
        await run_with_timeout(
            dispatch,
            [ToolCall("error", "tool-error", {})],
            _Response(),
            {},
            None,
            config=_config(),
            stream_id="error",
        )
    assert asyncio.get_running_loop().get_task_factory() is original


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="需要 Python 3.12+ 标准 eager factory",
)
@pytest.mark.parametrize("generator", [False, True])
async def test_eager_timeout_cancels_execution_tasks(generator: bool) -> None:
    """在真实 eager factory 下取消核心协程及多 yield 后的推进任务。"""
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()
    cancelled = asyncio.Event()

    async def execute() -> tuple[bool, str]:
        """挂起的真实执行体。"""
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True, "unreachable"

    async def iterate() -> Any:
        """多次 READY 后挂起。"""
        yield None
        yield (True, "intermediate")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    usable = _Usable(iterate() if generator else execute())

    async def dispatch(*args: Any) -> Any:
        """在已登记 dispatcher 中推进真实核心任务。"""
        return await execute_usable_instance(usable)

    loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
    try:
        response = _Response()
        outcomes = await asyncio.wait_for(
            run_with_timeout(
                dispatch,
                [ToolCall("eager", "tool-eager", {})],
                response,
                {},
                None,
                config=_config(),
                stream_id="eager",
            ),
            1,
        )
        assert outcomes == [(True, False)]
        assert cancelled.is_set()
        assert usable.tasks[-1].cancelled()
        assert all(task.done() for task in usable.tasks)
        assert len(usable.tasks) == (3 if generator else 1)
        assert loop.get_task_factory() is getattr(asyncio, "eager_task_factory")
    finally:
        loop.set_task_factory(original)
        for task in usable.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*usable.tasks, return_exceptions=True)


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="需要 Python 3.12+ 标准 eager factory",
)
@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "error",
        "external_cancel",
        "repeat_cancel",
        "concurrent",
        "background",
    ],
)
async def test_eager_preserves_completion_and_isolation(scenario: str) -> None:
    """在真实 eager 调度下重复现有完成、取消与隔离契约。"""
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()
    loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
    try:
        if scenario == "success":
            await test_success_and_disabled_mode_preserve_outcomes()
        elif scenario == "error":
            await test_dispatch_error_restores_factory()
        elif scenario == "external_cancel":
            await test_external_cancel_propagates_and_cancels_real_execution()
        elif scenario == "repeat_cancel":
            await test_uncooperative_real_execution_and_repeated_cancel(True)
        elif scenario == "concurrent":
            await test_concurrent_scopes_do_not_cancel_other_stream()
        else:
            await test_background_inherited_context_is_not_owned_by_parent()
        assert loop.get_task_factory() is getattr(asyncio, "eager_task_factory")
    finally:
        loop.set_task_factory(original)


async def test_track_is_idempotent_and_skips_finished_task() -> None:
    """同一任务只登记一次，已完成任务不加入待取消集合。"""
    release = asyncio.Event()
    task = _spawn(release.wait())
    try:
        async with ToolExecutionScope() as scope:
            scope.track(task)
            callbacks = list(task._callbacks or [])
            scope.track(task)
            assert list(task._callbacks or []) == callbacks
            assert task in scope._tasks
            release.set()
            await task
            await asyncio.sleep(0)
            scope.track(task)
            assert task not in scope._tasks
            assert await scope.cancel_pending(0) == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_planning_continues_after_timeout_without_counting_failure() -> None:
    """主聊天首个调用超时后仍继续下一调用，失败不计成功进展。"""
    config = AgenticChatterConfig()
    config.tools.tool_timeout_seconds = 0.02
    config.tools.tool_cancel_grace_seconds = 0.01
    chatter = AgenticChatter(
        stream_id="planning-timeout", plugin=SimpleNamespace(config=config)
    )
    calls = [ToolCall("slow", "tool-slow", {}), ToolCall("fast", "tool-fast", {})]
    submitted: list[str] = []

    async def dispatch(batch: Any, receiver: Any, *args: Any) -> Any:
        """首项超时，下一项返回成功。"""
        submitted.append(batch[0].id)
        if batch[0].id == "slow":
            await asyncio.Event().wait()
        _append(receiver, batch[0], "fast result")
        return [(True, True)]

    chatter.run_tool_call = dispatch
    state = TurnState(stream_id="planning-timeout")
    response = _Response()
    count = await chatter._execute_calls(calls, response, state, {}, [])
    assert submitted == ["slow", "fast"]
    assert count == 1
    assert [record.outcome for record in state.tool_ledger] == ["failure", "success"]
    assert [part.call_id for part in response.results] == ["slow", "fast"]

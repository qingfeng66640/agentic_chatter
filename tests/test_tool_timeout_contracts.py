"""嵌套 eager、工厂参数和取消清理的真实框架契约回归。"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from typing import Any

import pytest

from src.kernel.llm import LLMUsableExecution, ToolCall
from src.kernel.llm.usable_execution import execute_usable_instance

from ..tooling.timed_calls import run_with_timeout
from ..tooling.timeout import ToolExecutionScope
from .test_tool_timeout import _config, _Response, _spawn, _Usable


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("iterator", [False, True])
async def test_nested_execution_cancelled_at_all_depths(
    eager: bool,
    depth: int,
    iterator: bool,
) -> None:
    """核心真实调度器逐层嵌套：超时后释放门闩也不得产生迟到副作用。"""
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("标准 eager factory 需要 Python 3.12+")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    gate = asyncio.Event()
    effects: list[str] = []
    original_codes: list[Any] = []

    async def leaf() -> Any:
        """最内层挂起。"""
        await gate.wait()
        effects.append("late")
        return True, "ok"

    async def leaf_iterator() -> Any:
        """内层多次 yield 后挂起。"""
        yield None
        yield True, "partial"
        await gate.wait()
        effects.append("late")

    layers = [_Usable(leaf_iterator() if iterator else leaf())]
    for _ in range(depth - 1):

        async def outer(child: _Usable = layers[-1]) -> Any:
            """外层直接等待内层框架执行。"""
            return await execute_usable_instance(child)

        layers.append(_Usable(outer()))

    def factory(current_loop: Any, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        """按原推进协程 code 观测，断言没有插件 wrapper 改变身份。"""
        original_codes.append(coro.cr_code)
        if eager:
            return getattr(asyncio, "eager_task_factory")(current_loop, coro, **kwargs)
        return asyncio.Task(coro, loop=current_loop, **kwargs)

    async def dispatch(*args: Any) -> Any:
        """通过真实统一执行器运行最外层。"""
        ok, _ = await execute_usable_instance(layers[-1])
        return [(True, ok)]

    loop.set_task_factory(factory)
    try:
        assert await asyncio.wait_for(
            run_with_timeout(
                dispatch,
                [ToolCall("nested", "tool-nested", {})],
                _Response(),
                {},
                None,
                config=_config(0.04, 0.1),
                stream_id="nested-contract",
            ),
            2,
        ) == [(True, False)]
        assert all(layer.tasks[-1].cancelled() for layer in layers)
        assert all(task.done() for layer in layers for task in layer.tasks)
        assert LLMUsableExecution._await_result.__code__ in original_codes
        if iterator:
            assert LLMUsableExecution._advance_iterator.__code__ in original_codes
        gate.set()
        await asyncio.sleep(0)
        assert effects == []
        assert loop.get_task_factory() is factory
    finally:
        loop.set_task_factory(previous)
        tasks = [task for layer in layers for task in layer.tasks]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("grace", [0, 0.1])
async def test_cancel_before_execution_first_step_closes_business(grace: float) -> None:
    """推进 Task 从未启动时也释放原业务 coroutine 并完成框架状态。"""
    entered: list[bool] = []

    async def business() -> None:
        """取消前不应进入业务体。"""
        entered.append(True)

    original = business()
    async with ToolExecutionScope() as scope:
        execution = LLMUsableExecution(original)
        await scope.cancel_pending(grace)
        await asyncio.gather(execution.task, return_exceptions=True)
        await asyncio.sleep(0)
    assert entered == []
    assert inspect.getcoroutinestate(original) == inspect.CORO_CLOSED
    assert execution._status == "_DONE"


@pytest.mark.parametrize("grace", [0, 0.1])
async def test_cancel_before_generator_resume_closes_generator(grace: float) -> None:
    """首次推进前取消也运行 finally；副作用与 Task 状态均不能被 gather 掩盖。"""
    finalized = asyncio.Event()
    effects: list[str] = []

    async def generator() -> Any:
        """留在 READY，下一次推进不应执行。"""
        try:
            yield None
            effects.append("unexpected second step")
            raise AssertionError("不应执行第二步")
        finally:
            finalized.set()

    original = generator()
    async with ToolExecutionScope() as scope:
        execution = LLMUsableExecution(original)
        await execution.task
        assert execution._status == "_READY"
        execution.resume()
        second_task = execution.task
        await scope.cancel_pending(grace)
        # 零宽限只请求取消；等待 Task 收敛后检查状态，不忽略 gather 转换的异常。
        await asyncio.wait_for(asyncio.gather(second_task, return_exceptions=True), 1)
        assert effects == []
        assert second_task.cancelled()
        await asyncio.wait_for(finalized.wait(), 1)
    assert original.ag_frame is None
    assert execution._status == "_DONE"


async def test_unstarted_external_future_is_not_cancelled_by_cleanup() -> None:
    """scope 清理不擅自关闭尚未被核心 await 的共享 Future。"""
    future = asyncio.get_running_loop().create_future()
    async with ToolExecutionScope() as scope:
        execution = LLMUsableExecution(future)
        await scope.cancel_pending(0.1)
    assert not future.cancelled()
    future.set_result("shared")
    assert execution._status == "_DONE"


async def test_factory_error_closes_unstarted_business_without_retry() -> None:
    """原 factory 拒绝推进创建时只调用一次且释放两层协程。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    calls: list[Any] = []

    def rejecting_factory(loop: Any, coro: Any, **kwargs: Any) -> Any:
        """真实构造错误不得被当成签名兼容问题重试。"""
        calls.append(coro)
        raise TypeError("factory internal failure")

    async def business() -> None:
        """不应开始。"""
        raise AssertionError("must not start")

    original = business()
    loop.set_task_factory(rejecting_factory)
    try:
        async with ToolExecutionScope():
            with pytest.raises(TypeError, match="factory internal failure"):
                LLMUsableExecution(original)
        assert len(calls) == 1
        assert inspect.getcoroutinestate(calls[0]) == inspect.CORO_CLOSED
        assert inspect.getcoroutinestate(original) == inspect.CORO_CLOSED
    finally:
        loop.set_task_factory(previous)
        original.close()


async def test_cancelling_execution_rejects_new_business_before_eager_start() -> None:
    """取消 finally 内的新框架执行不进入业务体，也不遗留原协程。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    if hasattr(asyncio, "eager_task_factory"):
        loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
    bodies: list[Any] = []
    effects: list[str] = []

    async def inner() -> None:
        """必须在开始前被拒绝。"""
        effects.append("bad")

    async def outer() -> None:
        """收到取消后尝试创建新的框架任务。"""
        try:
            await asyncio.Event().wait()
        finally:
            body = inner()
            bodies.append(body)
            LLMUsableExecution(body)

    usable = _Usable(outer())

    async def dispatch(*args: Any) -> Any:
        """执行真实任务。"""
        return await execute_usable_instance(usable)

    try:
        await run_with_timeout(
            dispatch,
            [ToolCall("cancel", "tool-cancel", {})],
            _Response(),
            {},
            None,
            config=_config(),
            stream_id="cancel-new",
        )
        assert bodies and effects == []
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in bodies
        )
        assert all(task.done() for task in usable.tasks)
    finally:
        loop.set_task_factory(previous)
        for task in usable.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*usable.tasks, return_exceptions=True)


@pytest.mark.parametrize("name", [None, "dispatcher"])
async def test_modern_factory_forwards_all_kwargs_and_original_coroutine(
    name: Any,
) -> None:
    """模拟新版事件循环的参数转发契约，不替代目标 Python 版本的运行测试。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    seen: list[Any] = []
    context = contextvars.copy_context()

    def factory(loop: Any, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        """任意参数原样到达原 factory，原 factory 自行消费扩展参数。"""
        seen.append((coro, dict(kwargs)))
        assert kwargs.pop("custom_flag") is True
        return asyncio.Task(coro, loop=loop, **kwargs)

    async def probe() -> str:
        """普通协程，不改变其身份。"""
        return "ok"

    loop.set_task_factory(factory)
    try:
        async with ToolExecutionScope():
            coroutine = probe()
            task = loop.get_task_factory()(
                loop, coroutine, name=name, context=context, custom_flag=True
            )
            assert await task == "ok"
            assert seen == [
                (coroutine, {"name": name, "context": context, "custom_flag": True})
            ]
            if name is not None:
                assert task.get_name() == name
    finally:
        loop.set_task_factory(previous)


async def test_legacy_factory_name_context_and_unknown_parameter() -> None:
    """旧 factory 只适配可保真的 name/context；未知参数不静默丢弃。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    marker = contextvars.ContextVar("factory-contract", default="empty")
    token = marker.set("outer")
    calls: list[Any] = []

    def legacy(loop: Any, coro: Any) -> asyncio.Task[Any]:
        """只接受两个参数。"""
        calls.append(coro)
        return asyncio.Task(coro, loop=loop)

    async def probe() -> str:
        """返回实际运行 Context。"""
        return marker.get()

    loop.set_task_factory(legacy)
    try:
        async with ToolExecutionScope():
            factory = loop.get_task_factory()
            task = factory(
                loop, probe(), name="legacy-name", context=contextvars.Context()
            )
            assert await task == "empty"
            assert task.get_name() == "legacy-name"
            coro = probe()
            try:
                with pytest.raises(TypeError, match="custom_flag"):
                    factory(loop, coro, custom_flag=True)
            finally:
                coro.close()
            assert len(calls) == 1
    finally:
        loop.set_task_factory(previous)
        marker.reset(token)


@pytest.mark.parametrize("grace", [0, 0.03])
async def test_ready_generator_cleanup_obeys_grace(grace: float) -> None:
    """READY 生成器清理拒绝一次取消；释放门闩前必须证明取消实际送达。"""
    release = asyncio.Event()
    entered = asyncio.Event()
    cancel_received = asyncio.Event()
    finished = asyncio.Event()

    async def generator() -> Any:
        """finally 故意阻塞以模拟第三方清理。"""
        try:
            yield None
        finally:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancel_received.set()
                await release.wait()
            finished.set()

    original = generator()
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        async with ToolExecutionScope() as scope:
            execution = LLMUsableExecution(original)
            await execution.task
            pending = await scope.cancel_pending(grace)
            assert pending >= 1
            assert loop.time() - start < 0.5
        await asyncio.wait_for(entered.wait(), 1)
        # 验证 CancelledError 已被处理
        await asyncio.wait_for(cancel_received.wait(), 1)
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        await original.aclose()
    assert original.ag_frame is None
    assert execution._status == "_DONE"


async def test_generator_cleanup_after_factory_resume_error() -> None:
    """resume 创建失败后不能只关闭推进 coroutine 而遗留已 yield 的生成器。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    count = 0
    finalized = asyncio.Event()

    def factory(loop: Any, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        """仅拒绝第二次推进，清理任务仍按原 factory 创建。"""
        nonlocal count
        if coro.cr_code is LLMUsableExecution._advance_iterator.__code__:
            count += 1
            if count == 2:
                raise RuntimeError("resume rejected")
        return asyncio.Task(coro, loop=loop, **kwargs)

    async def generator() -> Any:
        """暂停后必须明确关闭。"""
        try:
            yield None
        finally:
            finalized.set()

    original = generator()
    loop.set_task_factory(factory)
    try:
        async with ToolExecutionScope() as scope:
            execution = LLMUsableExecution(original)
            await execution.task
            with pytest.raises(RuntimeError, match="resume rejected"):
                execution.resume()
            assert await scope.wait_cleanup(0.1) == 0
            assert finalized.is_set()
            assert execution._status == "_DONE"
    finally:
        loop.set_task_factory(previous)
        await original.aclose()


async def test_execution_moved_to_another_scope_is_not_closed_by_old_owner() -> None:
    """已暂停 execution 被另一作用域 resume 后，旧资源记录放弃清理权。"""
    release = asyncio.Event()
    entered = asyncio.Event()
    finished = asyncio.Event()

    async def generator() -> Any:
        """第一次 yield 后由另一任务继续推进。"""
        try:
            yield None
            entered.set()
            await release.wait()
            yield True, "other-owner"
        finally:
            finished.set()

    async def resume_elsewhere(execution: LLMUsableExecution) -> None:
        """新作用域获得推进及清理归属。"""
        async with ToolExecutionScope():
            execution.resume()
            await execution.task
            await execution.wait_done()

    original = generator()
    other = None
    try:
        async with ToolExecutionScope() as first:
            execution = LLMUsableExecution(original)
            await execution.task
            other = _spawn(resume_elsewhere(execution))
            await asyncio.wait_for(entered.wait(), 1)
            await first.cancel_pending(0.02)
            assert not other.done()
            assert not finished.is_set()
        release.set()
        await asyncio.wait_for(other, 1)
        assert finished.is_set()
    finally:
        release.set()
        if other is not None:
            await asyncio.gather(other, return_exceptions=True)
        await original.aclose()


async def test_track_consumes_previously_unretrieved_failure() -> None:
    """先证明异常未被任何路径消费，再验证 track；不能先 await/gather。"""

    async def fail() -> None:
        """真实失败 Task。"""
        raise RuntimeError("expected test exception")

    task = _spawn(fail())
    try:
        await asyncio.wait({task})
        assert task._log_traceback is True
        async with ToolExecutionScope() as scope:
            scope.track(task)
            assert task._log_traceback is False
            assert task not in scope._tasks
    finally:
        await asyncio.gather(task, return_exceptions=True)

"""工具清理生命周期测试：框架推进、跨作用域归属及异常安全取消。"""

from __future__ import annotations

import asyncio
import gc
import inspect
import weakref
from typing import Any

import pytest

from src.kernel.llm import LLMUsableExecution, ToolCall
from src.kernel.llm.usable_execution import execute_usable_instance

from ..tooling import execution_cleanup, timeout
from ..tooling.execution_cleanup import ExecutionCleanup
from ..tooling.timed_calls import run_with_timeout
from ..tooling.timeout import ToolExecutionScope
from .test_tool_timeout import _config, _Response, _Usable


class _ClosingIterator:
    """可控的手写迭代器或原生生成器代理，计数每次 aclose 调用及其并发。"""

    def __init__(self, native: bool = False) -> None:
        """初始化清理门闩、调用计数与可选的原生生成器代理。"""
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()
        self.steps = 0
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.native = self._generate() if native else None

    def __aiter__(self) -> _ClosingIterator:
        """保持代理自身以便记录真正的 aclose 调用次数。"""
        return self

    async def __anext__(self) -> None:
        """推进一次；native 分支实际进入原生异步生成器。"""
        self.steps += 1
        if self.native is not None:
            return await anext(self.native)
        return None

    async def _generate(self) -> Any:
        """在原生生成器 finally 中执行受控异步释放。"""
        try:
            yield None
            yield None
        finally:
            await self._finish()

    async def _finish(self) -> None:
        """收到一次取消后继续等待门闩，模拟协作取消不能强杀的清理。"""
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        self.finished.set()

    async def aclose(self) -> None:
        """计数调用而非 finally 次数，确保原生生成器重复关闭也能检出。"""
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.native is not None:
                await self.native.aclose()
            else:
                await self._finish()
        finally:
            self.active -= 1


async def _settle(tasks: list[asyncio.Task[Any]]) -> None:
    """有限等待已发出取消或已释放门闩的任务；不主动发取消以免掩盖断言。"""
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("grace", [0, 0.05])
async def test_cleanup_factory_failure_does_not_skip_execution_cancel(
    monkeypatch: pytest.MonkeyPatch,
    eager: bool,
    grace: float,
) -> None:
    """真实提交同时持有 READY 和 running 对象，拒绝清理创建不能跳过取消。"""
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("标准 eager factory 需要 Python 3.12+")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    gate = asyncio.Event()
    effects: list[str] = []
    rejected: list[Any] = []
    warnings: list[str] = []
    dispatchers: list[asyncio.Task[Any]] = []
    monkeypatch.setattr(execution_cleanup.logger, "warning", warnings.append)

    async def paused() -> Any:
        """仅由显式 aclose 释放，拒绝创建时不假称已释放。"""
        try:
            yield None
        finally:
            effects.append("closed")

    async def running() -> tuple[bool, str]:
        """通过真实统一执行器推进，取消后放行也不得产生副作用。"""
        await gate.wait()
        effects.append("late")
        return True, "late"

    original = paused()
    usable = _Usable(running())

    def factory(current_loop: Any, coro: Any, **kwargs: Any) -> Any:
        """只拒绝业务清理任务，其他任务按选定普通/eager 机制创建。"""
        if coro.cr_code is ExecutionCleanup._close_iterator.__code__:
            rejected.append(coro)
            raise RuntimeError("cleanup rejected")
        if eager:
            return getattr(asyncio, "eager_task_factory")(current_loop, coro, **kwargs)
        return asyncio.Task(coro, loop=current_loop, **kwargs)

    async def dispatch(*args: Any) -> Any:
        """READY 对象保留于同一 scope；另一业务通过真实核心执行器挂起。"""
        dispatchers.append(asyncio.current_task())
        ready = LLMUsableExecution(original)
        await ready.task
        assert ready._status == "_READY"
        return [await execute_usable_instance(usable)]

    loop.set_task_factory(factory)
    try:
        result = await asyncio.wait_for(
            run_with_timeout(
                dispatch,
                [ToolCall("reject", "tool-reject", {})],
                _Response(),
                {},
                None,
                config=_config(0.04, grace),
                stream_id="cleanup-reject",
            ),
            2,
        )
        assert result == [(True, False)]
        await _settle(dispatchers + usable.tasks)
        assert dispatchers and all(task.cancelled() for task in dispatchers)
        assert usable.tasks and all(task.cancelled() for task in usable.tasks)
        assert len(rejected) == 1
        assert inspect.getcoroutinestate(rejected[0]) == inspect.CORO_CLOSED
        assert sum("事件=工具清理创建失败" in msg for msg in warnings) == 1
        assert "closed" not in effects
        gate.set()
        await asyncio.sleep(0)
        assert "late" not in effects
        assert loop.get_task_factory() is factory
    finally:
        loop.set_task_factory(previous)
        for task in dispatchers + usable.tasks:
            if not task.done():
                task.cancel()
        await _settle(dispatchers + usable.tasks)
        await original.aclose()
        usable.execution.close()


async def test_done_callback_cleanup_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """推进前取消通过 done 回调清理，创建异常只记一次且不泄漏到 loop。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    handler = loop.get_exception_handler()
    errors: list[Any] = []
    warnings: list[str] = []
    monkeypatch.setattr(execution_cleanup.logger, "warning", warnings.append)
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    iterator = _ClosingIterator()

    def factory(current_loop: Any, coro: Any, **kwargs: Any) -> Any:
        """仅拒绝清理创建。"""
        if coro.cr_code is ExecutionCleanup._close_iterator.__code__:
            raise RuntimeError("callback cleanup rejected")
        return asyncio.Task(coro, loop=current_loop, **kwargs)

    loop.set_task_factory(factory)
    execution = None
    try:
        async with ToolExecutionScope() as scope:
            execution = LLMUsableExecution(iterator)
            await execution.task
            execution.resume()
            await scope.cancel_pending(0)
            await _settle([execution.task])
            await asyncio.sleep(0)
        assert execution.task.cancelled()
        assert execution._status == "_DONE"
        assert iterator.calls == 0
        assert sum("事件=工具清理创建失败" in msg for msg in warnings) == 1
        assert errors == []
    finally:
        loop.set_task_factory(previous)
        loop.set_exception_handler(handler)
        if execution is not None:
            if not execution.task.done():
                execution.task.cancel()
            await _settle([execution.task])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("eager", [False, True])
async def test_cross_scope_failed_resume_claims_cleanup_once(
    native: bool, eager: bool
) -> None:
    """两 scope 共用旧 Task 时，恢复失败仍终止且只能发起一次关闭。"""
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("标准 eager factory 需要 Python 3.12+")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    reject = False
    iterator = _ClosingIterator(native)
    cleanup_tasks: list[asyncio.Task[Any]] = []

    def factory(current_loop: Any, coro: Any, **kwargs: Any) -> Any:
        """恢复创建失败时保留旧 task；清理仍可正常开始。"""
        if reject and coro.cr_code is LLMUsableExecution._advance_iterator.__code__:
            raise RuntimeError("resume rejected")
        if eager:
            return getattr(asyncio, "eager_task_factory")(current_loop, coro, **kwargs)
        return asyncio.Task(coro, loop=current_loop, **kwargs)

    loop.set_task_factory(factory)
    try:
        async with ToolExecutionScope() as first:
            execution = LLMUsableExecution(iterator)
            await execution.task
            old_task = execution.task
            reject = True
            async with ToolExecutionScope() as second:
                with pytest.raises(RuntimeError, match="resume rejected"):
                    execution.resume()
                # 必须先检查 DONE，不能对旧 WORKING 空转状态调用 wait_done。
                assert execution._status == "_DONE"
                assert execution.task is old_task
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(execution.wait_done(), 1)
                cleanup_tasks.extend(second._cleanup_tasks)
                await asyncio.wait_for(iterator.entered.wait(), 1)
            await first.cancel_pending(0)
            cleanup_tasks.extend(first._cleanup_tasks)
            await asyncio.sleep(0)
            assert iterator.calls == 1
            assert iterator.max_active == 1
            assert not iterator.finished.is_set()
        iterator.release.set()
        await _settle(cleanup_tasks)
        assert iterator.finished.is_set()
        assert execution._status == "_DONE"
        if iterator.native is not None:
            assert iterator.native.ag_frame is None
    finally:
        loop.set_task_factory(previous)
        iterator.release.set()
        await _settle(cleanup_tasks)
        if iterator.native is not None:
            await iterator.native.aclose()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("grace", [0, 0.03])
async def test_closing_execution_cannot_resume(native: bool, grace: float) -> None:
    """慢关闭吞一次取消后，另一 scope 的 resume 不得执行或覆盖终态。"""
    iterator = _ClosingIterator(native)
    cleanup_tasks: list[asyncio.Task[Any]] = []
    try:
        async with ToolExecutionScope() as first:
            execution = LLMUsableExecution(iterator)
            await execution.task
            original_task = execution.task
            await first.cancel_pending(grace)
            cleanup_tasks.extend(first._cleanup_tasks)
            assert execution._status == "_DONE"
            cancellation = execution.exception
        await asyncio.wait_for(iterator.entered.wait(), 1)
        await asyncio.wait_for(iterator.cancelled.wait(), 1)
        async with ToolExecutionScope():
            execution.resume()  # 核心在非 READY 状态下无操作，不抛异常。
            assert execution.task is original_task
            assert iterator.steps == 1
            assert execution.exception is cancellation
        iterator.release.set()
        await _settle(cleanup_tasks)
        assert iterator.finished.is_set()
        assert execution._status == "_DONE"
        assert execution.exception is cancellation
        assert iterator.calls == iterator.max_active == 1
    finally:
        iterator.release.set()
        await _settle(cleanup_tasks)
        if iterator.native is not None:
            await iterator.native.aclose()


class _CleanupAbort(BaseException):
    """不会像 KeyboardInterrupt 一样中断测试运行器的 BaseException。"""


@pytest.mark.parametrize(
    "error_type", [RuntimeError, _CleanupAbort, asyncio.CancelledError]
)
async def test_abort_failure_still_cancels_all_tasks(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    """普通异常继续逐资源清理；BaseException 传播也不得绕过取消和恢复。"""
    previous = asyncio.get_running_loop().get_task_factory()
    warnings: list[str] = []
    monkeypatch.setattr(timeout.logger, "warning", warnings.append)
    reached: list[bool] = []
    tasks: list[asyncio.Task[Any]] = []

    async def business() -> None:
        """等待取消，无外部副作用。"""
        await asyncio.Event().wait()

    try:
        async with ToolExecutionScope() as scope:
            executions = [LLMUsableExecution(business()) for _ in range(2)]
            tasks = [execution.task for execution in executions]
            first = scope._resources[executions[0]]
            second = scope._resources[executions[1]]
            original_abort = second.abort

            def fail() -> None:
                """首个资源模拟任意清理异常。"""
                raise error_type("abort rejected")

            def observe() -> None:
                """证明普通异常不会跳过后续资源。"""
                reached.append(True)
                original_abort()

            monkeypatch.setattr(first, "abort", fail)
            monkeypatch.setattr(second, "abort", observe)
            if error_type is RuntimeError:
                await scope.cancel_pending(0.03)
                assert reached == [True]
                assert len(warnings) == 1
            else:
                with pytest.raises(error_type, match="abort rejected"):
                    await scope.cancel_pending(0.03)
            assert scope._closed
            assert asyncio.get_running_loop().get_task_factory() is previous
            await _settle(tasks)
            assert all(task.cancelled() for task in tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await _settle(tasks)


async def test_cleanup_creation_failure_is_not_retried_by_old_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败也是一次认领：旧 scope 不重复创建，也不重复记录失败日志。"""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    reject = False
    attempts: list[Any] = []
    warnings: list[str] = []
    iterator = _ClosingIterator()
    monkeypatch.setattr(execution_cleanup.logger, "warning", warnings.append)

    def factory(current_loop: Any, coro: Any, **kwargs: Any) -> Any:
        """拒绝恢复及随后的清理创建。"""
        if coro.cr_code is ExecutionCleanup._close_iterator.__code__:
            attempts.append(coro)
            raise RuntimeError("cleanup rejected")
        if reject and coro.cr_code is LLMUsableExecution._advance_iterator.__code__:
            raise RuntimeError("resume rejected")
        return asyncio.Task(coro, loop=current_loop, **kwargs)

    loop.set_task_factory(factory)
    try:
        async with ToolExecutionScope() as first:
            execution = LLMUsableExecution(iterator)
            await execution.task
            reject = True
            async with ToolExecutionScope():
                with pytest.raises(RuntimeError, match="resume rejected"):
                    execution.resume()
                assert execution._status == "_DONE"
            await first.cancel_pending(0.03)
        assert len(attempts) == 1
        assert inspect.getcoroutinestate(attempts[0]) == inspect.CORO_CLOSED
        assert len(warnings) == 1 and "事件=工具清理创建失败" in warnings[0]
        assert iterator.calls == 0
    finally:
        loop.set_task_factory(previous)


async def test_cleanup_claim_does_not_keep_execution_alive() -> None:
    """解除 scope、任务回调及业务引用后，一次性认领不会阻止 execution 回收。"""
    iterator = _ClosingIterator()
    iterator.release.set()
    async with ToolExecutionScope() as scope:
        execution = LLMUsableExecution(iterator)
        await execution.task
        await scope.cancel_pending(0.1)
        assert execution in execution_cleanup._CLEANUP_CLAIMS
        reference = weakref.ref(execution)
    del execution, scope
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None

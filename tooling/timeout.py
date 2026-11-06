"""普通 LLM 工具调用的超时作用域与执行任务取消。"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import weakref
from collections.abc import Coroutine
from typing import Any

from src.kernel.concurrency import get_task_manager
from src.kernel.llm.payload.tooling import LLMUsableExecution
from src.kernel.logger import get_logger

from .execution_cleanup import ExecutionCleanup

logger = get_logger("agentic_chatter")

_ACTIVE_SCOPE: contextvars.ContextVar[ToolExecutionScope | None] = (
    contextvars.ContextVar("agentic_chatter_tool_execution_scope", default=None)
)
_LOOP_STATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, _LoopFactoryState
] = weakref.WeakKeyDictionary()
_EXECUTION_TASK_CODES = {
    LLMUsableExecution._advance_iterator.__code__,
    LLMUsableExecution._await_result.__code__,
}


class _FactoryCall:
    """转发现代 factory 参数；仅对明确的旧签名适配 name/context。"""

    def __init__(self, factory: Any) -> None:
        """记录原任务工厂及其支持的关键字参数。

        Args:
            factory: 已安装的任务工厂；为 None 时直接构造 asyncio.Task。
        """
        self.factory = factory
        self.accepted: set[str] | None = None
        if factory is not None:
            try:
                parameters = inspect.signature(factory).parameters.values()
            except (TypeError, ValueError):
                return  # 不可内省的 factory 按现代契约调用；不捕获 TypeError 重试。
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
                self.accepted = {
                    p.name
                    for p in parameters
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }

    def __call__(
        self,
        loop: asyncio.AbstractEventLoop,
        coroutine: Coroutine[Any, Any, Any],
        kwargs: dict[str, Any],
    ) -> asyncio.Task[Any]:
        """调用原任务工厂，保持协程身份及代码对象不变。

        Args:
            loop: 创建任务的事件循环。
            coroutine: 待交给任务工厂的原始协程。
            kwargs: 事件循环传入的任务构造参数。

        Returns:
            asyncio.Task[Any]: 原任务工厂创建的任务。

        Raises:
            TypeError: 旧工厂无法接收非 name/context 参数，或原工厂调用失败。
                工厂异常直接传播，不通过重复调用尝试兼容。
        """
        if self.factory is None:
            return asyncio.Task(coroutine, loop=loop, **kwargs)
        forwarded = dict(kwargs)
        legacy_context = None
        legacy_name = None
        if self.accepted is not None:
            if "context" not in self.accepted:
                legacy_context = forwarded.pop("context", None)
            if "name" not in self.accepted:
                legacy_name = forwarded.pop("name", None)
            unsupported = forwarded.keys() - self.accepted
            if unsupported:
                raise TypeError(
                    f"旧 task factory 不支持参数: {', '.join(sorted(unsupported))}"
                )
        if legacy_context is not None:
            task = legacy_context.run(self.factory, loop, coroutine, **forwarded)
        else:
            task = self.factory(loop, coroutine, **forwarded)
        if legacy_name is not None:
            task.set_name(legacy_name)
        return task


class _LoopFactoryState:
    """一个事件循环上的链式 factory；不包装业务或推进协程。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, previous_factory: Any) -> None:
        """安装链式任务工厂并保留恢复入口。

        Args:
            loop: 当前执行作用域使用的事件循环。
            previous_factory: 安装追踪工厂前的原任务工厂。
        """
        self.users = 0
        self.previous_factory = previous_factory
        make_task = _FactoryCall(previous_factory)

        def tracking_factory(
            current_loop: asyncio.AbstractEventLoop,
            coroutine: Coroutine[Any, Any, Any],
            **kwargs: Any,
        ) -> asyncio.Task[Any]:
            """先记录构造中归属，确保 eager 嵌套创建时父任务已可识别。"""
            context = kwargs.get("context")
            scope = (
                context if context is not None else contextvars.copy_context()
            ).get(_ACTIVE_SCOPE)
            owned = (
                isinstance(scope, ToolExecutionScope)
                and not scope._closed
                and getattr(coroutine, "cr_code", None) in _EXECUTION_TASK_CODES
                and scope.owns_creator(asyncio.current_task(loop=current_loop))
            )
            if not owned:
                return make_task(current_loop, coroutine, kwargs)
            resource = scope.register_execution(coroutine)
            if scope.cancelling:
                coroutine.close()
                resource.abort()
                raise asyncio.CancelledError()
            scope._constructing.add(coroutine)
            try:
                task = make_task(current_loop, coroutine, kwargs)
            except BaseException:
                # factory 未接收成功的推进协程及其业务对象均由本作用域收尾。
                coroutine.close()
                resource.abort()
                raise
            finally:
                scope._constructing.discard(coroutine)
            resource.bind(task)
            scope.track(task)
            return task

        self.factory = tracking_factory
        loop.set_task_factory(tracking_factory)


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """读取已结束且无人等待任务的异常。"""
    if not task.cancelled():
        task.exception()


class ToolExecutionScope:
    """追踪一次工具提交的框架推进任务及业务清理。

    使用上下文变量与临时任务工厂识别直接推进和嵌套推进任务。
    普通派生后台任务不继承执行归属；取消与清理共用调用方提供的宽限预算。
    """

    def __init__(self) -> None:
        """初始化任务集合、清理预算与事件循环恢复状态。"""
        self._tasks: set[asyncio.Task[Any]] = set()
        self._constructing: set[Coroutine[Any, Any, Any]] = set()
        self._resources: dict[LLMUsableExecution, ExecutionCleanup] = {}
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._cleanup_deadline: float | None = None
        self._cleanup_cancel_requested: weakref.WeakSet[asyncio.Task[Any]] = (
            weakref.WeakSet()
        )
        self._context_token: contextvars.Token[ToolExecutionScope | None] | None = None
        self._closed = False
        self._cancelling = False
        self._state: _LoopFactoryState | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owner: asyncio.Task[Any] | None = None

    @property
    def cancelling(self) -> bool:
        """是否已开始取消，避免重复宽限等待。"""
        return self._cancelling

    def owns_creator(self, task: asyncio.Task[Any] | None) -> bool:
        """识别已登记任务或正在原 factory 内 eager 构造的同一协程。"""
        return task is not None and (
            task is self._owner
            or task in self._tasks
            or task.get_coro() in self._constructing
        )

    def register_execution(
        self, coroutine: Coroutine[Any, Any, Any]
    ) -> ExecutionCleanup:
        """在推进 Task 构造前取得业务对象的清理责任。"""
        candidate = ExecutionCleanup.from_coroutine(coroutine, self._schedule_cleanup)
        return self._resources.setdefault(candidate.execution, candidate)

    def _schedule_cleanup(
        self, coroutine: Coroutine[Any, Any, None]
    ) -> asyncio.Task[Any]:
        """清理仍用 TaskManager，显式清除执行归属以避免误当工具推进。"""
        token = _ACTIVE_SCOPE.set(None)
        try:
            task = (
                get_task_manager()
                .create_task(coroutine, name="agentic_tool_cleanup", daemon=True)
                .task
            )
        except BaseException:
            coroutine.close()
            raise
        finally:
            _ACTIVE_SCOPE.reset(token)
        self._cleanup_tasks.add(task)
        task.add_done_callback(_consume_task_result)
        task.add_done_callback(self._cleanup_tasks.discard)
        if self._cleanup_deadline is not None:
            timer = asyncio.get_running_loop().call_later(
                max(0.0, self._cleanup_deadline - asyncio.get_running_loop().time()),
                self._cancel_cleanup_once,
                task,
            )
            task.add_done_callback(lambda _task: timer.cancel())
        return task

    async def __aenter__(self) -> ToolExecutionScope:
        """安装或复用循环 factory。"""
        loop = asyncio.get_running_loop()
        self._owner = asyncio.current_task()
        state = _LOOP_STATES.get(loop)
        current_factory = loop.get_task_factory()
        if state is None or current_factory is not state.factory:
            state = _LoopFactoryState(loop, current_factory)
            _LOOP_STATES[loop] = state
        state.users += 1
        self._state = state
        self._loop = loop
        self._context_token = _ACTIVE_SCOPE.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """不吞异常；调用方管理取消预算。"""
        self.close()
        return False

    def track(self, task: asyncio.Task[Any]) -> None:
        """幂等追踪，已完成失败任务的异常也会被读取。"""
        if task in self._tasks:
            return
        if task.done():
            _consume_task_result(task)
            return
        self._tasks.add(task)
        task.add_done_callback(_consume_task_result)
        task.add_done_callback(self._tasks.discard)
        if self._closed or self._cancelling:
            task.cancel()

    def close(self) -> None:
        """恢复原 factory，不覆盖后来安装的第三方 factory。"""
        if self._closed:
            return
        self._closed = True
        if self._context_token is not None:
            _ACTIVE_SCOPE.reset(self._context_token)
            self._context_token = None
        self._owner = None
        self._tasks.clear()
        self._resources.clear()
        if self._state is not None and self._loop is not None:
            self._state.users -= 1
            if self._state.users == 0:
                if self._loop.get_task_factory() is self._state.factory:
                    self._loop.set_task_factory(self._state.previous_factory)
                if _LOOP_STATES.get(self._loop) is self._state:
                    _LOOP_STATES.pop(self._loop, None)

    def _cancel_cleanup_once(self, task: asyncio.Task[Any]) -> None:
        """截止定时器与异常退出共用幂等取消，避免中断清理的异常收尾。"""
        if task.done() or task in self._cleanup_cancel_requested:
            return
        self._cleanup_cancel_requested.add(task)
        task.cancel()

    async def wait_cleanup(self, grace_seconds: float) -> int:
        """在宽限预算内等待已调度的清理任务。

        Args:
            grace_seconds: 最长等待秒数；零表示不等待，但仍给予清理首次执行机会。

        Returns:
            int: 等待结束时尚未完成的清理任务数量。
        """
        pending = {task for task in self._cleanup_tasks if not task.done()}
        if pending and grace_seconds > 0:
            _, pending = await asyncio.wait(pending, timeout=grace_seconds)
        for task in pending:
            # 零宽限也允许 aclose 获得首次执行机会；不等待它完成。
            asyncio.get_running_loop().call_later(0, self._cancel_cleanup_once, task)
        return len(pending)

    async def cancel_pending(self, grace_seconds: float) -> int:
        """请求取消受管任务，并在统一截止时间内等待业务清理。

        普通资源清理异常不阻断其他任务的取消。外部取消或其他 BaseException
        仍向上传播，但不能跳过受管任务取消和作用域恢复。

        Args:
            grace_seconds: 取消与清理共用的最长等待秒数，允许为零。

        Returns:
            int: 收尾时尚未结束的受管推进任务与清理任务总数。
                该计数不代表外部副作用或任意派生后台任务的完成状态。
        """
        self._cancelling = True
        deadline = asyncio.get_running_loop().time() + grace_seconds
        self._cleanup_deadline = deadline
        try:
            try:
                for resource in list(self._resources.values()):
                    try:
                        resource.abort()
                    except Exception as exc:
                        logger.warning(
                            f"工具执行清理失败 事件=工具清理失败 类型={type(exc).__name__}"
                        )
            finally:
                # 包括清理抛出 BaseException 的路径；不能遗漏任何已登记推进任务。
                for task in list(self._tasks):
                    if not task.done():
                        task.cancel()
            # done 回调可能启动生成器清理任务，因此每次完成后重新取集合。
            while True:
                pending = {t for t in self._tasks | self._cleanup_tasks if not t.done()}
                remaining = deadline - asyncio.get_running_loop().time()
                if not pending or remaining <= 0:
                    break
                await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
        finally:
            # 第二次外部取消也不能丢掉已开始的清理任务。
            for task in list(self._cleanup_tasks):
                if not task.done():
                    asyncio.get_running_loop().call_later(
                        0, self._cancel_cleanup_once, task
                    )
            pending = {t for t in self._tasks | self._cleanup_tasks if not t.done()}
            self.close()
        return len(pending)

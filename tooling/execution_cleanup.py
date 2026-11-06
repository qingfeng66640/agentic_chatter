"""框架推进任务取消后的业务对象清理；不处理外部共享 Task/Future。"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Callable, Coroutine
from typing import Any

from src.kernel.llm.payload.tooling import LLMUsableExecution
from src.kernel.logger import get_logger

logger = get_logger("agentic_chatter")

# 同一 execution 只在同一事件循环推进；同步认领之间不让出控制。
# 标记的是清理尝试而非成功，失败也不重试；弱引用不延长 execution 的生命周期。
_CLEANUP_CLAIMS: weakref.WeakSet[LLMUsableExecution] = weakref.WeakSet()


class ExecutionCleanup:
    """管理框架执行对象的取消清理责任。

    推进任务停止后，关闭未启动的业务协程，或调度已暂停迭代器的异步清理。
    同一执行对象只允许发起一次清理尝试；外部共享 Task 和 Future 不由此处关闭。

    Attributes:
        execution: 核心执行包装对象，限定在同一事件循环中使用。
        business: 尚未转交或清理的业务协程、迭代器或等待对象。
        task: 最近一次成功绑定的推进任务。
        aborted: 是否已请求取消清理。
        disposed: 本记录是否已释放清理责任。
        schedule: 清理调度函数；创建失败时负责关闭未接收的协程。
    """

    def __init__(
        self,
        execution: LLMUsableExecution,
        business: Any,
        schedule: Callable[[Coroutine[Any, Any, None]], asyncio.Task[Any]],
    ) -> None:
        """初始化清理记录并保存推进任务快照。

        Args:
            execution: 本次清理对应的核心执行包装对象。
            business: 推进协程持有的业务对象。
            schedule: 使用统一宽限预算创建清理任务的函数。
        """
        self.execution = execution
        self.business = business
        self.task: asyncio.Task[Any] | None = None
        # resume 先改状态再创建 Task；创建失败时 execution.task 仍为上一轮任务。
        self._registered_task = execution.task
        self.aborted = False
        self.disposed = False
        self.schedule = schedule

    @classmethod
    def from_coroutine(
        cls,
        coroutine: Coroutine[Any, Any, Any],
        schedule: Callable[[Coroutine[Any, Any, None]], asyncio.Task[Any]],
    ) -> ExecutionCleanup:
        """从已识别的核心推进协程创建清理记录。

        Args:
            coroutine: 调用方已通过代码对象识别的核心推进协程。
            schedule: 创建业务清理任务的函数。

        Returns:
            ExecutionCleanup: 保留执行实例与业务对象的清理记录。

        Raises:
            RuntimeError: 协程帧中的执行实例不符合核心契约时抛出。
        """
        frame = getattr(coroutine, "cr_frame", None)
        values = frame.f_locals if frame is not None else {}
        execution = values.get("self")
        if not isinstance(execution, LLMUsableExecution):
            raise RuntimeError("LLMUsableExecution 推进协程契约已改变")
        business = values.get("execution", execution._aiter)
        return cls(execution, business, schedule)

    def bind(self, task: asyncio.Task[Any]) -> None:
        """记录最新推进任务；完成回调兼顾首次运行前取消。"""
        self.task = task
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        """运行中的任务结束后才允许关闭其持有的执行对象。"""
        if task.cancelled():
            self.aborted = True
        if self.aborted:
            self.cleanup_if_idle()

    def abort(self) -> None:
        """请求清理；不抢占仍在执行的协程或生成器。"""
        self.aborted = True
        self.cleanup_if_idle()

    def cleanup_if_idle(self) -> None:
        """在仍持有清理责任且推进已停止时释放业务对象。

        先同步认领清理并终止执行，再关闭未启动协程或调度迭代器的
        ``aclose()``。清理创建失败仅记录日志，不重试，也不保证资源已释放。
        """
        if self.disposed:
            return
        current = self.execution.task
        owner = self.task if self.task is not None else self._registered_task
        if current is not owner:
            # 只有真实换了 Task 才算转移；创建失败留下的注册快照仍归本记录清理。
            self.disposed = True
            self.business = None
            return
        if owner is not None and not owner.done():
            return
        if self.execution in _CLEANUP_CLAIMS:
            # 两个 scope 可能仍指向同一个旧 Task；只有首个记录负责释放。
            self.disposed = True
            self.business = None
            return
        _CLEANUP_CLAIMS.add(self.execution)
        self.disposed = True
        business = self.business
        self.business = None
        # 先关闭 resume 的 READY 门禁，再执行任何业务清理代码。
        # DONE 只表示不可继续推进，不承诺 aclose 成功或已结束。
        self._mark_done()
        # 不取消或 close 传入的共享 Task/Future，也不强行 close 挂起协程。
        if inspect.iscoroutine(business):
            if inspect.getcoroutinestate(business) == inspect.CORO_CREATED:
                business.close()
        elif hasattr(business, "__anext__") and callable(
            getattr(business, "aclose", None)
        ):
            try:
                self.schedule(self._close_iterator(business))
            except Exception as exc:
                # schedule 负责关闭未接收的协程；本边界同时覆盖 done 回调路径。
                logger.warning(
                    f"工具迭代器清理创建失败 事件=工具清理创建失败 类型={type(exc).__name__}"
                )

    def _mark_done(self) -> None:
        """补齐从未进入核心 try/finally 的取消状态，避免调度器空转。"""
        if self.execution._status != "_DONE":
            self.execution.exception = asyncio.CancelledError()
            self.execution._status = "_DONE"

    async def _close_iterator(self, iterator: Any) -> None:
        """在调用方统一宽限预算内运行 aclose；不能强杀拒绝取消的清理。"""
        try:
            await iterator.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"工具迭代器清理失败 事件=工具清理失败 类型={type(exc).__name__}"
            )

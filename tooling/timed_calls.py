"""插件级工具提交预算、结果隔离和超时回灌。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

from src.kernel.concurrency import get_task_manager
from src.kernel.llm import LLMPayload, ROLE, ToolResult
from src.kernel.logger import get_logger

from .timeout import ToolExecutionScope

logger = get_logger("agentic_chatter")
DEFAULT_TOOL_TIMEOUT = 30.0
DEFAULT_CANCEL_GRACE = 1.0
DISPATCH_SIGNATURE = "agentic_chatter:action:dispatch_task"
DISPATCH_GRACE = 30.0


def _positive_budget(value: Any, *, allow_zero: bool = False) -> float:
    """拒绝负数、非数与无限预算，避免配置意外关闭保护。"""
    budget = float(value)
    if not math.isfinite(budget) or budget < 0 or (budget == 0 and not allow_zero):
        raise ValueError("工具时间预算必须为有限正数（全局预算可为 0）")
    return budget


def submission_timeout(config: Any, calls: list[Any], registry: Any) -> float | None:
    """计算一次工具提交的总超时预算。

    整批预算为各调用预算之和，不改变 planning 或 batch 的提交边界。
    ``dispatch_task`` 未显式覆盖时，预算兼容后台任务的执行时限。

    Args:
        config: 提供 tools 和 tasks 配置节的插件配置对象。
        calls: 本次提交的工具调用，顺序与调度器输入一致。
        registry: 按调用名称查询组件及其完整签名的注册表。

    Returns:
        float | None: 提交预算秒数；全局预算为零时返回 None，表示关闭保护。

    Raises:
        ValueError: 配置预算为负数、非有限值或覆盖预算不为正数时抛出。
    """
    tools = getattr(config, "tools", None)
    default = _positive_budget(
        getattr(tools, "tool_timeout_seconds", DEFAULT_TOOL_TIMEOUT), allow_zero=True
    )
    if default == 0:
        return None
    overrides = getattr(tools, "tool_timeout_overrides", {})
    total = 0.0
    for call in calls:
        getter = getattr(registry, "get", None)
        component = getter(call.name) if callable(getter) else None
        signature_getter = getattr(component, "get_signature", None)
        signature = signature_getter() if callable(signature_getter) else None
        budget = default
        if signature in overrides:
            budget = _positive_budget(overrides[signature])
        elif signature == DISPATCH_SIGNATURE:
            tasks = getattr(config, "tasks", None)
            budget = max(
                budget,
                _positive_budget(getattr(tasks, "timeout_seconds", 300.0))
                + DISPATCH_GRACE,
            )
        total += budget
    if not math.isfinite(total):
        raise ValueError("工具提交的总预算必须为有限数值")
    return total


class _ResultBuffer:
    """只接收本次提交的 ToolResult，不让迟到执行改写真实响应链。"""

    def __init__(self, response: Any) -> None:
        """复制响应上下文并初始化本次提交的结果缓冲。

        Args:
            response: 实际响应对象；仅复制其已有 payload 列表，不直接写入。
        """
        self.payloads = list(getattr(response, "payloads", ()) or ())
        self.results: list[ToolResult] = []
        self.closed = False

    def add_payload(self, payload: LLMPayload, position: Any = None) -> _ResultBuffer:
        """模拟 dispatch 所需的响应接口；封口后丢弃迟到结果。"""
        if self.closed:
            return self
        if payload.role != ROLE.TOOL_RESULT:
            raise ValueError("工具调度只应回写 TOOL_RESULT")
        parts = (
            payload.content if isinstance(payload.content, list) else [payload.content]
        )
        self.results.extend(part for part in parts if isinstance(part, ToolResult))
        self.payloads.append(payload)
        return self


def _flush_results(
    buffer: _ResultBuffer,
    calls: list[Any],
    response: Any,
    outcomes: list[tuple[bool, bool]] | None,
    *,
    timeout: float,
) -> list[tuple[bool, bool]]:
    """按调用顺序回写结果，并仅为超时缺项补充说明。

    Args:
        buffer: 已封口的提交结果缓冲。
        calls: 原始工具调用列表。
        response: 接收最终 TOOL_RESULT 的真实响应对象。
        outcomes: 调度器返回的完整状态列表；超时未返回时为 None。
        timeout: 用于缺项说明的提交预算秒数。

    Returns:
        list[tuple[bool, bool]]: 每个调用的捕获与成功状态。
            缺少完整调度状态时保守记录为失败，不从结果正文推断成功。
    """
    remaining = list(buffer.results)
    results: list[tuple[bool, bool]] = []
    for index, call in enumerate(calls):
        match_index = next(
            (
                i
                for i, result in enumerate(remaining)
                if result.call_id == call.id and result.name == call.name
            ),
            None,
        )
        result = remaining.pop(match_index) if match_index is not None else None
        if result is None and outcomes is not None:
            # 正常完成时沿用框架“无可回显结果”的行为，不编造结果。
            results.append(outcomes[index] if index < len(outcomes) else (False, False))
            continue
        if result is None:
            result = ToolResult(
                value=(
                    f"工具提交超过 {timeout:g} 秒预算，本调用未取得完成结果，已请求取消。"
                    "可能尚未执行或已产生部分副作用，不能确认回滚；"
                    "不要自动重试有副作用的调用，请改用其它方式或向用户说明。"
                ),
                call_id=call.id,
                name=call.name,
            )
            success = False
        elif outcomes is not None:
            success = bool(outcomes[index][1]) if index < len(outcomes) else False
        else:
            # dispatch 尚未返回状态数组；只保留正文，不猜测已捕获调用是否成功。
            success = False
        response.add_payload(LLMPayload(ROLE.TOOL_RESULT, result))
        results.append((True, success))
    return results


async def run_with_timeout(
    dispatch: Callable[..., Awaitable[list[tuple[bool, bool]]]],
    calls: list[Any],
    response: Any,
    registry: Any,
    trigger: Any,
    *,
    config: Any,
    stream_id: str,
) -> list[tuple[bool, bool]]:
    """限时等待一次原生工具提交，原样保留批次与 Manager 执行路径。

    使用 ``asyncio.wait`` 限制等待时间，不无限等待拒绝取消的协程。
    超时后封口结果缓冲，再请求取消调度器与受管推进任务；不承诺强制停止
    底层线程、清理任务或回滚外部副作用。

    Args:
        dispatch: 原生工具调度函数，仍通过框架 Manager 执行调用。
        calls: 按原提交模式组织的工具调用列表。
        response: 接收最终工具结果的响应对象。
        registry: 工具注册表，原样传给调度函数。
        trigger: 本次调用的触发消息或上下文，原样传给调度函数。
        config: 插件配置，提供提交预算与取消宽限。
        stream_id: 当前对话流标识，用于任务命名和诊断日志。

    Returns:
        list[tuple[bool, bool]]: 与 calls 顺序一致的捕获与成功状态列表。

    Raises:
        ValueError: 超时预算或取消宽限配置非法时抛出。
        asyncio.CancelledError: 外部取消在完成有界收尾后继续传播。
    """
    timeout = submission_timeout(config, calls, registry)
    if timeout is None or not calls:
        return await dispatch(calls, response, registry, trigger)
    grace = _positive_budget(
        getattr(
            getattr(config, "tools", None),
            "tool_cancel_grace_seconds",
            DEFAULT_CANCEL_GRACE,
        ),
        allow_zero=True,
    )
    buffer = _ResultBuffer(response)
    async with ToolExecutionScope() as scope:

        async def tracked_dispatch() -> list[tuple[bool, bool]]:
            """登记 dispatcher 后再进入 Manager 调用链。"""
            current = asyncio.current_task()
            if current is None:
                raise RuntimeError("工具提交必须运行于 Task 中")
            scope.track(current)
            return await dispatch(calls, buffer, registry, trigger)

        dispatcher = tracked_dispatch()
        try:
            task = (
                get_task_manager()
                .create_task(
                    dispatcher,
                    name=f"agentic_tool_submission_{stream_id[:8]}",
                    daemon=True,
                )
                .task
            )
        except BaseException:
            dispatcher.close()
            buffer.closed = True
            await scope.cancel_pending(grace)
            raise
        scope.track(task)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if done:
                outcomes = task.result()
                buffer.closed = True
                await scope.wait_cleanup(grace)
                return _flush_results(
                    buffer, calls, response, outcomes, timeout=timeout
                )
            buffer.closed = True
            pending = await scope.cancel_pending(grace)
            logger.warning(
                f"[{stream_id[:8]}] 工具提交超时 事件=工具超时 "
                f"预算秒={timeout:g} 调用数={len(calls)} 已追踪任务取消未收敛={pending}"
            )
            return _flush_results(buffer, calls, response, None, timeout=timeout)
        except BaseException:
            buffer.closed = True
            if not scope.cancelling:
                await scope.cancel_pending(grace)
            raise

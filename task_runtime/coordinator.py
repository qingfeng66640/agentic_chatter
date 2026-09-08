"""复杂任务后台执行与消息分流。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.kernel.concurrency import get_task_manager

from .control import TaskMessageKind, classify_task_message
from .executor import TaskCommand, TaskExecutor, TaskRequest
from .models import TaskStatus
from .persistence import TaskStateStore
from .runtime import TaskRuntime


@dataclass(slots=True)
class ActiveTask:
    """一个正在后台运行的任务。"""

    executor: TaskExecutor
    task: Any
    task_info_id: str
    chatter: Any
    request: TaskRequest
    store: TaskStateStore | None


_ACTIVE: dict[str, ActiveTask] = {}


def register_task(
    chatter: Any,
    runtime: TaskRuntime,
    request: TaskRequest,
    store: TaskStateStore | None = None,
) -> None:
    """登记可恢复任务，等待用户控制后再启动。"""
    existing = _ACTIVE.get(runtime.state.task_id)
    if existing is not None:
        return
    idle_task = _IdlePlaceholder()
    executor = TaskExecutor(chatter, runtime, store)
    _ACTIVE[runtime.state.task_id] = ActiveTask(
        executor,
        idle_task,
        "",
        chatter,
        request,
        store,
    )


class _IdlePlaceholder:
    """已结束任务占位：表示当前没有活跃后台协程。"""

    def done(self) -> bool:
        """占位任务始终视为已结束。"""
        return True

    def cancelled(self) -> bool:
        """占位任务未被取消。"""
        return False


def start_task(
    chatter: Any,
    runtime: TaskRuntime,
    request: TaskRequest,
    store: TaskStateStore | None = None,
) -> asyncio.Task[Any]:
    """启动受任务管理器追踪的后台执行，已启动任务不重复运行。"""
    existing = _ACTIVE.get(runtime.state.task_id)
    if existing is not None and not existing.task.done():
        return existing.task

    executor = TaskExecutor(chatter, runtime, store)

    async def runner() -> None:
        runtime.state.metadata["validation_steps"] = tuple(request.validation_steps)
        if request.result_schema is not None:
            runtime.state.metadata["result_schema"] = request.result_schema
        try:
            await executor.run(request)
        finally:
            if store is not None:
                store.save(runtime)
            current = _ACTIVE.get(runtime.state.task_id)
            if current is not None and current.executor is executor:
                if runtime.state.status not in (
                    TaskStatus.PAUSED,
                    TaskStatus.WAITING_USER,
                    TaskStatus.FAILED,
                ):
                    _ACTIVE.pop(runtime.state.task_id, None)

    task_info = get_task_manager().create_task(
        runner(),
        name=f"agentic_task_{runtime.state.task_id[:12]}",
        daemon=True,
        timeout=runtime.state.budget.timeout_seconds,
    )
    _ACTIVE[runtime.state.task_id] = ActiveTask(
        executor,
        task_info.task,
        task_info.task_id,
        chatter,
        request,
        store,
    )
    return task_info.task


async def route_message(task_id: str, text: str) -> bool:
    """将任务流中的新消息分流到后台执行器，必要时恢复执行。"""
    active = _ACTIVE.get(task_id)
    if active is None:
        return False
    if active.task.done() and active.task.cancelled() and active.executor.runtime.state.status == TaskStatus.CANCELLED:
        return False
    message = classify_task_message(text, task_id)
    if message.kind == TaskMessageKind.CHAT:
        return False
    if message.kind == TaskMessageKind.CONTROL and message.text == "status":
        result = active.executor.runtime.state.result
        summary = result.summary if result is not None else (
            f"任务状态：{active.executor.runtime.state.status.value}"
        )
        await active.executor._send_final_text(summary)
        return True
    if message.kind == TaskMessageKind.CONTROL and message.text in ("pause", "cancel"):
        if active.executor.runtime.state.status == TaskStatus.RUNNING and not active.task.done():
            if message.text == "pause":
                active.executor.runtime.pause()
            else:
                active.executor.runtime.cancel()
            get_task_manager().cancel_task(active.task_info_id)
        return True
    if active.task.done() and message.kind == TaskMessageKind.INPUT:
        if active.executor.runtime.state.status in (
            TaskStatus.PAUSED,
            TaskStatus.WAITING_USER,
            TaskStatus.FAILED,
        ):
            active.executor.runtime.resume()
            start_task(
                active.chatter,
                active.executor.runtime,
                active.request,
                active.store,
            )
            active = _ACTIVE[task_id]
    elif active.task.done() and message.kind == TaskMessageKind.CONTROL:
        if message.text == "resume" and active.executor.runtime.state.status in (
            TaskStatus.PAUSED,
            TaskStatus.WAITING_USER,
            TaskStatus.FAILED,
        ):
            active.executor.runtime.resume()
            start_task(
                active.chatter,
                active.executor.runtime,
                active.request,
                active.store,
            )
            active = _ACTIVE[task_id]
        elif message.text != "resume":
            return True

    if message.kind == TaskMessageKind.CONTROL and message.text == "resume" and not active.task.done():
        await active.executor.send_command(TaskCommand("resume"))
        return True
    await active.executor.send_command(
        TaskCommand(
            message.text if message.kind == TaskMessageKind.CONTROL else "input",
            "" if message.kind == TaskMessageKind.CONTROL else message.text,
        )
    )
    return True


def get_active_task(task_id: str) -> ActiveTask | None:
    """获取后台任务。"""
    return _ACTIVE.get(task_id)


def forget_task(task_id: str) -> None:
    """移除已结束任务的后台索引。"""
    _ACTIVE.pop(task_id, None)

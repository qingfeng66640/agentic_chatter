"""复杂任务后台执行与消息分流。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

from .control import TaskMessageKind, classify_task_message
from .executor import TaskCommand, TaskExecutor, TaskRequest
from .models import TaskResult, TaskStatus
from .persistence import TaskStateStore
from .runtime import TaskRuntime, get_task_runtime_manager

logger = get_logger("agentic_chatter")

# runner 内 wait_for 相对任务预算的额外宽限秒数；WatchDog 跳过 daemon 任务，
# 超时兜底必须在插件内执行
_RUNNER_TIMEOUT_GRACE_SECONDS = 30.0

# 可恢复执行的任务状态：这些状态的任务可由 resume/start_task 继续推进
RESUMABLE_TASK_STATUSES = (
    TaskStatus.PAUSED,
    TaskStatus.WAITING_USER,
    TaskStatus.FAILED,
)

# 后台任务结束时需要回灌主 Agent 的状态集合：
# SUCCEEDED 仅在非直出任务（deliver_final_text=False）时推送，
# 直发任务用户已收到结果文本，再推送会造成双重回复。
NOTIFY_WORTHY_STATUSES = (
    TaskStatus.SUCCEEDED,
    TaskStatus.WAITING_USER,
)

# 已完成任务事件的暂存注册表：按流分桶，主 Agent 被唤醒后一次性 drain。
# 单槽 resume 事件存在覆盖窗口，数据本体必须持久暂存直到被消费。
_COMPLETED_EVENTS: dict[str, list[dict[str, Any]]] = {}


def push_completed_event(stream_id: str, payload: dict[str, Any]) -> None:
    """暂存一条任务完成事件，同一 task_id 幂等去重。

    Args:
        stream_id: 目标聊天流 ID。
        payload: 事件数据，至少包含 task_id。
    """
    task_id = str(payload.get("task_id", "") or "")
    events = _COMPLETED_EVENTS.setdefault(stream_id, [])
    if task_id and any(event.get("task_id") == task_id for event in events):
        return
    events.append(payload)


def drain_completed_events(stream_id: str) -> list[dict[str, Any]]:
    """取出并清空指定流的全部完成事件（一次性消费）。"""
    return _COMPLETED_EVENTS.pop(stream_id, [])


def has_completed_events(stream_id: str) -> bool:
    """判断指定流是否还有未消费的完成事件。"""
    return bool(_COMPLETED_EVENTS.get(stream_id))


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


def _should_report_completion(
    runtime: TaskRuntime,
    request: TaskRequest,
) -> bool:
    """按状态与请求开关判定是否需要回灌主 Agent。"""
    status = runtime.state.status
    if status in (TaskStatus.PAUSED, TaskStatus.CANCELLED):
        return False
    if status not in NOTIFY_WORTHY_STATUSES:
        return False
    if status == TaskStatus.WAITING_USER:
        return True
    # SUCCEEDED：直发任务用户已看到结果，默认不回灌
    return bool(request.report_events) and not request.deliver_final_text


def _build_completion_payload(
    runtime: TaskRuntime,
    result: TaskResult,
) -> dict[str, Any]:
    """构建回灌事件的数据载荷。"""
    summary = str(result.summary or "")[: max(0, runtime.state.budget.max_result_size)]
    return {
        "task_id": runtime.state.task_id,
        "task_type": runtime.state.task_type.value,
        "status": status.value if (status := runtime.state.status) else "",
        "summary": summary,
        "error": str(result.error or "")[: runtime.state.budget.max_result_size],
    }


async def _notify_completion(
    chatter: Any,
    runtime: TaskRuntime,
    request: TaskRequest,
    result: TaskResult,
) -> None:
    """任务结束后按需推送完成事件并唤醒主 Agent。"""
    if not _should_report_completion(runtime, request):
        return
    stream_id = runtime.state.stream_id
    push_completed_event(stream_id, _build_completion_payload(runtime, result))
    from src.core.managers.chatter_manager import get_chatter_manager

    await get_chatter_manager().resume_chatter(stream_id, source="sub_agent")
    logger.info(
        f"[{stream_id[:8]}] 任务完成已回灌主 Agent event=task_completion_reported "
        f"任务ID={runtime.state.task_id} 状态={runtime.state.status.value}"
    )


async def _run_guarded(
    executor: TaskExecutor,
    runtime: TaskRuntime,
    request: TaskRequest,
    *,
    timeout: float,
) -> TaskResult:
    """带超时兜底执行任务：悬挂时暂停任务使其可恢复。"""
    try:
        return await asyncio.wait_for(executor.run(request), timeout=timeout)
    except asyncio.TimeoutError:
        timeout_text = "任务执行超时"
        if runtime.is_active():
            result = runtime.finish(
                TaskStatus.PAUSED, "任务执行超时已暂停", error=timeout_text
            )
        else:
            result = runtime.state.result or TaskResult(
                runtime.state.status,
                timeout_text,
            )
        try:
            await executor.send_notice("任务执行超时已暂停，可发送「继续任务」恢复")
        except Exception:  # noqa: BLE001 - 通知失败不影响状态落点
            pass
        return result


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

    # 已登记（register_task）的任务复用其 executor，避免双实例分叉
    executor = (
        existing.executor
        if existing is not None
        else TaskExecutor(chatter, runtime, store)
    )

    async def runner() -> None:
        runtime.state.metadata["validation_steps"] = tuple(request.validation_steps)
        if request.result_schema is not None:
            runtime.state.metadata["result_schema"] = request.result_schema
        try:
            result = await _run_guarded(
                executor,
                runtime,
                request,
                timeout=max(
                    1.0,
                    runtime.state.budget.timeout_seconds
                    + _RUNNER_TIMEOUT_GRACE_SECONDS,
                ),
            )
            await _notify_completion(chatter, runtime, request, result)
        finally:
            if store is not None:
                store.save(runtime)
            current = _ACTIVE.get(runtime.state.task_id)
            if current is not None and current.executor is executor:
                if runtime.state.status not in RESUMABLE_TASK_STATUSES:
                    _ACTIVE.pop(runtime.state.task_id, None)
                if runtime.state.status in (TaskStatus.SUCCEEDED, TaskStatus.CANCELLED):
                    get_task_runtime_manager().remove(runtime.state.task_id)

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


def _resume_and_restart(active: ActiveTask) -> None:
    """恢复可恢复任务并重新拉起后台执行。"""
    active.executor.runtime.resume()
    start_task(
        active.chatter,
        active.executor.runtime,
        active.request,
        active.store,
    )


async def _cancel_active(active: ActiveTask, *, notify: bool = True) -> TaskResult:
    """取消任务并清理后台索引，覆盖运行/暂停/等待/占位四种状态。"""
    runtime = active.executor.runtime
    result = runtime.cancel()
    if not active.task.done():
        get_task_manager().cancel_task(active.task_info_id)
    else:
        # _IdlePlaceholder：无 runner finally，必须显式清理与落盘
        _ACTIVE.pop(runtime.state.task_id, None)
        get_task_runtime_manager().remove(runtime.state.task_id)
        if active.store is not None:
            active.store.save(runtime)
    if notify:
        goal = str(runtime.state.user_goal or "").strip()[:30] or "未命名任务"
        await active.executor.send_notice(f"任务「{goal}」已取消")
    return result


async def cancel_task_runtime(runtime: TaskRuntime, *, notify: bool = False) -> str:
    """取消任务运行时的公开入口，返回确认文本（不主动发通知）。"""
    active = _ACTIVE.get(runtime.state.task_id)
    if active is not None:
        await _cancel_active(active, notify=notify)
    else:
        runtime.cancel()
        get_task_runtime_manager().remove(runtime.state.task_id)
    goal = str(runtime.state.user_goal or "").strip()[:30] or "未命名任务"
    return f"任务「{goal}」已取消"


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
        await active.executor.send_notice(summary)
        return True
    if message.kind == TaskMessageKind.CONTROL and message.text in ("pause", "cancel"):
        if message.text == "cancel":
            # cancel 无条件执行：RUNNING/PAUSED/WAITING_USER/占位任务均可取消
            await _cancel_active(active)
        elif (
            active.executor.runtime.state.status == TaskStatus.RUNNING
            and not active.task.done()
        ):
            active.executor.runtime.pause()
            get_task_manager().cancel_task(active.task_info_id)
        return True
    if active.task.done() and message.kind == TaskMessageKind.INPUT:
        if active.executor.runtime.state.status in RESUMABLE_TASK_STATUSES:
            _resume_and_restart(active)
            active = _ACTIVE[task_id]
    elif active.task.done() and message.kind == TaskMessageKind.CONTROL:
        if message.text == "resume" and active.executor.runtime.state.status in (
            RESUMABLE_TASK_STATUSES
        ):
            _resume_and_restart(active)
            active = _ACTIVE[task_id]
        elif message.text != "resume":
            return True

    if message.kind == TaskMessageKind.CONTROL and message.text == "resume" and not active.task.done():
        await active.executor.send_command(TaskCommand("resume"))
        return True
    # 运行至此只可能是 INPUT：status/pause/cancel 已在前置分支拦截，
    # resume 的两种状态（done/not done）也已被上面两段覆盖。
    await active.executor.send_command(TaskCommand("input", message.text))
    return True


def get_active_task(task_id: str) -> ActiveTask | None:
    """获取后台任务。"""
    return _ACTIVE.get(task_id)


def forget_task(task_id: str) -> None:
    """移除已结束任务的后台索引。"""
    _ACTIVE.pop(task_id, None)

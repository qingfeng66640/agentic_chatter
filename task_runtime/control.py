"""任务控制和消息分流协议。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import TaskResult
from .runtime import TaskRuntimeManager


class TaskMessageKind(StrEnum):
    """任务执行期间的新消息类型。"""

    CONTROL = "control"
    INPUT = "input"
    CHAT = "chat"


@dataclass(frozen=True, slots=True)
class TaskMessage:
    """经过分类的任务相关消息。"""

    kind: TaskMessageKind
    text: str
    task_id: str | None = None


_CONTROL_WORDS = {
    "暂停任务": "pause",
    "继续任务": "resume",
    "取消任务": "cancel",
    "查看任务状态": "status",
}


def classify_task_message(text: str, task_id: str | None = None) -> TaskMessage:
    """将用户消息分类为控制、任务输入或普通聊天。"""
    normalized = str(text or "").strip()
    if normalized in _CONTROL_WORDS:
        return TaskMessage(TaskMessageKind.CONTROL, _CONTROL_WORDS[normalized], task_id)
    if task_id and (
        normalized.startswith("补充任务")
        or normalized.startswith("补充资料")
        or normalized.startswith(f"{task_id} ")
    ):
        return TaskMessage(TaskMessageKind.INPUT, normalized, task_id)
    return TaskMessage(TaskMessageKind.CHAT, normalized)


def control_task(
    manager: TaskRuntimeManager,
    task_id: str,
    command: str,
) -> TaskResult | None:
    """执行任务控制命令。"""
    runtime = manager.get(task_id)
    if runtime is None:
        return None
    if command == "status":
        return runtime.state.result or TaskResult(
            runtime.state.status,
            f"任务状态：{runtime.state.status.value}",
            completed_steps=tuple(runtime.state.completed_steps),
            remaining_steps=(runtime.state.current_step,)
            if runtime.state.current_step
            else (),
            error=runtime.state.error,
        )
    if command == "pause":
        return runtime.pause()
    if command == "resume":
        runtime.resume()
        return TaskResult(runtime.state.status, "任务已恢复，等待执行器继续")
    if command == "cancel":
        return runtime.cancel()
    raise ValueError(f"未知任务控制命令：{command}")

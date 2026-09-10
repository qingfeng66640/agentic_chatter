"""任务控制和消息分流协议。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import TaskResult, TaskStatus
from .runtime import TaskRuntime, TaskRuntimeManager


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

# 控制词白名单（整串匹配，不做子串匹配以防误伤普通聊天）。
# 裸"取消""停止""停一下"不入表：聊天中误伤概率高。
_CANCEL_PHRASES = (
    "取消任务",
    "取消这个任务",
    "取消掉任务",
    "停止任务",
    "停掉任务",
    "终止任务",
    "放弃任务",
    "别做了",
    "别弄了",
    "不用做了",
    "不要做了",
    "不用继续了",
)
_PAUSE_PHRASES = ("暂停任务", "任务暂停", "先暂停任务")
_RESUME_PHRASES = ("继续任务", "恢复任务", "接着做任务", "继续做任务")
_STATUS_PHRASES = (
    "任务状态",
    "任务进度",
    "查看任务状态",
    "查看任务进度",
    "任务怎么样了",
    "任务状态如何",
    "现在任务状态",
    "任务列表",
    "查看任务",
    "有什么任务",
)

# 仍占用并发名额的状态；pause 只对这些状态之外的活动任务有意义
_RUNNING_STATUSES = (TaskStatus.PLANNING, TaskStatus.RUNNING)
_RESUMABLE_STATUSES = (TaskStatus.PAUSED, TaskStatus.WAITING_USER, TaskStatus.FAILED)

# 任务状态中文标签
STATUS_LABELS: dict[TaskStatus, str] = {
    TaskStatus.CREATED: "已创建",
    TaskStatus.PLANNING: "规划中",
    TaskStatus.RUNNING: "运行中",
    TaskStatus.WAITING_USER: "等待用户",
    TaskStatus.PAUSED: "已暂停",
    TaskStatus.SUCCEEDED: "成功",
    TaskStatus.FAILED: "失败",
    TaskStatus.CANCELLED: "已取消",
}

_TRIM_PUNCTUATION = "。！!？?～~，, 	"


def _normalize(text: str) -> str:
    """归一化控制消息：去首尾空白与标点。"""
    return str(text or "").strip().strip(_TRIM_PUNCTUATION).strip()


def _match_phrase(normalized: str) -> str | None:
    """按白名单短语表匹配控制命令，命中返回命令名。"""
    if normalized in _PAUSE_PHRASES:
        return "pause"
    if normalized in _RESUME_PHRASES:
        return "resume"
    if normalized in _CANCEL_PHRASES:
        return "cancel"
    if normalized in _STATUS_PHRASES:
        return "status"
    return None


def classify_task_message(text: str, task_id: str | None = None) -> TaskMessage:
    """将用户消息分类为控制、任务输入或普通聊天。"""
    normalized = str(text or "").strip()
    if normalized in _CONTROL_WORDS:
        return TaskMessage(TaskMessageKind.CONTROL, _CONTROL_WORDS[normalized], task_id)
    command = _match_phrase(_normalize(normalized))
    if command is not None:
        return TaskMessage(TaskMessageKind.CONTROL, command, task_id)
    # 控制动词 + 序号引用（如“取消任务2”“暂停任务1”）也视为控制消息；
    # 序号由 match_task_ref 解析
    import re

    if re.fullmatch(r"(?:取消|停止|终止|放弃|暂停|继续|恢复)任务\d+号?", _normalize(normalized)):
        verb = _normalize(normalized)[0:2]
        command = {
            "取消": "cancel",
            "停止": "cancel",
            "终止": "cancel",
            "放弃": "cancel",
            "暂停": "pause",
            "继续": "resume",
            "恢复": "resume",
        }.get(verb)
        if command is not None:
            return TaskMessage(TaskMessageKind.CONTROL, command, task_id)
    if normalized.startswith("补充任务") or normalized.startswith("补充资料"):
        return TaskMessage(TaskMessageKind.INPUT, normalized, task_id)
    if task_id and normalized.startswith(f"{task_id} "):
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


def describe_tasks(tasks: list[TaskRuntime]) -> str:
    """渲染多任务列表：序号、状态、目标摘要与进度。"""
    if not tasks:
        return "当前没有进行中的任务。"
    lines = [f"当前共 {len(tasks)} 个任务："]
    for index, runtime in enumerate(tasks, start=1):
        state = runtime.state
        goal = str(state.user_goal or "").strip()[:30] or "（无目标）"
        lines.append(
            f"{index}. [{STATUS_LABELS.get(state.status, state.status.value)}] {goal}"
            f"（迭代 {state.iterations}/{state.budget.max_iterations}，"
            f"工具 {state.tool_calls}/{state.budget.max_tool_calls}）"
            f"任务ID={state.task_id[:8]}"
        )
    lines.append("取消/暂停可带序号（如“取消任务2”）或任务ID前缀；序号为当前排序。")
    return "\n".join(lines)


def match_task_ref(text: str, tasks: list[TaskRuntime]) -> TaskRuntime | None:
    """从消息文本解析任务引用：序号（#N/N号/任务N）或任务ID前缀（≥6位）。"""
    normalized = _normalize(text)
    if not normalized or not tasks:
        return None
    import re

    token = normalized.split()[0]
    raw: str | None = None
    # 序号引用：整串或首 token 形如 #N / N号 / 任务N
    ordinal = re.fullmatch(r"(?:#(\d+))|(?:(\d+)号)|(?:任务(\d+))", token)
    if ordinal is not None:
        raw = next((group for group in ordinal.groups() if group), None)
    elif re.fullmatch(
        r"(?:取消|停止|终止|放弃|暂停|继续|恢复)任务\d+号?", normalized
    ):
        # 控制动词 + 序号（如“取消任务2”）：检索内嵌的 任务N 引用
        embedded = re.search(r"任务(\d+)", normalized)
        raw = embedded.group(1) if embedded else None
    if raw is not None:
        position = int(raw)
        if 1 <= position <= len(tasks):
            return tasks[position - 1]
        return None
    if len(token) >= 6:
        candidates = [
            runtime
            for runtime in tasks
            if runtime.state.task_id.startswith(token)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def resolve_control_target(
    command: str,
    text: str,
    tasks: list[TaskRuntime],
) -> tuple[TaskRuntime | None, str]:
    """解析控制命令的目标任务；无法唯一定位时返回消歧提示。

    Returns:
        tuple[TaskRuntime | None, str]: (目标任务, 消歧提示文本；空串表示无需提示)。
    """
    if not tasks:
        return None, ""
    if command == "status":
        return None, ""
    ref_task = match_task_ref(text, tasks)
    if ref_task is not None:
        return ref_task, ""
    if command == "pause":
        candidates = [
            runtime for runtime in tasks if runtime.state.status in _RUNNING_STATUSES
        ]
    elif command == "resume":
        candidates = [
            runtime for runtime in tasks if runtime.state.status in _RESUMABLE_STATUSES
        ]
    else:
        candidates = list(tasks)
    if len(candidates) == 1:
        return candidates[0], ""
    return None, describe_tasks(tasks)

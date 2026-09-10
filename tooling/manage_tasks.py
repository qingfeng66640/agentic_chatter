"""后台任务管理工具。

主 Agent 的 LLM 通过 ``tool-manage_tasks`` 查询与取消当前聊天流的
后台任务：列出全部任务及状态/进度、查询单个任务详情、取消指定任务。
该工具不进入任务模板白名单，因此任务内的 sub-agent 看不到它。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from src.core.components.base.tool import BaseTool

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from ..task_runtime import TaskRuntime

# 任务ID前缀引用的最小长度，防止短前缀误匹配
_MIN_REF_PREFIX_LENGTH = 6


def _format_task_detail(runtime: "TaskRuntime") -> str:
    """渲染单个任务的详情快照。"""
    from ..task_runtime.control import STATUS_LABELS

    state = runtime.state
    lines = [
        f"任务 {state.task_id[:8]}："
        f"[{STATUS_LABELS.get(state.status, state.status.value)}] "
        f"{str(state.user_goal or '').strip()[:30] or '（无目标）'}",
        f"类型：{state.task_type.value}；"
        f"迭代 {state.iterations}/{state.budget.max_iterations}；"
        f"工具调用 {state.tool_calls}/{state.budget.max_tool_calls}；"
        f"已完成步骤 {len(state.completed_steps)} 项",
    ]
    if state.current_step:
        lines.append(f"当前步骤：{state.current_step[:60]}")
    if state.error:
        lines.append(f"最近错误：{state.error[:100]}")
    result = state.result
    if result is not None and result.summary:
        lines.append(f"结果摘要：{result.summary[:200]}")
    return "\n".join(lines)


def _resolve_task_ref(
    task_ref: str,
    tasks: list["TaskRuntime"],
) -> tuple["TaskRuntime | None", str]:
    """按序号或任务ID前缀定位任务。

    Returns:
        tuple[TaskRuntime | None, str]: (命中的任务, 错误提示；命中时为空串)。
    """
    normalized = str(task_ref or "").strip()
    if not normalized:
        return None, "请提供任务序号（list 输出中的序号）或任务ID前缀"
    if normalized.isdigit():
        position = int(normalized)
        if 1 <= position <= len(tasks):
            return tasks[position - 1], ""
        return None, f"序号 {position} 超出范围；当前共 {len(tasks)} 个任务"
    if len(normalized) >= _MIN_REF_PREFIX_LENGTH:
        candidates = [
            runtime
            for runtime in tasks
            if runtime.state.task_id.startswith(normalized)
        ]
        if len(candidates) == 1:
            return candidates[0], ""
        if len(candidates) > 1:
            ids = "、".join(runtime.state.task_id[:8] for runtime in candidates)
            return None, f"ID 前缀匹配到多个任务：{ids}，请使用更长的前缀"
        return None, "没有任务匹配该 ID 前缀"
    return None, f"ID 前缀至少 {_MIN_REF_PREFIX_LENGTH} 位，或直接使用序号"


class ManageTasksTool(BaseTool):
    """列出、查询和取消当前聊天流的后台任务。"""

    tool_name = "manage_tasks"
    tool_description = (
        "管理当前对话的后台任务。command=list 列出全部任务及状态/进度/目标；"
        "command=status 查询单个任务详情；command=cancel 取消指定任务。"
        "task_ref 填列表中的序号或任务ID前缀。"
        "用户问“任务怎么样了/任务列表”或要求停止某个任务时调用。"
    )

    async def execute(
        self,
        command: Annotated[str, "操作：list（默认）/ status / cancel"] = "list",
        task_ref: Annotated[str, "任务序号（list 输出中的序号）或任务ID前缀（至少6位）"] = "",
    ) -> tuple[bool, str]:
        """执行任务管理命令。

        Args:
            command: list/status/cancel 之一，非法值回退 list。
            task_ref: status 与 cancel 必填的任务引用。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        # 延迟导入：task_runtime 包加载链较重，避免模块导入期的循环依赖
        from ..task_runtime import get_task_runtime_manager
        from ..task_runtime.control import describe_tasks

        stream_id = self.get_current_stream_id()
        if not stream_id:
            return False, "无法确定当前聊天流"

        manager = get_task_runtime_manager()
        tasks = manager.get_active_tasks(stream_id)
        action = str(command or "").strip().lower()
        if action not in ("list", "status", "cancel"):
            action = "list"

        if action == "list":
            if not tasks:
                return True, "当前没有进行中的任务。"
            return True, describe_tasks(list(tasks))

        target, error = _resolve_task_ref(task_ref, tasks)
        if target is None:
            if not tasks:
                return False, "当前没有进行中的任务。"
            return False, f"{error}；可先调用 list 查看任务序号"

        if action == "status":
            return True, _format_task_detail(target)
        # cancel
        from ..task_runtime.coordinator import cancel_task_runtime

        message = await cancel_task_runtime(target, notify=False)
        return True, message

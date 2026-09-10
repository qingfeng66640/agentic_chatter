"""复杂任务派发动作。

主 chatter 的 LLM 通过 ``action-dispatch_task`` 显式呼出 sub-agent
（TaskExecutor）同步执行复杂任务。sub-agent 使用独立任务循环与任务
预算（config.tasks.*），不受主 chatter 行动循环的最大迭代数影响；
任务结果或错误以文本返回给调用方，由主 chatter 向用户转述。

已知权衡：本动作同步等待任务完成，等待期间行动循环无法响应新消息，
最长阻塞 ``timeout_seconds + 30`` 秒，由任务预算兜底。
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from src.core.components.base.action import BaseAction

from ..chatter import AgenticChatter, _build_task_budget
from ..task_runtime import (
    TaskBudget,
    TaskRequest,
    TaskResult,
    TaskStateStore,
    TaskStatus,
    TaskType,
    get_task_runtime_manager,
    get_task_templates,
)
from ..task_runtime.coordinator import get_active_task, register_task

# 任务类型名到枚举的映射，供 LLM 参数解析使用
_TASK_TYPE_NAMES: dict[str, TaskType] = {
    task_type.value: task_type
    for task_type in (
        TaskType.GENERAL,
        TaskType.RESEARCH,
        TaskType.CODE_ANALYSIS,
        TaskType.CODE_CHANGE,
        TaskType.CONTENT_EDIT,
    )
}

# wait_for 相对任务预算的额外宽限秒数，让内部预算超时先产出可恢复的 PAUSED
_TIMEOUT_GRACE_SECONDS = 30.0


def _resolve_task_type(name: str) -> TaskType:
    """把 LLM 传入的类型名解析为任务类型，非法值回退通用任务。"""
    return _TASK_TYPE_NAMES.get(str(name or "").strip(), TaskType.GENERAL)


def _status_label(status: TaskStatus) -> str:
    """返回任务状态的中文说明。"""
    labels = {
        TaskStatus.SUCCEEDED: "成功",
        TaskStatus.FAILED: "失败",
        TaskStatus.PAUSED: "已暂停（用户可发送“继续任务”恢复）",
        TaskStatus.WAITING_USER: "等待用户补充",
        TaskStatus.CANCELLED: "已取消",
    }
    return labels.get(status, status.value)


def _format_task_result(result: TaskResult) -> str:
    """把结构化任务结果格式化为回灌主 LLM 的文本。"""
    lines = [f"任务状态：{_status_label(result.status)}"]
    if result.summary:
        lines.append(f"结果摘要：{result.summary}")
    if result.error:
        lines.append(f"错误信息：{result.error}")
    if result.remaining_steps:
        lines.append(f"剩余事项：{'；'.join(result.remaining_steps)}")
    return "\n".join(lines)


class DispatchTaskAction(BaseAction):
    """把复杂任务交给独立 sub-agent 同步执行。"""

    action_name = "dispatch_task"
    action_description = (
        "把一个复杂的多步任务交给独立的 sub-agent 同步执行，完成后返回结果摘要。"
        "适合需要多轮工具调用的调研、代码分析、代码修改、内容整理等任务。"
        "调用后当前行动会等待 sub-agent 完成（受任务预算时限约束），"
        "期间无法响应新消息；拿到结果后由你向用户转述。"
        "同一对话任务数量有上限（见配置 max_concurrent_tasks）；"
        "不要派发简单的一次性查询。"
    )
    primary_action = False
    associated_types = ["text"]

    async def execute(
        self,
        objective: Annotated[
            str,
            "任务目标：完整、自包含的指令，包含 sub-agent 需要的全部上下文",
        ],
        task_type: Annotated[
            str,
            "任务类型：general/research/code_analysis/code_change/content_edit，留空为 general",
        ] = "",
    ) -> tuple[bool, str]:
        """呼出 sub-agent 执行任务并等待结果。

        Args:
            objective: 任务目标描述，需要自包含。
            task_type: 任务类型名，非法值回退 general。

        Returns:
            tuple[bool, str]: (派发是否成功, 结果或错误说明)。
        """
        config = getattr(self.plugin, "config", None)
        tasks_config = getattr(config, "tasks", None)
        if tasks_config is not None and not bool(
            getattr(tasks_config, "action_enabled", True)
        ):
            return False, "任务子代理未启用"

        objective = str(objective or "").strip()
        if not objective:
            return False, "任务目标不能为空"

        stream_id = str(getattr(self.chat_stream, "stream_id", "") or "")
        if not stream_id:
            return False, "无法确定当前聊天流"

        manager = get_task_runtime_manager()
        cap = max(
            1,
            int(getattr(tasks_config, "max_concurrent_tasks", 2) or 2),
        ) if tasks_config is not None else 2
        active_tasks = manager.get_active_tasks(stream_id)
        if len(active_tasks) >= cap:
            return False, (
                f"当前对话已有 {len(active_tasks)} 个任务在运行（上限 {cap}），"
                "请等待完成或让用户取消；可调用 tool-manage_tasks 查看"
            )

        task_type_enum = _resolve_task_type(task_type)
        budget = _build_task_budget(config) if config is not None else TaskBudget()
        denied_tools = tuple(getattr(tasks_config, "denied_tools", ()) or ())
        default_allowed = tuple(getattr(tasks_config, "default_allowed_tools", ()) or ())
        try:
            runtime = manager.create(
                stream_id,
                objective,
                task_type=task_type_enum,
                allowed_tools=default_allowed or None,
                budget=budget,
                denied_tools=denied_tools,
            )
        except ValueError as exc:
            return False, f"任务创建失败：{exc}"
        runtime.start()

        try:
            template = get_task_templates().get(task_type_enum)
            runtime.state.metadata["validation_steps"] = tuple(template.validation_steps)
            if template.result_schema is not None:
                runtime.state.metadata["result_schema"] = template.result_schema

            checkpoint_directory = str(
                getattr(tasks_config, "checkpoint_directory", "") or ""
            ).strip()
            store = TaskStateStore(checkpoint_directory) if checkpoint_directory else None
            chatter = AgenticChatter(stream_id, self.plugin)
            request = TaskRequest(
                objective=objective,
                trigger_message=self._get_last_context_message(),
                result_schema=template.result_schema,
                validation_steps=template.validation_steps,
                deliver_final_text=False,
            )
            register_task(chatter, runtime, request, store)
        except Exception as exc:
            runtime.cancel()
            manager.remove(runtime.state.task_id)
            return False, f"任务启动失败: {exc}"

        try:
            active = get_active_task(runtime.state.task_id)
            if active is None:
                return False, "任务登记丢失，无法执行"
            result = await asyncio.wait_for(
                active.executor.run(request),
                timeout=max(1.0, float(budget.timeout_seconds) + _TIMEOUT_GRACE_SECONDS),
            )
        except asyncio.TimeoutError:
            return False, f"任务超时（{budget.timeout_seconds:.0f} 秒），已取消"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 结果必须回灌 LLM 而非中断行动循环
            return False, f"任务执行异常: {exc}"
        return True, _format_task_result(result)

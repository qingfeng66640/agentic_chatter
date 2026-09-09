"""复杂任务模板。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import TaskBudget, TaskType
from .policy import ToolPolicy


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    """一种任务类型的默认策略。"""

    task_type: TaskType
    description: str
    budget: TaskBudget
    policy: ToolPolicy
    validation_steps: tuple[str, ...] = ()
    result_schema: dict[str, object] | None = None


class TaskTemplateRegistry:
    """任务模板注册表。"""

    def __init__(self, templates: tuple[TaskTemplate, ...] = ()) -> None:
        """创建模板注册表。"""
        self._templates = {template.task_type: template for template in templates}

    def register(self, template: TaskTemplate) -> None:
        """注册或替换任务模板。"""
        self._templates[template.task_type] = template

    def get(self, task_type: TaskType) -> TaskTemplate:
        """获取任务模板，不存在时使用通用模板。"""
        template = self._templates.get(task_type)
        if template is not None:
            return template
        fallback = self._templates.get(TaskType.GENERAL)
        if fallback is None:
            raise KeyError(f"未注册任务模板：{task_type}")
        return fallback

    def list(self) -> tuple[TaskTemplate, ...]:
        """列出已注册模板。"""
        return tuple(self._templates.values())


def get_task_templates() -> TaskTemplateRegistry:
    """创建默认任务模板注册表。"""
    readonly = ("read*", "search*", "list*", "query*", "test*", "explore*")
    return TaskTemplateRegistry(
        (
            TaskTemplate(TaskType.GENERAL, "通用复杂任务", TaskBudget(), ToolPolicy(readonly), ("检查任务结果",)),
            TaskTemplate(TaskType.RESEARCH, "资料查询与整理", TaskBudget(max_iterations=16, max_tool_calls=32), ToolPolicy(readonly), ("核对来源",)),
            TaskTemplate(TaskType.CODE_ANALYSIS, "代码分析", TaskBudget(max_iterations=16, max_tool_calls=32), ToolPolicy(("read*", "search*", "list*", "test*")), ("运行相关测试",)),
            TaskTemplate(TaskType.CODE_CHANGE, "代码修改", TaskBudget(max_iterations=20, max_tool_calls=40), ToolPolicy(("read*", "search*", "list*", "edit*", "test*"), require_confirmation_tools=("delete*", "commit*", "push*", "publish*")), ("运行测试",)),
            TaskTemplate(TaskType.CONTENT_EDIT, "内容修改", TaskBudget(max_iterations=16, max_tool_calls=32), ToolPolicy(("read*", "search*", "edit*", "test*", "preview*"), require_confirmation_tools=("publish*",)), ("检查变更范围",)),
        )
    )


def build_task_report_prompt(events: tuple[dict[str, object], ...]) -> str:
    """把任务完成事件渲染为主 Agent 汇报提示词。

    Args:
        events: 完成事件列表，每条含 task_type/status/summary/error。

    Returns:
        str: 拼接后的汇报提示词。
    """
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        status = str(event.get("status", "") or "")
        summary = str(event.get("summary", "") or "").strip()
        error = str(event.get("error", "") or "").strip()
        task_type = str(event.get("task_type", "") or "")
        lines.append(f"[任务{index}] 类型={task_type} 状态={status}")
        if summary:
            lines.append(f"结果摘要：{summary}")
        if error:
            lines.append(f"失败原因：{error}")
    joined = "\n".join(lines)
    return (
        "你的一个后台 sub-agent 任务刚刚结束，以下是执行结果：\n"
        f"{joined}\n"
        "请结合上下文自然地向用户汇报这个结果；"
        "如果任务在等待用户确认或补充输入，请说明需要用户做什么，"
        "并提示用户可以用「补充任务：」前缀继续该任务。"
        "如果结果与当前话题无关或用户并不关心，可以简短带过。"
    )

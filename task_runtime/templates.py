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

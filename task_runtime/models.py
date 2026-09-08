"""复杂任务运行时的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskStatus(StrEnum):
    """任务生命周期状态。"""

    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(StrEnum):
    """初版支持的任务类型。"""

    GENERAL = "general"
    RESEARCH = "research"
    CODE_ANALYSIS = "code_analysis"
    CODE_CHANGE = "code_change"
    CONTENT_EDIT = "content_edit"


@dataclass(frozen=True, slots=True)
class TaskBudget:
    """单个任务的有限资源预算。"""

    max_iterations: int = 12
    max_tool_calls: int = 24
    max_same_signature_calls: int = 2
    max_no_progress_steps: int = 3
    max_failures: int = 3
    max_subtasks: int = 1
    timeout_seconds: float = 300.0
    max_result_size: int = 6000

    def __post_init__(self) -> None:
        """校验预算不可为负。"""
        if any(
            value <= 0
            for value in (
                self.max_iterations,
                self.max_tool_calls,
                self.max_same_signature_calls,
                self.max_no_progress_steps,
                self.max_failures,
                self.max_subtasks,
                self.timeout_seconds,
                self.max_result_size,
            )
        ):
            raise ValueError("任务预算各项必须为正数")


@dataclass(frozen=True, slots=True)
class TaskCheckpoint:
    """任务步骤边界的检查点。"""

    task_id: str
    step_id: str
    status: str
    summary: str = ""
    artifacts: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class TaskResult:
    """对主 Chatter 暴露的有界任务结果。"""

    status: TaskStatus
    summary: str
    completed_steps: tuple[str, ...] = ()
    remaining_steps: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    needs_user_input: bool = False
    error: str = ""

    def bounded(self, max_size: int) -> "TaskResult":
        """将摘要和错误限制在任务预算内。"""
        limit = max(0, max_size)
        return TaskResult(
            status=self.status,
            summary=self.summary[:limit],
            completed_steps=self.completed_steps,
            remaining_steps=self.remaining_steps,
            artifacts=self.artifacts,
            validation=self.validation,
            needs_user_input=self.needs_user_input,
            error=self.error[:limit],
        )


@dataclass(slots=True)
class TaskState:
    """任务运行时的可变状态。"""

    stream_id: str
    user_goal: str
    task_type: TaskType = TaskType.GENERAL
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    require_confirmation_tools: tuple[str, ...] = ()
    budget: TaskBudget = field(default_factory=TaskBudget)
    task_id: str = field(default_factory=lambda: uuid4().hex)
    parent_turn_id: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    checkpoints: list[TaskCheckpoint] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    failures: int = 0
    no_progress_steps: int = 0
    signature_counts: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_result_summary: str = ""
    error: str = ""
    result: TaskResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None

    def __post_init__(self) -> None:
        """初始化任务的执行起点。"""
        if self.started_at is None and self.status in (TaskStatus.PLANNING, TaskStatus.RUNNING):
            self.started_at = self.created_at

    def elapsed_seconds(self) -> float:
        """返回任务自启动以来的墙上时钟秒数。"""
        start = self.started_at or self.created_at
        return max(0.0, (datetime.now(timezone.utc) - start).total_seconds())

    def to_result(self) -> TaskResult | None:
        """返回当前结构化结果。"""
        return self.result

    def touch(self) -> None:
        """更新任务的修改时间。"""
        self.updated_at = datetime.now(timezone.utc)

    def transition(self, target: TaskStatus) -> None:
        """执行受限状态迁移。"""
        allowed = {
            TaskStatus.CREATED: {TaskStatus.PLANNING, TaskStatus.CANCELLED},
            TaskStatus.PLANNING: {TaskStatus.RUNNING, TaskStatus.WAITING_USER, TaskStatus.FAILED, TaskStatus.CANCELLED},
            TaskStatus.RUNNING: {TaskStatus.WAITING_USER, TaskStatus.PAUSED, TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED},
            TaskStatus.WAITING_USER: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
            TaskStatus.PAUSED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
            TaskStatus.SUCCEEDED: set(),
            TaskStatus.FAILED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
            TaskStatus.CANCELLED: set(),
        }
        if target not in allowed[self.status]:
            raise ValueError(f"不允许的任务状态迁移：{self.status} -> {target}")
        self.status = target
        if target == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
        self.touch()

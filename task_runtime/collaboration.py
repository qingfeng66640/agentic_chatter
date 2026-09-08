"""受控子任务编排与结果合并。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from .executor import TaskExecutor, TaskRequest
from .models import TaskBudget, TaskResult, TaskState, TaskStatus
from .runtime import TaskRuntime


@dataclass(frozen=True, slots=True)
class SubtaskSpec:
    """一个有限子任务。"""

    name: str
    objective: str
    depends_on: tuple[str, ...] = ()


class TaskExecutorFactory:
    """在父任务预算内创建真实子任务执行器。"""

    def __init__(self, chatter: object, parent: TaskRuntime) -> None:
        """创建工厂。"""
        self.chatter = chatter
        self.parent = parent

    def _create_child(self, objective: str) -> TaskRuntime | None:
        """按父任务剩余预算创建子任务。"""
        parent = self.parent.state
        used = int(parent.metadata.get("subtasks_used", 0))
        if used >= parent.budget.max_subtasks:
            return None
        budget = parent.budget
        reserved_iterations = int(parent.metadata.get("subtasks_iterations_reserved", 0))
        reserved_tool_calls = int(parent.metadata.get("subtasks_tool_calls_reserved", 0))
        reserved_failures = int(parent.metadata.get("subtasks_failures_reserved", 0))
        remaining_iterations = max(
            0, budget.max_iterations - parent.iterations - reserved_iterations
        )
        remaining_tool_calls = max(
            0, budget.max_tool_calls - parent.tool_calls - reserved_tool_calls
        )
        remaining_failures = max(
            0, budget.max_failures - parent.failures - reserved_failures
        )
        if not remaining_iterations or not remaining_tool_calls or not remaining_failures:
            return None
        child_budget = TaskBudget(
            max_iterations=max(1, min(remaining_iterations, budget.max_iterations // 2)),
            max_tool_calls=max(1, min(remaining_tool_calls, budget.max_tool_calls // 2)),
            max_same_signature_calls=budget.max_same_signature_calls,
            max_no_progress_steps=budget.max_no_progress_steps,
            max_failures=max(1, min(remaining_failures, budget.max_failures // 2)),
            max_subtasks=1,
            timeout_seconds=max(1.0, budget.timeout_seconds / 2),
            max_result_size=budget.max_result_size,
        )
        parent.metadata["subtasks_used"] = used + 1
        parent.metadata["subtasks_iterations_reserved"] = reserved_iterations + child_budget.max_iterations
        parent.metadata["subtasks_tool_calls_reserved"] = reserved_tool_calls + child_budget.max_tool_calls
        parent.metadata["subtasks_failures_reserved"] = reserved_failures + child_budget.max_failures
        return TaskRuntime(
            TaskState(
                stream_id=parent.stream_id,
                user_goal=objective,
                parent_turn_id=parent.task_id,
                budget=child_budget,
                allowed_tools=parent.allowed_tools,
                denied_tools=parent.denied_tools,
                require_confirmation_tools=parent.require_confirmation_tools,
            )
        )

    async def execute(
        self,
        spec: SubtaskSpec,
        *,
        dependency_context: str = "",
        result_schema: dict[str, object] | None = None,
        validation_steps: tuple[str, ...] = (),
    ) -> TaskResult:
        """执行子任务并把实际消耗计入父任务。"""
        objective = (
            f"{dependency_context}\n{spec.objective}"
            if dependency_context
            else spec.objective
        )
        child = self._create_child(objective)
        if child is None:
            return TaskResult(
                TaskStatus.FAILED,
                "达到父任务子任务预算",
                error="子任务预算耗尽",
            )
        parent = self.parent.state
        try:
            result = await TaskExecutor(self.chatter, child).run(
                TaskRequest(
                    objective=objective,
                    result_schema=result_schema,
                    validation_steps=validation_steps,
                )
            )
        finally:
            parent.metadata["subtasks_iterations_reserved"] = max(
                0,
                int(parent.metadata.get("subtasks_iterations_reserved", 0))
                - child.state.budget.max_iterations,
            )
            parent.metadata["subtasks_tool_calls_reserved"] = max(
                0,
                int(parent.metadata.get("subtasks_tool_calls_reserved", 0))
                - child.state.budget.max_tool_calls,
            )
            parent.metadata["subtasks_failures_reserved"] = max(
                0,
                int(parent.metadata.get("subtasks_failures_reserved", 0))
                - child.state.budget.max_failures,
            )
            parent.iterations += child.state.iterations
            parent.tool_calls += child.state.tool_calls
            parent.failures += child.state.failures
            parent.touch()
        return result


class TaskCollaboration:
    """按依赖和并发上限执行子任务。"""

    def __init__(self, max_concurrency: int = 2, max_subtasks: int = 4) -> None:
        """创建协作编排器。"""
        self.max_concurrency = max(1, max_concurrency)
        self.max_subtasks = max(1, max_subtasks)

    async def run(
        self,
        specs: tuple[SubtaskSpec, ...],
        execute: Callable[..., Awaitable[TaskResult]],
        *,
        parent_budget: int | None = None,
    ) -> TaskResult:
        """执行依赖图并合并结构化结果。"""
        limit = self.max_subtasks
        if parent_budget is not None:
            limit = min(limit, max(0, parent_budget))
        if len(specs) > limit:
            return TaskResult(TaskStatus.FAILED, "子任务数量超过父任务预算")
        names = {spec.name for spec in specs}
        if len(names) != len(specs):
            return TaskResult(TaskStatus.FAILED, "子任务名称重复")
        if any(dependency not in names for spec in specs for dependency in spec.depends_on):
            return TaskResult(TaskStatus.FAILED, "子任务依赖不存在")

        pending = {spec.name: spec for spec in specs}
        results: dict[str, TaskResult] = {}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(spec: SubtaskSpec) -> tuple[str, TaskResult]:
            async with semaphore:
                try:
                    dependency_context = "\n".join(
                        f"前置任务 {name}：{results[name].summary[:1000]}"
                        for name in spec.depends_on
                    )
                    try:
                        result = await execute(
                            spec,
                            dependency_context=dependency_context,
                        )
                    except TypeError as error:
                        if "dependency_context" not in str(error):
                            raise
                        result = await execute(spec)
                    return spec.name, result
                except Exception as error:
                    return spec.name, TaskResult(
                        TaskStatus.FAILED,
                        f"子任务执行异常：{spec.name}",
                        error=str(error),
                    )

        while pending:
            ready = tuple(
                spec
                for spec in pending.values()
                if all(dependency in results for dependency in spec.depends_on)
            )
            if not ready:
                return TaskResult(TaskStatus.FAILED, "子任务依赖存在环")
            batch = await asyncio.gather(*(guarded(spec) for spec in ready))
            for name, result in batch:
                results[name] = result
                pending.pop(name, None)
            if any(result.status == TaskStatus.FAILED for _, result in batch):
                break

        completed = tuple(
            name for name, result in results.items() if result.status == TaskStatus.SUCCEEDED
        )
        failed = tuple(
            name for name, result in results.items() if result.status == TaskStatus.FAILED
        )
        waiting = any(result.status == TaskStatus.WAITING_USER for result in results.values())
        return TaskResult(
            TaskStatus.WAITING_USER if waiting else TaskStatus.FAILED if failed else TaskStatus.SUCCEEDED,
            "；".join(f"{name}：{result.summary}" for name, result in results.items()),
            completed_steps=completed,
            remaining_steps=failed + tuple(pending),
            artifacts=tuple(item for result in results.values() for item in result.artifacts),
            validation=tuple(item for result in results.values() for item in result.validation),
            needs_user_input=waiting,
            error="部分子任务失败" if failed else "",
        )

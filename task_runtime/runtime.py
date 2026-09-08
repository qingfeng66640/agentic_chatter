"""单任务、串行、有限预算的任务运行时。"""

from __future__ import annotations

from typing import Protocol

from .models import TaskBudget, TaskCheckpoint, TaskResult, TaskState, TaskStatus, TaskType
from .policy import ToolPermission, ToolPolicy
from .templates import TaskTemplateRegistry, get_task_templates


class RuntimeStore(Protocol):
    """完整任务状态存储协议。"""

    def save(self, runtime: "TaskRuntime") -> None:
        """保存任务。"""
        ...

    def load(self, task_id: str) -> "TaskRuntime | None":
        """加载任务。"""
        ...


class TaskRuntime:
    """管理一个复杂任务的状态、预算和检查点。"""

    def __init__(self, state: TaskState) -> None:
        """创建任务运行时。"""
        self.state = state
        self.policy = ToolPolicy(
            allowed_tools=state.allowed_tools,
            denied_tools=state.denied_tools,
            require_confirmation_tools=state.require_confirmation_tools,
            default_allow=not bool(state.allowed_tools),
        )

    def start(self) -> None:
        """启动或恢复任务。"""
        if self.state.status == TaskStatus.CREATED:
            self.state.transition(TaskStatus.PLANNING)
            self.state.transition(TaskStatus.RUNNING)
        elif self.state.status in (TaskStatus.PAUSED, TaskStatus.WAITING_USER, TaskStatus.FAILED):
            self.state.transition(TaskStatus.RUNNING)
        else:
            raise ValueError(f"任务当前不可启动：{self.state.status}")

    def check_budget(self) -> TaskResult | None:
        """检查全部任务预算。"""
        budget = self.state.budget
        reason = ""
        if self.state.iterations >= budget.max_iterations:
            reason = "达到任务最大迭代次数"
        elif self.state.tool_calls >= budget.max_tool_calls:
            reason = "达到任务最大工具调用次数"
        elif self.state.no_progress_steps >= budget.max_no_progress_steps:
            reason = "达到任务最大无进展步骤数"
        elif self.state.failures >= budget.max_failures:
            reason = "达到任务最大失败次数"
        elif self.state.elapsed_seconds() >= budget.timeout_seconds:
            reason = "达到任务执行时限"
        if not reason:
            return None
        self.state.status = TaskStatus.PAUSED
        self.state.error = reason
        return self.finish(TaskStatus.PAUSED, reason, error=reason)

    def check_tool(self, name: str, signature: str, policy: ToolPolicy | None = None) -> ToolPermission:
        """在执行入口检查固化权限和重复预算。"""
        permission = self.policy.permission(name)
        if permission == ToolPermission.ALLOW and policy is not None:
            permission = policy.permission(name)
        if permission != ToolPermission.ALLOW:
            return permission
        count = self.state.signature_counts.get(signature, 0)
        if count >= self.state.budget.max_same_signature_calls:
            return ToolPermission.DENY
        if self.state.tool_calls >= self.state.budget.max_tool_calls:
            return ToolPermission.DENY
        self.state.signature_counts[signature] = count + 1
        self.state.tool_calls += 1
        self.state.touch()
        return ToolPermission.ALLOW

    def classify_tool(self, name: str, signature: str) -> ToolPermission:
        """使用任务固化策略判定工具。"""
        return self.check_tool(name, signature)

    def record_iteration(self, progressed: bool) -> TaskResult | None:
        """记录迭代及进展。"""
        self.state.iterations += 1
        self.state.no_progress_steps = 0 if progressed else self.state.no_progress_steps + 1
        self.state.touch()
        return self.check_budget()

    def add_failure(self) -> TaskResult | None:
        """记录一次失败。"""
        self.state.failures += 1
        self.state.touch()
        return self.check_budget()

    def checkpoint(self, step_id: str, summary: str = "", artifacts: tuple[str, ...] = ()) -> None:
        """保存步骤边界的内存检查点。"""
        self.state.current_step = step_id
        if step_id not in self.state.completed_steps:
            self.state.completed_steps.append(step_id)
        checkpoint = TaskCheckpoint(self.state.task_id, step_id, "completed", summary, artifacts)
        self.state.checkpoints.append(checkpoint)
        self.state.last_result_summary = summary
        self.state.metadata["checkpoint_step"] = step_id
        self.state.touch()

    def finish(
        self,
        status: TaskStatus,
        summary: str,
        *,
        error: str = "",
        remaining_steps: tuple[str, ...] = (),
        validation: tuple[str, ...] = (),
    ) -> TaskResult:
        """生成与任务状态一致的有界结果。"""
        terminal = (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.PAUSED,
            TaskStatus.WAITING_USER,
            TaskStatus.CANCELLED,
        )
        if status not in terminal:
            raise ValueError(f"收尾状态必须是任务终态：{status}")
        if self.state.result is not None:
            if self.state.status != status:
                raise ValueError(f"任务已处于终态：{self.state.status}")
            return self.state.result
        if self.state.status != status:
            self.state.transition(status)
        self.state.error = error
        self.state.result = TaskResult(
            status=status,
            summary=summary,
            completed_steps=tuple(self.state.completed_steps),
            remaining_steps=remaining_steps,
            artifacts=tuple(
                artifact
                for checkpoint in self.state.checkpoints
                for artifact in checkpoint.artifacts
            ),
            validation=validation,
            needs_user_input=status == TaskStatus.WAITING_USER,
            error=error,
        ).bounded(self.state.budget.max_result_size)
        self.state.touch()
        return self.state.result

    def adopt_result(self, result: TaskResult) -> TaskResult:
        """接收子任务编排产生的完整结构化结果。"""
        if result.status not in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.WAITING_USER,
            TaskStatus.CANCELLED,
            TaskStatus.PAUSED,
        ):
            raise ValueError(f"不能接收非终态结果：{result.status}")
        if self.state.result is not None:
            return self.state.result
        if self.state.status != result.status:
            self.state.transition(result.status)
        self.state.error = result.error
        self.state.result = result.bounded(self.state.budget.max_result_size)
        self.state.last_result_summary = result.summary
        self.state.touch()
        return self.state.result

    def pause(self) -> TaskResult:
        """暂停运行中的任务。"""
        if self.state.status == TaskStatus.PAUSED:
            return self.state.result or self.finish(TaskStatus.PAUSED, "任务已暂停")
        if self.state.status != TaskStatus.RUNNING:
            raise ValueError(f"任务当前不可暂停：{self.state.status}")
        return self.finish(TaskStatus.PAUSED, "任务已暂停")

    def resume(self) -> None:
        """恢复任务并清除旧终态结果。"""
        if self.state.status not in (TaskStatus.PAUSED, TaskStatus.WAITING_USER, TaskStatus.FAILED):
            raise ValueError(f"任务当前不可恢复：{self.state.status}")
        self.state.result = None
        self.state.error = ""
        self.start()

    def cancel(self) -> TaskResult:
        """取消未完成任务。"""
        if self.state.status == TaskStatus.SUCCEEDED:
            return self.state.result or self.finish(TaskStatus.SUCCEEDED, "任务已完成")
        if self.state.status == TaskStatus.CANCELLED:
            return self.state.result or self.finish(TaskStatus.CANCELLED, "任务已取消")
        if self.state.result is not None:
            self.state.result = None
        return self.finish(TaskStatus.CANCELLED, "任务已取消")

    def complete(
        self,
        summary: str,
        validation: tuple[str, ...] = (),
    ) -> TaskResult:
        """成功完成任务。"""
        return self.finish(TaskStatus.SUCCEEDED, summary, validation=validation)

    def fail(self, summary: str, error: str = "") -> TaskResult:
        """失败结束任务。"""
        return self.finish(TaskStatus.FAILED, summary, error=error)

    def set_waiting_user(self, summary: str) -> TaskResult:
        """等待用户补充。"""
        return self.finish(TaskStatus.WAITING_USER, summary)

    def is_active(self) -> bool:
        """判断任务是否处于执行态。"""
        return self.state.status in (TaskStatus.PLANNING, TaskStatus.RUNNING)

    def snapshot(self) -> dict[str, object]:
        """导出有界状态快照。"""
        return {
            "task_id": self.state.task_id,
            "stream_id": self.state.stream_id,
            "status": self.state.status.value,
            "task_type": self.state.task_type.value,
            "current_step": self.state.current_step,
            "completed_steps": tuple(self.state.completed_steps),
            "iterations": self.state.iterations,
            "tool_calls": self.state.tool_calls,
            "failures": self.state.failures,
            "error": self.state.error,
        }


class TaskRuntimeManager:
    """按聊天流管理单个活动任务。"""

    def __init__(
        self,
        templates: TaskTemplateRegistry | None = None,
        store: RuntimeStore | None = None,
    ) -> None:
        """创建任务管理器。"""
        self._tasks: dict[str, TaskRuntime] = {}
        self._templates = templates or get_task_templates()
        self._store: RuntimeStore | None = None
        self.configure_store(store)

    def create(
        self,
        stream_id: str,
        user_goal: str,
        *,
        task_type: TaskType = TaskType.GENERAL,
        parent_turn_id: str | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        budget: TaskBudget | None = None,
        denied_tools: tuple[str, ...] | None = None,
    ) -> TaskRuntime:
        """为聊天流创建唯一活动任务。"""
        active = self.get_active(stream_id)
        if active is not None:
            raise ValueError(f"聊天流已有活动任务：{active.state.task_id}")
        template = self._templates.get(task_type)
        state = TaskState(
            stream_id=stream_id,
            user_goal=user_goal,
            task_type=task_type,
            parent_turn_id=parent_turn_id,
            allowed_tools=allowed_tools or template.policy.allowed_tools,
            denied_tools=denied_tools or template.policy.denied_tools,
            require_confirmation_tools=template.policy.require_confirmation_tools,
            budget=budget or template.budget,
        )
        runtime = TaskRuntime(state)
        self._tasks[state.task_id] = runtime
        if self._store is not None:
            self._store.save(runtime)
        return runtime

    def configure_store(self, store: RuntimeStore | None) -> None:
        """配置任务状态存储，并登记已有任务。"""
        self._store = store
        if store is None:
            return
        load_active = getattr(store, "load_active", None)
        if callable(load_active):
            for runtime in load_active():
                self.register(runtime)

    def register(self, runtime: TaskRuntime) -> None:
        """注册恢复出的任务运行时。"""
        self._tasks[runtime.state.task_id] = runtime

    def restore(self, task_id: str) -> TaskRuntime | None:
        """从完整状态存储恢复任务。"""
        if self._store is None:
            return None
        runtime = self._store.load(task_id)
        if runtime is not None:
            self._tasks[task_id] = runtime
        return runtime

    def get(self, task_id: str) -> TaskRuntime | None:
        """按 ID 获取任务。"""
        return self._tasks.get(task_id)

    def get_active(self, stream_id: str) -> TaskRuntime | None:
        """获取聊天流的活动或可恢复任务。"""
        active = (
            TaskStatus.CREATED,
            TaskStatus.PLANNING,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_USER,
            TaskStatus.PAUSED,
        )
        return next(
            (
                runtime
                for runtime in self._tasks.values()
                if runtime.state.stream_id == stream_id and runtime.state.status in active
            ),
            None,
        )

    def list_active(self) -> tuple[TaskRuntime, ...]:
        """列出活动或暂停任务。"""
        return tuple(
            runtime
            for runtime in self._tasks.values()
            if runtime.is_active() or runtime.state.status == TaskStatus.PAUSED
        )

    def persist(self, task_id: str) -> bool:
        """持久化指定任务。"""
        runtime = self.get(task_id)
        if runtime is None or self._store is None:
            return False
        self._store.save(runtime)
        return True

    def remove(self, task_id: str) -> bool:
        """移除任务。"""
        return self._tasks.pop(task_id, None) is not None


_TASK_MANAGER = TaskRuntimeManager()


def get_task_runtime_manager() -> TaskRuntimeManager:
    """获取插件级任务管理器。"""
    return _TASK_MANAGER

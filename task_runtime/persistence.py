"""任务状态的 JSON 持久化与恢复。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    TaskBudget,
    TaskCheckpoint,
    TaskResult,
    TaskState,
    TaskStatus,
    TaskType,
)
from .runtime import TaskRuntime


class TaskStateStore:
    """保存完整任务状态的 JSON 存储。"""

    def __init__(self, root: str | Path) -> None:
        """初始化存储目录。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        """构造安全任务路径。"""
        safe = "".join(char for char in str(task_id) if char.isalnum() or char in "-_")
        if not safe:
            raise ValueError("task_id 不能为空")
        return self.root / f"{safe}.json"

    def save(self, runtime: TaskRuntime) -> None:
        """原子保存完整任务状态。"""
        state = asdict(runtime.state)
        state["status"] = runtime.state.status.value
        state["task_type"] = runtime.state.task_type.value
        state["created_at"] = runtime.state.created_at.isoformat()
        state["updated_at"] = runtime.state.updated_at.isoformat()
        state["started_at"] = (
            runtime.state.started_at.isoformat()
            if runtime.state.started_at is not None
            else None
        )
        if runtime.state.result is not None:
            state["result"] = asdict(runtime.state.result)
            state["result"]["status"] = runtime.state.result.status.value
        state["checkpoints"] = [
            {
                "task_id": checkpoint.task_id,
                "step_id": checkpoint.step_id,
                "status": checkpoint.status,
                "summary": checkpoint.summary,
                "artifacts": checkpoint.artifacts,
                "updated_at": checkpoint.updated_at.isoformat(),
            }
            for checkpoint in runtime.state.checkpoints
        ]
        target = self._path(runtime.state.task_id)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def load(self, task_id: str) -> TaskRuntime | None:
        """恢复任务运行时。"""
        path = self._path(task_id)
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        result_data = data.get("result")
        result = None
        if result_data is not None:
            result = TaskResult(
                status=TaskStatus(result_data["status"]),
                summary=str(result_data.get("summary", "")),
                completed_steps=tuple(result_data.get("completed_steps", ())),
                remaining_steps=tuple(result_data.get("remaining_steps", ())),
                artifacts=tuple(result_data.get("artifacts", ())),
                validation=tuple(result_data.get("validation", ())),
                needs_user_input=bool(result_data.get("needs_user_input", False)),
                error=str(result_data.get("error", "")),
            )

        state = TaskState(
            stream_id=str(data["stream_id"]),
            user_goal=str(data["user_goal"]),
            task_type=TaskType(data.get("task_type", TaskType.GENERAL.value)),
            constraints=tuple(data.get("constraints", ())),
            success_criteria=tuple(data.get("success_criteria", ())),
            allowed_tools=tuple(data.get("allowed_tools", ())),
            denied_tools=tuple(data.get("denied_tools", ())),
            require_confirmation_tools=tuple(
                data.get("require_confirmation_tools", ())
            ),
            budget=TaskBudget(**data.get("budget", {})),
            task_id=str(data["task_id"]),
            parent_turn_id=data.get("parent_turn_id"),
            status=TaskStatus(data.get("status", TaskStatus.PAUSED.value)),
            current_step=str(data.get("current_step", "")),
            completed_steps=list(data.get("completed_steps", ())),
            iterations=int(data.get("iterations", 0)),
            tool_calls=int(data.get("tool_calls", 0)),
            failures=int(data.get("failures", 0)),
            no_progress_steps=int(data.get("no_progress_steps", 0)),
            signature_counts=dict(data.get("signature_counts", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_result_summary=str(data.get("last_result_summary", "")),
            error=str(data.get("error", "")),
            result=result,
            metadata=dict(data.get("metadata", {})),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
        )
        state.checkpoints = [
            TaskCheckpoint(
                task_id=str(item["task_id"]),
                step_id=str(item["step_id"]),
                status=str(item["status"]),
                summary=str(item.get("summary", "")),
                artifacts=tuple(item.get("artifacts", ())),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            for item in data.get("checkpoints", ())
        ]
        return TaskRuntime(state)

    def list_task_ids(self) -> tuple[str, ...]:
        """列出存储中的任务 ID。"""
        return tuple(path.stem for path in sorted(self.root.glob("*.json")))

    def load_active(self) -> tuple[TaskRuntime, ...]:
        """恢复全部可继续任务。"""
        active_statuses = {
            TaskStatus.CREATED,
            TaskStatus.PLANNING,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_USER,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
        }
        runtimes = tuple(
            runtime
            for task_id in self.list_task_ids()
            if (runtime := self.load(task_id)) is not None
            and runtime.state.status in active_statuses
        )
        return runtimes

    def delete(self, task_id: str) -> None:
        """删除任务状态。"""
        path = self._path(task_id)
        if path.exists():
            path.unlink()

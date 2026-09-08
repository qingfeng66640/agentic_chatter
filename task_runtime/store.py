"""任务检查点持久化接口及 JSON 文件实现。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import TaskCheckpoint


class TaskCheckpointStore:
    """检查点存储协议的最小实现接口。"""

    def save(self, checkpoint: TaskCheckpoint) -> None:
        """保存检查点。"""
        raise NotImplementedError

    def load(self, task_id: str) -> TaskCheckpoint | None:
        """加载检查点。"""
        raise NotImplementedError

    def delete(self, task_id: str) -> None:
        """删除检查点。"""
        raise NotImplementedError


class JsonCheckpointStore(TaskCheckpointStore):
    """以每任务一个 JSON 文件保存检查点。"""

    def __init__(self, root: str | Path) -> None:
        """创建检查点目录。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        """获取安全的任务文件路径。"""
        safe_id = "".join(char for char in task_id if char.isalnum() or char in "-_")
        if not safe_id:
            raise ValueError("task_id 不能为空")
        return self.root / f"{safe_id}.json"

    def save(self, checkpoint: TaskCheckpoint) -> None:
        """原子替换保存检查点。"""
        target = self._path(checkpoint.task_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        data: dict[str, Any] = asdict(checkpoint)
        data["updated_at"] = checkpoint.updated_at.isoformat()
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def load(self, task_id: str) -> TaskCheckpoint | None:
        """读取检查点。"""
        path = self._path(task_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        from datetime import datetime

        return TaskCheckpoint(
            task_id=str(data["task_id"]),
            step_id=str(data["step_id"]),
            status=str(data["status"]),
            summary=str(data.get("summary", "")),
            artifacts=tuple(data.get("artifacts", ())),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def delete(self, task_id: str) -> None:
        """删除检查点。"""
        path = self._path(task_id)
        if path.exists():
            path.unlink()

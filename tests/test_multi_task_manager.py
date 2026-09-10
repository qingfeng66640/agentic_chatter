"""TaskRuntimeManager 多任务并发的测试。"""

from __future__ import annotations

import pytest

from ..task_runtime.models import TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntimeManager


def test_get_active_tasks_sorted_by_created_at() -> None:
    manager = TaskRuntimeManager()
    first = manager.create("stream", "第一个", task_type=TaskType.RESEARCH)
    second = manager.create("stream", "第二个")
    tasks = manager.get_active_tasks("stream")
    assert [task.state.task_id for task in tasks] == [
        first.state.task_id,
        second.state.task_id,
    ]


def test_get_active_tasks_excludes_terminal_statuses() -> None:
    manager = TaskRuntimeManager()
    done = manager.create("stream", "已完成")
    done.start()
    done.complete("结果")
    paused = manager.create("stream", "已暂停")
    paused.start()
    paused.pause()
    tasks = manager.get_active_tasks("stream")
    assert [task.state.task_id for task in tasks] == [paused.state.task_id]


def test_get_active_shim_returns_latest_task() -> None:
    manager = TaskRuntimeManager()
    manager.create("stream", "第一个")
    second = manager.create("stream", "第二个")
    assert manager.get_active("stream") is second
    assert manager.get_active("other-stream") is None


def test_create_rejects_when_capacity_reached() -> None:
    manager = TaskRuntimeManager()
    manager.create("stream", "第一个")
    manager.create("stream", "第二个")
    with pytest.raises(ValueError, match="已达上限"):
        manager.create("stream", "第三个")


def test_capacity_does_not_count_other_streams() -> None:
    manager = TaskRuntimeManager()
    manager.create("stream-a", "A流任务一")
    manager.create("stream-a", "A流任务二")
    task = manager.create("stream-b", "B流任务")
    assert task.state.stream_id == "stream-b"


def test_remove_reduces_active_count() -> None:
    manager = TaskRuntimeManager()
    first = manager.create("stream", "第一个")
    manager.create("stream", "第二个")
    assert manager.remove(first.state.task_id) is True
    assert len(manager.get_active_tasks("stream")) == 1


def test_concurrent_recovery_tasks_share_capacity() -> None:
    manager = TaskRuntimeManager()
    manager.configure_limits(1)
    paused = manager.create("stream", "暂停任务")
    paused.start()
    paused.pause()
    with pytest.raises(ValueError, match="已达上限"):
        manager.create("stream", "新任务")
    paused.cancel()
    task = manager.create("stream", "新任务")
    assert task.state.status == TaskStatus.CREATED

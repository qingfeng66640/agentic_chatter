"""任务消息 mailbox 分流。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .control import TaskMessage, TaskMessageKind, classify_task_message


@dataclass(frozen=True, slots=True)
class TaskEnvelope:
    """绑定任务版本的消息信封。"""

    task_id: str
    generation: int
    message: TaskMessage


class TaskMessageMailbox:
    """按 task_id 隔离控制和补充输入。"""

    def __init__(self) -> None:
        """创建任务消息 mailbox。"""
        self._queues: dict[str, asyncio.Queue[TaskEnvelope]] = {}
        self._generations: dict[str, int] = {}

    def _queue(self, task_id: str) -> asyncio.Queue[TaskEnvelope]:
        """获取任务队列。"""
        return self._queues.setdefault(task_id, asyncio.Queue())

    async def put(self, task_id: str, text: str, *, kind: TaskMessageKind | None = None) -> TaskEnvelope:
        """写入任务控制或补充消息。"""
        generation = self._generations.get(task_id, 0) + 1
        self._generations[task_id] = generation
        message = classify_task_message(text, task_id)
        if kind is not None:
            message = TaskMessage(kind, message.text, task_id)
        envelope = TaskEnvelope(task_id, generation, message)
        await self._queue(task_id).put(envelope)
        return envelope

    async def get(self, task_id: str) -> TaskEnvelope:
        """读取任务消息。"""
        return await self._queue(task_id).get()

    def generation(self, task_id: str) -> int:
        """获取任务最新消息版本。"""
        return self._generations.get(task_id, 0)

    def discard(self, task_id: str) -> None:
        """丢弃任务消息队列。"""
        self._queues.pop(task_id, None)
        self._generations.pop(task_id, None)

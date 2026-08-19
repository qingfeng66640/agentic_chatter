"""按聊天流隔离的连续输入 mailbox。

mailbox 固定每个回合领取的消息快照，并把生成期间到达的新消息留给下一回合。
锁只保护内存状态迁移，不覆盖模型、发送或未读确认等外部操作。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


def message_key(message: Any) -> str:
    """返回消息跨未读快照稳定的键。"""
    message_id = str(getattr(message, "message_id", "") or "").strip()
    if message_id:
        return f"id:{message_id}"

    fingerprint = {
        "stream_id": str(getattr(message, "stream_id", "") or ""),
        "time": getattr(message, "time", None),
        "sender_id": str(getattr(message, "sender_id", "") or ""),
        "sender_name": str(getattr(message, "sender_name", "") or ""),
        "message_type": str(getattr(message, "message_type", "") or ""),
        "reply_to": str(getattr(message, "reply_to", "") or ""),
        "content": str(getattr(message, "content", "") or ""),
        "processed_plain_text": str(
            getattr(message, "processed_plain_text", "") or ""
        ),
    }
    serialized = json.dumps(
        fingerprint,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"fallback:{digest}"


@dataclass(frozen=True)
class TurnClaim:
    """一次回合原子领取的不可变消息集合。"""

    stream_id: str
    owner: object
    generation: int
    keys: tuple[str, ...]
    messages: tuple[Any, ...]


class StreamMailbox:
    """单个聊天流的 pending/claimed 消息状态。"""

    def __init__(self, stream_id: str) -> None:
        """初始化空 mailbox。"""
        self.stream_id = stream_id
        self._lock = asyncio.Lock()
        self._owner: object | None = None
        self._generation = 0
        self._pending: dict[str, Any] = {}
        self._pending_started_at: float | None = None
        self._consecutive_interruptions = 0
        self._claim: TurnClaim | None = None

    async def pending_state(self, window_seconds: float) -> tuple[int, float]:
        """返回 pending 数量及固定合并窗口剩余秒数。"""
        async with self._lock:
            if not self._pending or self._pending_started_at is None:
                return 0, 0.0
            remaining = max(
                0.0,
                float(window_seconds) - (time.monotonic() - self._pending_started_at),
            )
            return len(self._pending), remaining

    async def interruption_state(self) -> int:
        """返回当前 stream 的连续输入中断次数。"""
        async with self._lock:
            return self._consecutive_interruptions

    async def record_interruption(self) -> int:
        """记录一次实际发生的输入中断并返回最新次数。"""
        async with self._lock:
            self._consecutive_interruptions += 1
            return self._consecutive_interruptions

    async def reset_interruptions(self) -> None:
        """重置当前 stream 的连续输入中断次数。"""
        async with self._lock:
            self._consecutive_interruptions = 0

    async def has_pending_key(self, key: str) -> bool:
        """判断指定消息键是否仍在 pending。"""
        async with self._lock:
            return key in self._pending

    async def pending_count(self) -> int:
        """返回当前 pending 消息数量。"""
        async with self._lock:
            return len(self._pending)

    async def try_acquire(self, owner: object) -> int | None:
        """尝试取得当前 stream 的执行所有权。"""
        async with self._lock:
            if self._owner is not None:
                return None
            self._generation += 1
            self._owner = owner
            return self._generation

    async def merge_snapshot(self, messages: list[Any] | tuple[Any, ...]) -> int:
        """按首次出现顺序合并未读快照，并返回新增消息数。"""
        async with self._lock:
            claimed_keys = set(self._claim.keys) if self._claim is not None else set()
            added = 0
            for message in messages:
                key = message_key(message)
                if key in claimed_keys or key in self._pending:
                    continue
                self._pending[key] = message
                added += 1
            if added and self._pending_started_at is None:
                self._pending_started_at = time.monotonic()
            return added

    async def claim_pending(self, owner: object, generation: int) -> TurnClaim | None:
        """由当前 owner 原子领取全部 pending 消息。"""
        async with self._lock:
            if not self._matches(owner, generation) or self._claim is not None:
                return None
            if not self._pending:
                return None
            claim = TurnClaim(
                stream_id=self.stream_id,
                owner=owner,
                generation=generation,
                keys=tuple(self._pending),
                messages=tuple(self._pending.values()),
            )
            self._pending.clear()
            self._pending_started_at = None
            self._claim = claim
            return claim

    async def release_claim(self, claim: TurnClaim) -> bool:
        """释放失败回合的 claim，并把消息恢复到 pending 前部。"""
        async with self._lock:
            if not self._is_current_claim(claim):
                return False
            restored = dict(zip(claim.keys, claim.messages, strict=True))
            restored.update(self._pending)
            self._pending = restored
            if self._pending and self._pending_started_at is None:
                self._pending_started_at = time.monotonic()
            self._claim = None
            return True

    async def commit_claim(self, claim: TurnClaim) -> bool:
        """确认当前 claim，不影响运行期间新增的 pending 消息。"""
        async with self._lock:
            if not self._is_current_claim(claim):
                return False
            self._claim = None
            self._consecutive_interruptions = 0
            return True

    async def release_owner(self, owner: object, generation: int) -> bool:
        """释放当前执行所有权；错误 owner 或 generation 不生效。"""
        async with self._lock:
            if not self._matches(owner, generation):
                return False
            if self._claim is not None:
                return False
            self._owner = None
            return True

    async def pending_keys(self) -> tuple[str, ...]:
        """返回当前 pending 消息键快照。"""
        async with self._lock:
            return tuple(self._pending)

    def _matches(self, owner: object, generation: int) -> bool:
        """检查 owner 与 generation 是否仍指向当前回合。"""
        return self._owner is owner and self._generation == generation

    def _is_current_claim(self, claim: TurnClaim) -> bool:
        """检查 claim 是否属于当前 owner 和 generation。"""
        return (
            self._claim is claim
            and claim.stream_id == self.stream_id
            and self._matches(claim.owner, claim.generation)
        )


_MAILBOXES: dict[str, StreamMailbox] = {}


def get_stream_mailbox(stream_id: str) -> StreamMailbox:
    """获取指定 stream 的进程内 mailbox。"""
    mailbox = _MAILBOXES.get(stream_id)
    if mailbox is None:
        mailbox = StreamMailbox(stream_id)
        _MAILBOXES[stream_id] = mailbox
    return mailbox


def clear_mailbox_registry() -> None:
    """清空 mailbox registry，供测试隔离使用。"""
    _MAILBOXES.clear()

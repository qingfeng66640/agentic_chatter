"""按聊天流隔离的连续输入 mailbox。

mailbox 固定每个回合领取的消息快照，并把生成期间到达的新消息留给下一回合。
锁只保护内存状态迁移，不覆盖模型、发送或未读确认等外部操作。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


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
        "processed_plain_text": str(getattr(message, "processed_plain_text", "") or ""),
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


@dataclass(frozen=True)
class FailedTurn:
    """失败回合的有界诊断摘要，不保存消息正文或供应商错误正文。"""

    generation: int
    message_count: int
    message_digest: str
    failure_kind: str
    error_type: str
    status_code: int | None
    spoke: bool
    tool_count: int
    input_acknowledged: bool
    acknowledgement_error_type: str


MAX_FAILED_TURN_RECORDS = 32


def safe_error_type(error_type: str) -> str:
    """仅保留异常类型名中的标识符字符，避免控制字符或正文混入日志。"""
    return (
        "".join(
            character
            for character in error_type
            if character.isascii() and (character.isalnum() or character in "_.")
        )[:80]
        or "UnknownError"
    )


class StreamMailbox:
    """单个聊天流的待处理、已领取和失败隔离状态。

    失败键在 mailbox 生命周期内不驱逐，避免摘要淘汰后旧快照再次触发执行。
    未确认的失败输入保留原对象；已确认输入仅保留键和有界诊断摘要。
    这些状态仅存在于当前进程，不提供跨重启恢复或自动重试。
    """

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
        self._failed_keys: set[str] = set()
        self._failed_messages: dict[str, Any] = {}
        self._failed_turns: deque[FailedTurn] = deque(maxlen=MAX_FAILED_TURN_RECORDS)

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

    async def merge_snapshot(
        self,
        messages: list[Any] | tuple[Any, ...],
        *,
        owner: object | None = None,
        generation: int | None = None,
    ) -> int:
        """合并未读快照，跳过失败输入并返回新增消息数。

        Args:
            messages: 核心提供的未读消息快照。
            owner: 快照读取者的所有权标识；传入时必须与 generation 同时有效。
            generation: 读取快照时的回合代次，用于拒绝过期回合的迟到合并。
        """
        async with self._lock:
            if (owner is not None or generation is not None) and not self._matches(
                owner, generation
            ):
                return 0
            claimed_keys = set(self._claim.keys) if self._claim is not None else set()
            added = 0
            for message in messages:
                key = message_key(message)
                if (
                    key in claimed_keys
                    or key in self._pending
                    or key in self._failed_keys
                ):
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

    async def commit_claim(
        self,
        claim: TurnClaim,
        apply: Callable[[], None] | None = None,
    ) -> bool:
        """确认当前 claim；可在同一锁内先应用外部上下文变更。"""
        async with self._lock:
            if not self._is_current_claim(claim):
                return False
            if apply is not None:
                apply()
            self._claim = None
            self._consecutive_interruptions = 0
            return True

    async def fail_claim(
        self,
        claim: TurnClaim,
        *,
        failure_kind: str,
        error_type: str,
        spoke: bool,
        tool_count: int,
        status_code: int | None = None,
        apply: Callable[[], None] | None = None,
    ) -> FailedTurn | None:
        """终结失败回合，隔离输入后再尝试确认其已被处理。

        Args:
            claim: 当前所有者领取的消息快照。
            failure_kind: model_error 或 execution_error，其他值归为后者。
            error_type: 异常类名，不接受错误正文作为诊断内容。
            spoke: 本回合是否已确认发言。
            tool_count: 已记录的工具执行数量，不代表全部副作用均已确认。
            status_code: 可取得的 HTTP 状态码，仅保留 100 到 599 的整数。
            apply: 将原输入归入历史的同步回调；应自行保证异常时的就地回滚。

        Returns:
            FailedTurn | None: 失败记录；claim 已失效时返回 None 且不调用 apply。

        失败终态独立于输入确认。apply 异常不会撤销隔离或重新入队，
        原消息继续留在失败隔离区；确认成功也不表示业务执行成功。
        """
        async with self._lock:
            if not self._is_current_claim(claim):
                return None
            self._failed_keys.update(claim.keys)
            self._failed_messages.update(zip(claim.keys, claim.messages, strict=True))
            self._claim = None
            self._consecutive_interruptions = 0
            acknowledged = False
            acknowledgement_error = ""
            try:
                if apply is not None:
                    apply()
                    acknowledged = True
            except Exception as exc:
                acknowledgement_error = safe_error_type(type(exc).__name__)
            finally:
                if acknowledged:
                    for key in claim.keys:
                        self._failed_messages.pop(key, None)
                record = FailedTurn(
                    generation=claim.generation,
                    message_count=len(claim.messages),
                    message_digest=hashlib.sha256(
                        "\n".join(claim.keys).encode("utf-8")
                    ).hexdigest()[:12],
                    failure_kind=(
                        failure_kind
                        if failure_kind in ("model_error", "execution_error")
                        else "execution_error"
                    ),
                    error_type=safe_error_type(error_type),
                    status_code=(
                        status_code
                        if type(status_code) is int and 100 <= status_code <= 599
                        else None
                    ),
                    spoke=bool(spoke),
                    tool_count=max(0, int(tool_count)),
                    input_acknowledged=acknowledged,
                    acknowledgement_error_type=acknowledgement_error,
                )
                self._failed_turns.append(record)
            return record

    async def recent_failed_turns(self) -> tuple[FailedTurn, ...]:
        """返回最近的失败摘要；摘要淘汰不会移除失败隔离键。"""
        async with self._lock:
            return tuple(self._failed_turns)

    async def failed_messages(self) -> tuple[Any, ...]:
        """返回尚未安全归入历史的失败输入，仅供诊断，不自动重新入队。"""
        async with self._lock:
            return tuple(self._failed_messages.values())

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

    def _matches(self, owner: object | None, generation: int | None) -> bool:
        """检查 owner 与 generation 是否仍指向当前回合。"""
        return (
            owner is not None
            and self._owner is owner
            and self._generation == generation
        )

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

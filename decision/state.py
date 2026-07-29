"""聊天流参与状态。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock


@dataclass(slots=True)
class ParticipationState:
    """一个聊天流中的近期参与状态。"""

    last_seen_at: float = 0.0
    last_reply_at: float = 0.0
    last_reply_topic: str = ""
    consecutive_participation: int = 0
    consecutive_silence: int = 0


class ParticipationStore:
    """按聊天流隔离并自动过期的参与状态仓库。"""

    def __init__(self) -> None:
        """初始化状态仓库。"""
        self._states: dict[str, ParticipationState] = {}
        self._lock = RLock()

    def get(
        self,
        stream_id: str,
        *,
        ttl_seconds: float,
        max_streams: int,
        now: float | None = None,
    ) -> ParticipationState:
        """获取聊天流状态，并清理过期或过量记录。"""
        current = time.time() if now is None else now
        with self._lock:
            self._prune(current, ttl_seconds, max_streams)
            return self._states.setdefault(stream_id, ParticipationState())

    def record(
        self,
        stream_id: str,
        *,
        responded: bool,
        topic: str,
        ttl_seconds: float,
        max_streams: int,
        now: float | None = None,
    ) -> None:
        """记录一轮已完成的回复或静默决策。"""
        current = time.time() if now is None else now
        with self._lock:
            self._prune(current, ttl_seconds, max_streams)
            state = self._states.setdefault(stream_id, ParticipationState())
            state.last_seen_at = current
            if responded:
                state.last_reply_at = current
                state.last_reply_topic = topic[:80]
                state.consecutive_participation += 1
                state.consecutive_silence = 0
            else:
                state.consecutive_silence += 1
                state.consecutive_participation = 0

    def clear(self) -> None:
        """清空所有状态，供测试或卸载时使用。"""
        with self._lock:
            self._states.clear()

    def _prune(self, now: float, ttl_seconds: float, max_streams: int) -> None:
        """移除过期记录并限制流数量。"""
        ttl = max(1.0, ttl_seconds)
        expired = [
            stream_id
            for stream_id, state in self._states.items()
            if state.last_seen_at and now - state.last_seen_at > ttl
        ]
        for stream_id in expired:
            self._states.pop(stream_id, None)

        limit = max(1, max_streams)
        if len(self._states) <= limit:
            return
        oldest = sorted(
            self._states,
            key=lambda stream_id: self._states[stream_id].last_seen_at,
        )
        for stream_id in oldest[: len(self._states) - limit]:
            self._states.pop(stream_id, None)


_STORE = ParticipationStore()


def get_participation_store() -> ParticipationStore:
    """返回全局参与状态仓库。"""
    return _STORE

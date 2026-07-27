"""跨流全局心智存储。

本模块提供 ``GlobalMind`` 进程级单例，用于在多个聊天流之间共享
情绪状态与高度压缩的对话摘要，解决 bot 在不同群聊中人格割裂的问题。

设计要点：

1. **必须是模块级单例，不能做成 Service**。框架中 ``service_api.get_service()``
   每次返回新实例，无法承载跨流共享状态。
2. **只存压缩摘要，不存原文**。跨流信息一旦膨胀就会稀释模型对当前
   对话的注意力，因此每条摘要都有长度上限，渲染时还有总长硬截断。
3. **纯内存 + 惰性过期**。不做持久化，重启即重置；过期流在读取时过滤，
   不起后台清理任务。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock

# 单条流摘要主题的最大字符数，超出截断
MAX_TOPIC_CHARS = 40
# 单条近期要闻的最大字符数，超出截断
MAX_NOTE_CHARS = 60
# 情绪值的取值范围
MOOD_MIN = -1.0
MOOD_MAX = 1.0


def _truncate(text: str, limit: int) -> str:
    """将文本截断到指定长度，超出部分以省略号结尾。

    Args:
        text: 原始文本。
        limit: 最大字符数，必须为正数。

    Returns:
        str: 截断后的文本；原文本不超长时原样返回。
    """
    cleaned = " ".join(str(text or "").split())
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)] + "…"


@dataclass
class StreamDigest:
    """单个聊天流的压缩摘要。

    Attributes:
        stream_id: 聊天流 ID。
        stream_name: 聊天流显示名（群名或对方昵称）。
        chat_type: 聊天类型，private 或 group。
        topic: 一句话主题，描述该流当前在聊什么。
        unread_count: 该流当前的未读消息数。
        updated_at: 摘要最后更新时间戳。
    """

    stream_id: str
    stream_name: str = ""
    chat_type: str = ""
    topic: str = ""
    unread_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def age_minutes(self, now: float | None = None) -> float:
        """计算摘要距今的分钟数。

        Args:
            now: 参考时间戳；为 None 时取当前时间。

        Returns:
            float: 距今分钟数，最小为 0。
        """
        reference = time.time() if now is None else now
        return max(0.0, (reference - self.updated_at) / 60.0)


@dataclass
class MoodState:
    """情绪状态。

    情绪用一个连续值表示，正数偏愉悦、负数偏烦躁，基线为 0。
    情绪不会永久停留，会随时间自然衰减回基线。

    Attributes:
        value: 情绪值，取值范围 [-1.0, 1.0]。
        label: 最近一次情绪变化的简短描述，用于渲染给模型看。
        updated_at: 情绪最后更新时间戳。
    """

    value: float = 0.0
    label: str = ""
    updated_at: float = field(default_factory=time.time)

    def decayed(self, decay_per_minute: float, now: float | None = None) -> float:
        """计算按时间衰减后的情绪值。

        Args:
            decay_per_minute: 每分钟向基线衰减的幅度，非负数。
            now: 参考时间戳；为 None 时取当前时间。

        Returns:
            float: 衰减后的情绪值；衰减到跨越基线时收敛为 0。
        """
        reference = time.time() if now is None else now
        elapsed_minutes = max(0.0, (reference - self.updated_at) / 60.0)
        decay = max(0.0, float(decay_per_minute)) * elapsed_minutes

        if self.value > 0:
            return max(0.0, self.value - decay)
        if self.value < 0:
            return min(0.0, self.value + decay)
        return 0.0

    def describe(self, decay_per_minute: float, now: float | None = None) -> str:
        """将当前情绪渲染为一句自然语言描述。

        Args:
            decay_per_minute: 每分钟衰减幅度。
            now: 参考时间戳；为 None 时取当前时间。

        Returns:
            str: 情绪描述；情绪接近基线时返回空字符串。
        """
        current = self.decayed(decay_per_minute, now)

        if current >= 0.6:
            mood_word = "心情很好"
        elif current >= 0.2:
            mood_word = "心情不错"
        elif current <= -0.6:
            mood_word = "有点烦躁"
        elif current <= -0.2:
            mood_word = "情绪一般"
        else:
            return ""

        if self.label:
            return f"{mood_word}（{_truncate(self.label, MAX_NOTE_CHARS)}）"
        return mood_word


class GlobalMind:
    """跨流全局心智。

    维护三类跨流共享状态：

    - **情绪**：一份全局 mood，在任意流中产生的情绪会带到其他流。
    - **流摘要**：每个流一行压缩摘要，让 bot 知道自己在别处聊了什么。
    - **近期要闻**：跨流可见的重要事项，例如对某人做出的承诺。

    本类是线程安全的，所有读写都通过内部锁保护。
    """

    def __init__(self) -> None:
        """初始化空的全局心智。"""
        self._lock = RLock()
        self._mood = MoodState()
        self._stream_moods: dict[str, MoodState] = {}
        self._digests: dict[str, StreamDigest] = {}
        self._notes: deque[str] = deque(maxlen=64)

    # ------------------------------------------------------------------
    # 情绪
    # ------------------------------------------------------------------

    def get_mood(self) -> MoodState:
        """获取当前情绪状态的快照。

        Returns:
            MoodState: 情绪状态副本，修改副本不会影响内部状态。
        """
        with self._lock:
            return MoodState(
                value=self._mood.value,
                label=self._mood.label,
                updated_at=self._mood.updated_at,
            )

    def nudge_mood(
        self,
        delta: float,
        label: str = "",
        *,
        decay_per_minute: float = 0.05,
    ) -> float:
        """在当前情绪基础上叠加一个增量。

        叠加前会先按时间衰减，确保久远的情绪不会一直累积。

        Args:
            delta: 情绪增量，正数偏愉悦、负数偏烦躁。
            label: 本次情绪变化的简短原因描述。
            decay_per_minute: 每分钟衰减幅度，用于叠加前的衰减计算。

        Returns:
            float: 叠加并钳制到合法区间后的新情绪值。
        """
        with self._lock:
            base = self._mood.decayed(decay_per_minute)
            updated = max(MOOD_MIN, min(MOOD_MAX, base + float(delta)))
            self._mood = MoodState(
                value=updated,
                label=_truncate(label, MAX_NOTE_CHARS) if label else self._mood.label,
                updated_at=time.time(),
            )
            return updated

    def get_stream_mood(self, stream_id: str) -> MoodState:
        """获取某个聊天流的独立情绪快照。

        Args:
            stream_id: 聊天流 ID。

        Returns:
            MoodState: 独立情绪状态副本。
        """
        with self._lock:
            mood = self._stream_moods.get(str(stream_id or "").strip(), MoodState())
            return MoodState(value=mood.value, label=mood.label, updated_at=mood.updated_at)

    def nudge_stream_mood(
        self,
        stream_id: str,
        delta: float,
        label: str = "",
        *,
        decay_per_minute: float = 0.05,
    ) -> float:
        """调整某个聊天流的独立情绪。

        Args:
            stream_id: 聊天流 ID。
            delta: 情绪增量。
            label: 变化原因。
            decay_per_minute: 每分钟衰减幅度。

        Returns:
            float: 调整后的情绪值。
        """
        key = str(stream_id or "").strip()
        with self._lock:
            current = self._stream_moods.get(key, MoodState())
            value = max(
                MOOD_MIN,
                min(MOOD_MAX, current.decayed(decay_per_minute) + float(delta)),
            )
            self._stream_moods[key] = MoodState(
                value=value,
                label=_truncate(label, MAX_NOTE_CHARS) if label else current.label,
                updated_at=time.time(),
            )
            return value

    def reset_mood(self) -> None:
        """将全局与各流独立情绪重置回基线。"""
        with self._lock:
            self._mood = MoodState()
            self._stream_moods.clear()

    # ------------------------------------------------------------------
    # 流摘要
    # ------------------------------------------------------------------

    def update_digest(
        self,
        stream_id: str,
        *,
        stream_name: str = "",
        chat_type: str = "",
        topic: str = "",
        unread_count: int = 0,
    ) -> None:
        """写入或更新某个聊天流的压缩摘要。

        Args:
            stream_id: 聊天流 ID；为空时直接忽略本次调用。
            stream_name: 聊天流显示名。
            chat_type: 聊天类型。
            topic: 一句话主题，会被截断到 ``MAX_TOPIC_CHARS``。
            unread_count: 当前未读消息数。
        """
        key = str(stream_id or "").strip()
        if not key:
            return

        with self._lock:
            existing = self._digests.get(key)
            self._digests[key] = StreamDigest(
                stream_id=key,
                stream_name=stream_name or (existing.stream_name if existing else ""),
                chat_type=chat_type or (existing.chat_type if existing else ""),
                topic=_truncate(topic, MAX_TOPIC_CHARS)
                or (existing.topic if existing else ""),
                unread_count=max(0, int(unread_count)),
                updated_at=time.time(),
            )

    def list_digests(
        self,
        *,
        exclude_stream_id: str = "",
        max_streams: int = 6,
        stale_minutes: float = 120.0,
        now: float | None = None,
    ) -> list[StreamDigest]:
        """列出其他聊天流的摘要，按最近活跃度排序。

        Args:
            exclude_stream_id: 要排除的流 ID，通常是当前正在处理的流。
            max_streams: 最多返回几条摘要。
            stale_minutes: 过期阈值（分钟），超过该时长未更新的流会被过滤。
            now: 参考时间戳；为 None 时取当前时间。

        Returns:
            list[StreamDigest]: 按更新时间倒序排列的摘要列表。
        """
        if max_streams <= 0:
            return []

        excluded = str(exclude_stream_id or "").strip()
        reference = time.time() if now is None else now
        threshold = max(0.0, float(stale_minutes))

        with self._lock:
            candidates = [
                digest
                for digest in self._digests.values()
                if digest.stream_id != excluded
                and digest.age_minutes(reference) <= threshold
            ]

        candidates.sort(key=lambda item: item.updated_at, reverse=True)
        return candidates[:max_streams]

    def drop_digest(self, stream_id: str) -> bool:
        """移除某个聊天流的摘要。

        Args:
            stream_id: 要移除的流 ID。

        Returns:
            bool: 是否确实移除了一条摘要。
        """
        with self._lock:
            return self._digests.pop(str(stream_id or "").strip(), None) is not None

    # ------------------------------------------------------------------
    # 近期要闻
    # ------------------------------------------------------------------

    def add_note(self, note: str, *, limit: int = 8) -> None:
        """追加一条跨流可见的近期要闻。

        重复内容不会被重复追加；超出上限时最旧的要闻会被挤出。

        Args:
            note: 要闻内容，会被截断到 ``MAX_NOTE_CHARS``。
            limit: 保留的要闻条数上限。
        """
        text = _truncate(note, MAX_NOTE_CHARS)
        if not text:
            return

        with self._lock:
            if text in self._notes:
                return
            self._notes.append(text)
            while limit > 0 and len(self._notes) > limit:
                self._notes.popleft()

    def list_notes(self, *, limit: int = 8) -> list[str]:
        """列出近期要闻，最新的排在最后。

        Args:
            limit: 最多返回几条。

        Returns:
            list[str]: 要闻列表。
        """
        if limit <= 0:
            return []
        with self._lock:
            return list(self._notes)[-limit:]

    def clear(self) -> None:
        """清空全部全局心智状态。主要供测试使用。"""
        with self._lock:
            self._mood = MoodState()
            self._stream_moods.clear()
            self._digests.clear()
            self._notes.clear()


_GLOBAL_MIND = GlobalMind()


def get_global_mind() -> GlobalMind:
    """获取进程级全局心智单例。

    Returns:
        GlobalMind: 全局唯一的心智实例。
    """
    return _GLOBAL_MIND

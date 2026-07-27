"""全局感知块渲染。

将 :class:`GlobalMind` 中的跨流状态渲染成一段紧凑的文本，注入到
提示词中。渲染过程有严格的长度控制：跨流信息一旦膨胀就会稀释
模型对当前对话的注意力，因此这里按优先级逐段追加，一旦超出
``max_chars`` 上限就停止。

优先级从高到低：情绪 > 其他流摘要 > 近期要闻。情绪最短且最影响
表达风格，因此优先保留。
"""

from __future__ import annotations

from .store import GlobalMind, StreamDigest

# 渲染块的包裹标签
BLOCK_OPEN = "<global_awareness>"
BLOCK_CLOSE = "</global_awareness>"


def _format_digest_line(digest: StreamDigest) -> str:
    """将单条流摘要渲染为一行文本。

    Args:
        digest: 流摘要。

    Returns:
        str: 形如 ``- 某某群 · 3分钟前 · 在聊周末爬山 · 2条未读`` 的文本行。
    """
    parts: list[str] = []

    name = digest.stream_name.strip() or ("私聊" if digest.chat_type == "private" else "某个对话")
    parts.append(name)

    age = digest.age_minutes()
    if age < 1:
        parts.append("刚刚")
    elif age < 60:
        parts.append(f"{int(age)}分钟前")
    else:
        parts.append(f"{int(age / 60)}小时前")

    if digest.topic:
        parts.append(f"在聊{digest.topic}")

    if digest.unread_count > 0:
        parts.append(f"{digest.unread_count}条未读")

    return "- " + " · ".join(parts)


def render_global_awareness(
    mind: GlobalMind,
    *,
    current_stream_id: str,
    max_chars: int = 600,
    max_streams: int = 6,
    stale_minutes: float = 120.0,
    share_mood: bool = True,
    mood_decay_per_minute: float = 0.05,
    recent_notes_limit: int = 8,
) -> str:
    """渲染全局感知块。

    Args:
        mind: 全局心智实例。
        current_stream_id: 当前正在处理的流 ID，其摘要不会重复渲染。
        max_chars: 渲染结果的总字符数硬上限（含包裹标签）。
        max_streams: 最多渲染几个其他流的摘要。
        stale_minutes: 流摘要的过期阈值（分钟）。
        share_mood: 是否渲染情绪。
        mood_decay_per_minute: 情绪的每分钟衰减幅度。
        recent_notes_limit: 最多渲染几条近期要闻。

    Returns:
        str: 渲染好的全局感知块；无内容可渲染时返回空字符串。
    """
    if max_chars <= 0:
        return ""

    # 预留包裹标签的开销
    budget = max_chars - len(BLOCK_OPEN) - len(BLOCK_CLOSE) - 2
    if budget <= 0:
        return ""

    sections: list[str] = []
    used = 0

    def try_append(text: str) -> bool:
        """在预算范围内追加一段文本。

        Args:
            text: 待追加的文本。

        Returns:
            bool: 是否成功追加。
        """
        nonlocal used
        if not text:
            return False
        cost = len(text) + 1
        if used + cost > budget:
            return False
        sections.append(text)
        used += cost
        return True

    # 优先级 1：情绪。最短，且直接影响表达风格。
    if share_mood:
        mood_text = mind.get_mood().describe(mood_decay_per_minute)
        if mood_text:
            try_append(f"你现在{mood_text}。")

    # 优先级 2：其他流的摘要。
    digests = mind.list_digests(
        exclude_stream_id=current_stream_id,
        max_streams=max_streams,
        stale_minutes=stale_minutes,
    )
    if digests:
        header_appended = False
        for digest in digests:
            line = _format_digest_line(digest)
            if not header_appended:
                if not try_append("你同时还在这些对话里："):
                    break
                header_appended = True
            if not try_append(line):
                break

    # 优先级 3：近期要闻。
    notes = mind.list_notes(limit=recent_notes_limit)
    if notes:
        header_appended = False
        for note in notes:
            if not header_appended:
                if not try_append("你最近记着这些事："):
                    break
                header_appended = True
            if not try_append(f"- {note}"):
                break

    if not sections:
        return ""

    body = "\n".join(sections)
    return f"{BLOCK_OPEN}\n{body}\n{BLOCK_CLOSE}"

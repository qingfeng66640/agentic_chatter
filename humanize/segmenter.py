"""消息分段与打字节奏模拟。

真人在聊天时不会把一大段话一次性甩出来，而是分成几条陆续发送，
中间还有打字的停顿。本模块把模型产出的整段文本切成若干条，
并计算每条之间应有的发送延迟。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 优先在这些标点处切分，它们通常是自然的语义边界
SENTENCE_ENDINGS = "。！？!?…\n"
# 次选切分点，语义边界较弱
SOFT_BREAKS = "，,、；;："

# 模型有时会把内心独白混进正文，这些前缀会被剥离
_THOUGHT_PREFIX = re.compile(
    r"^\s*(?:\[?(?:内心|心理|思考|OS|os)[:：\]]?\s*|__SUSPEND__\s*)",
)


@dataclass
class Segment:
    """一条待发送的消息分段。

    Attributes:
        text: 消息文本。
        delay: 发送本条前应等待的秒数，用于模拟打字耗时。
    """

    text: str
    delay: float = 0.0


def clean_reply_text(text: str) -> str:
    """清洗模型输出，剥离不该发出去的内容。

    Args:
        text: 模型的原始文本输出。

    Returns:
        str: 清洗后的文本；无有效内容时返回空字符串。
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = _THOUGHT_PREFIX.sub("", cleaned).strip()

    # 剥离整体包裹的引号，模型偶尔会把回复整个引起来
    if len(cleaned) >= 2 and cleaned[0] in "\"“'「" and cleaned[-1] in "\"”'」":
        cleaned = cleaned[1:-1].strip()

    return cleaned


def _split_once(text: str, limit: int) -> tuple[str, str]:
    """在长度上限附近寻找最佳切分点，切成两段。

    优先在句末标点处切分；找不到则退而求其次在弱标点处切分；
    再找不到就硬切。

    Args:
        text: 待切分文本。
        limit: 首段的目标最大长度。

    Returns:
        tuple[str, str]: (首段, 剩余部分)。
    """
    if len(text) <= limit:
        return text, ""

    window = text[: limit + 1]

    for index in range(len(window) - 1, 0, -1):
        if window[index] in SENTENCE_ENDINGS:
            return text[: index + 1].strip(), text[index + 1 :].strip()

    for index in range(len(window) - 1, 0, -1):
        if window[index] in SOFT_BREAKS:
            return text[: index + 1].strip(), text[index + 1 :].strip()

    return text[:limit].strip(), text[limit:].strip()


def segment_reply(
    text: str,
    *,
    enabled: bool = True,
    max_segment_chars: int = 60,
    max_segments: int = 4,
    typing_cps: float = 8.0,
    max_typing_delay: float = 4.0,
) -> list[Segment]:
    """将回复文本切分为若干条消息并计算发送延迟。

    Args:
        text: 模型产出的回复文本。
        enabled: 是否启用分段；关闭时整段作为单条返回。
        max_segment_chars: 单条消息的目标最大字符数。
        max_segments: 最多切分成几条；超出后剩余内容合并到最后一条。
        typing_cps: 模拟打字速度（字符/秒）；小于等于 0 时不产生延迟。
        max_typing_delay: 单条消息的最大延迟秒数。

    Returns:
        list[Segment]: 分段列表；输入为空时返回空列表。
    """
    cleaned = clean_reply_text(text)
    if not cleaned:
        return []

    if not enabled or max_segment_chars <= 0 or max_segments <= 1:
        return [Segment(text=cleaned, delay=0.0)]

    pieces: list[str] = []
    remaining = cleaned

    while remaining and len(pieces) < max_segments - 1:
        head, remaining = _split_once(remaining, max_segment_chars)
        if not head:
            break
        pieces.append(head)

    if remaining:
        pieces.append(remaining)

    if not pieces:
        pieces = [cleaned]

    segments: list[Segment] = []
    for index, piece in enumerate(pieces):
        if index == 0 or typing_cps <= 0:
            delay = 0.0
        else:
            delay = min(max(0.0, max_typing_delay), len(piece) / typing_cps)
        segments.append(Segment(text=piece, delay=delay))

    return segments

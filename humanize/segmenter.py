"""消息分段与打字节奏模拟。

真人在聊天时不会把一大段话一次性甩出来，而是分成几条陆续发送，
中间还有打字的停顿。本模块把模型产出的整段文本切成若干条，
并计算每条之间应有的发送延迟。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

# 优先在这些标点处切分，它们通常是自然的语义边界
SENTENCE_ENDINGS = "。！？!?…\n"
# 次选切分点，语义边界较弱
SOFT_BREAKS = "，,、；;："

# 模型有时会把内心独白混进正文，仅清理具备明确边界的标记块
_TAGGED_THOUGHT_BLOCK = re.compile(
    r"<\s*(?P<tag>think|analysis|reasoning)\b[^>]*>.*?</\s*(?P=tag)\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BRACKETED_THOUGHT_BLOCK = re.compile(
    r"(?:\[(?P<square>思考|分析|推理|内心|心理活动|OS)\]"
    r".*?\[/(?P=square)\]"
    r"|【(?P<corner>思考|分析|推理|内心|心理活动|OS)】"
    r".*?【/(?P=corner)】)",
    flags=re.IGNORECASE | re.DOTALL,
)
_THOUGHT_TO_REPLY = re.compile(
    r"^\s*(?:\[(?:思考|分析|推理|内心|心理活动|OS)\]"
    r"|【(?:思考|分析|推理|内心|心理活动|OS)】"
    r"|(?:思考|分析|推理)(?:过程)?\s*[:：])"
    r".*?(?:\[(?:最终)?回复\]|【(?:最终)?回复】|(?:最终)?回复\s*[:：])\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_SUSPEND_PREFIX = re.compile(r"^\s*__SUSPEND__\s*")
_FRAMEWORK_MESSAGE_LINE = re.compile(
    r"【[^】\r\n]+】"
    r"(?:<[^<>\r\n]+> )?"
    r"(?:\[[^\[\]\r\n]+\] )?"
    r"[^\r\n]+?"
    r" (?:\[[^\[\]\r\n]+\])?： ?[^\r\n]*"
)


@dataclass(frozen=True)
class CleanReplyResult:
    """回复清洗结果及被移除的明确思考内容。"""

    text: str
    removed_thoughts: tuple[str, ...] = ()


@dataclass
class Segment:
    """一条待发送的消息分段。

    Attributes:
        text: 消息文本。
        delay: 发送本条前应等待的秒数，用于模拟打字耗时。
    """

    text: str
    delay: float = 0.0


def clean_reply_text_with_metadata(text: str) -> CleanReplyResult:
    """清洗模型输出并保留被移除的明确思考内容。"""
    cleaned = str(text or "").strip()
    if not cleaned:
        return CleanReplyResult("")

    removed: list[str] = []

    def remove_block(match: re.Match[str]) -> str:
        value = match.group(0)
        inner = re.sub(r"^\s*(?:<[^>]+>|\[[^]]+\]|【[^】]+】)", "", value)
        inner = re.sub(r"(?:</[^>]+>|\[/[^]]+\]|【/[^】]+】)\s*$", "", inner)
        inner = inner.strip()
        if inner:
            removed.append(inner)
        return ""

    cleaned = _TAGGED_THOUGHT_BLOCK.sub(remove_block, cleaned)
    cleaned = _BRACKETED_THOUGHT_BLOCK.sub(remove_block, cleaned)

    thought_section = _THOUGHT_TO_REPLY.match(cleaned)
    if thought_section is not None:
        value = thought_section.group(0)
        inner = re.sub(
            r"^\s*(?:\[(?:思考|分析|推理|内心|心理活动|OS)\]"
            r"|【(?:思考|分析|推理|内心|心理活动|OS)】"
            r"|(?:思考|分析|推理)(?:过程)?\s*[:：])",
            "",
            value,
            flags=re.IGNORECASE,
        )
        inner = re.sub(
            r"(?:\[(?:最终)?回复\]|【(?:最终)?回复】|(?:最终)?回复\s*[:：])\s*$",
            "",
            inner,
            flags=re.IGNORECASE,
        ).strip()
        if inner:
            removed.append(inner)
        cleaned = cleaned[thought_section.end() :]

    cleaned = _SUSPEND_PREFIX.sub("", cleaned).strip()

    if len(cleaned) >= 2 and cleaned[0] in "\"“'「" and cleaned[-1] in "\"”'」":
        cleaned = cleaned[1:-1].strip()

    return CleanReplyResult(cleaned, tuple(removed))


def clean_reply_text(text: str) -> str:
    """清洗模型输出，剥离不该发出去的内容。"""
    return clean_reply_text_with_metadata(text).text


def is_framework_message_line(text: str) -> bool:
    """判断文本是否完整复述了一条框架格式化消息行。"""
    return _FRAMEWORK_MESSAGE_LINE.fullmatch(str(text or "").strip()) is not None


def detect_provider_error_text(text: str) -> str | None:
    """识别被上游错误包装为正常正文的明显供应商异常。"""
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return None

    submission_blocked = "the prompt could not be submitted" in normalized
    sensitive_prompt = (
        "prompt contains sensitive words" in normalized
        or "prompt contains sensitive content" in normalized
    )
    google_policy = (
        "generative ai prohibited use policy" in normalized
        or "policies.google.com/terms/generative-ai/use-policy" in normalized
    )
    gemini_guidance = (
        "try rephrasing the prompt" in normalized
        or "ai.google.dev/gemini-api/docs/troubleshooting" in normalized
    )
    if submission_blocked and sensitive_prompt and (google_policy or gemini_guidance):
        return "google_prompt_policy_block"
    return None


def detect_reply_decision_json_text(text: str) -> str | None:
    """识别被错误作为可见回复输出的完整子决策 JSON。"""
    candidate = str(text or "").strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[7:-3].strip()
    elif not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    action = payload.get("action")
    if action not in {"respond", "silent"}:
        return None
    confidence = payload.get("confidence")
    interrupt_cost = payload.get("interrupt_cost")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        return None
    if (
        isinstance(interrupt_cost, bool)
        or not isinstance(interrupt_cost, (int, float))
        or not math.isfinite(interrupt_cost)
        or not 0.0 <= interrupt_cost <= 1.0
    ):
        return None
    if not isinstance(payload.get("addressee"), str):
        return None
    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list) or not all(
        isinstance(value, str) for value in reason_codes
    ):
        return None
    if not isinstance(payload.get("brief_reason"), str):
        return None
    return action


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

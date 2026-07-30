"""本地回复决策信号提取。"""

from __future__ import annotations

from collections.abc import Mapping
import re
import time
from typing import Any

from .models import DecisionAction, DecisionFeatures, DecisionSource, ReplyDecision
from .state import ParticipationState

_QUESTION_RE = re.compile(r"[?？]|(?:吗|么|呢|怎么|为什么|为何|能否|可以|请问|帮我|麻烦|求助)")
_CLOSURE_RE = re.compile(r"^(?:好(?:的|吧|啦)?|收到|知道了|明白了|行|嗯+|晚安|先这样|下次聊|没事了)[。！!~～…]*$")
_LOW_INFO_RE = re.compile(r"^(?:[哈呵嘿嗯哦啊诶]+|[\W_]+|[1６6]+|ok|OK|hh+)$")


def message_text(message: Any) -> str:
    """提取消息的纯文本内容。"""
    return str(
        getattr(message, "processed_plain_text", None)
        or getattr(message, "content", "")
        or ""
    ).strip()


def _mentioned_ids(message: Any) -> set[str]:
    """从消息扩展字段提取被提及账号。"""
    extra = getattr(message, "extra", {})
    if not isinstance(extra, dict):
        return set()
    values = extra.get("at_users") or extra.get("mentioned_user_ids") or []
    if isinstance(values, (str, int)):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return set()
    mentioned: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            user_id = value.get("user_id")
            if user_id is not None and str(user_id):
                mentioned.add(str(user_id))
        elif value is not None and str(value):
            mentioned.add(str(value))
    return mentioned


def _bot_was_mentioned(message: Any, bot_id: str) -> bool:
    """判断平台或兼容字段是否确认当前 Bot 被 @。"""
    extra = getattr(message, "extra", {})
    if isinstance(extra, dict) and extra.get("bot_was_mentioned") is True:
        return True
    mentioned_ids = _mentioned_ids(message)
    return bool(bot_id and bot_id in mentioned_ids)


def hard_rule_decision(
    *,
    is_private: bool,
    bot_id: str,
    bot_nickname: str,
    unread_messages: list[Any],
    bot_message_ids: set[str],
) -> ReplyDecision | None:
    """执行少量高可靠硬规则。"""
    texts = [message_text(message) for message in unread_messages]
    valid_texts = [text for text in texts if text]
    if not valid_texts:
        return ReplyDecision(
            DecisionAction.SILENT,
            DecisionSource.HARD_RULE,
            reasons=["empty_messages"],
        )

    if is_private:
        return ReplyDecision(
            DecisionAction.RESPOND,
            DecisionSource.HARD_RULE,
            reasons=["private_chat"],
        )

    for message, text in zip(unread_messages, texts, strict=False):
        reply_to = str(getattr(message, "reply_to", "") or "")
        if reply_to and reply_to in bot_message_ids:
            return ReplyDecision(
                DecisionAction.RESPOND,
                DecisionSource.HARD_RULE,
                reasons=["reply_to_bot"],
            )
        if _bot_was_mentioned(message, bot_id):
            return ReplyDecision(
                DecisionAction.RESPOND,
                DecisionSource.HARD_RULE,
                reasons=["mention_bot"],
            )
        if bot_nickname and (
            text.startswith(bot_nickname)
            or f"@{bot_nickname}" in text
            or f"＠{bot_nickname}" in text
        ):
            return ReplyDecision(
                DecisionAction.RESPOND,
                DecisionSource.HARD_RULE,
                reasons=["nickname_address"],
            )
    return None


def extract_features(
    *,
    unread_messages: list[Any],
    bot_id: str,
    bot_nickname: str,
    bot_message_ids: set[str],
    participation: ParticipationState,
    semantic_continuity: float | None,
    bot_history_continuity: float | None,
    participation_window_seconds: float,
    rhythm_cooldown_seconds: float,
    now: float | None = None,
) -> DecisionFeatures:
    """从消息结构、文本和参与状态提取可解释信号。"""
    current = time.time() if now is None else now
    texts = [message_text(message) for message in unread_messages]
    joined = "\n".join(texts)
    features = DecisionFeatures()

    if bot_nickname and any(
        text.startswith(bot_nickname)
        or f"@{bot_nickname}" in text
        or f"＠{bot_nickname}" in text
        for text in texts
    ):
        features.direct_address = 1.0
        features.reasons.append("nickname_address")

    if _QUESTION_RE.search(joined):
        features.question_or_request = 1.0
        features.reasons.append("question_or_request")

    last_text = texts[-1] if texts else ""
    if last_text and _CLOSURE_RE.fullmatch(last_text):
        features.topic_closure = 1.0
        features.reasons.append("topic_closure")
    if last_text and (len(last_text) <= 4 or _LOW_INFO_RE.fullmatch(last_text)):
        features.low_information = 1.0
        features.reasons.append("low_information")

    senders = [
        str(getattr(message, "sender_id", "") or "")
        for message in unread_messages
        if getattr(message, "sender_id", "")
    ]
    distinct_senders = len(set(senders))
    if distinct_senders >= 3:
        features.interruption_cost = 1.0
        features.reasons.append("multi_party_fast_flow")
    elif distinct_senders == 2:
        features.interruption_cost = 0.5

    directed_other = False
    for message in unread_messages:
        mentioned = _mentioned_ids(message)
        reply_to = str(getattr(message, "reply_to", "") or "")
        if mentioned and not _bot_was_mentioned(message, bot_id):
            directed_other = True
        if reply_to and reply_to not in bot_message_ids:
            directed_other = True
    if directed_other:
        features.directed_elsewhere = 1.0
        features.reasons.append("directed_elsewhere")

    if semantic_continuity is None:
        features.uncertainty += 0.18
        features.reasons.append("semantic_unknown")
    else:
        features.semantic_continuity = max(0.0, min(1.0, semantic_continuity))
    if bot_history_continuity is not None:
        features.bot_history_continuity = max(
            0.0, min(1.0, bot_history_continuity)
        )

    since_reply = (
        current - participation.last_reply_at
        if participation.last_reply_at > 0
        else float("inf")
    )
    if since_reply <= max(1.0, participation_window_seconds):
        decay = 1.0 - since_reply / max(1.0, participation_window_seconds)
        features.participation_momentum = max(0.0, decay)
    if since_reply <= max(0.0, rhythm_cooldown_seconds):
        features.rhythm_cooldown = 1.0
        features.reasons.append("rhythm_cooldown")

    features.silence_momentum = min(1.0, participation.consecutive_silence / 5.0)
    if len(last_text) < 8:
        features.uncertainty += 0.12
    if directed_other and features.question_or_request:
        features.uncertainty += 0.12
        features.reasons.append("conflicting_signals")
    features.contribution_value = min(1.0, len(joined) / 120.0)
    return features

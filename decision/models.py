"""回复决策领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DecisionAction(StrEnum):
    """回复决策动作。"""

    RESPOND = "respond"
    SILENT = "silent"


class DecisionSource(StrEnum):
    """回复决策来源。"""

    DISABLED = "disabled"
    HARD_RULE = "hard_rule"
    LOCAL = "local"
    SUB_ACTOR = "sub_actor"
    FALLBACK = "fallback"


@dataclass(slots=True)
class DecisionFeatures:
    """本轮本地决策信号。"""

    direct_address: float = 0.0
    semantic_continuity: float = 0.0
    bot_history_continuity: float = 0.0
    participation_momentum: float = 0.0
    question_or_request: float = 0.0
    contribution_value: float = 0.0
    directed_elsewhere: float = 0.0
    interruption_cost: float = 0.0
    topic_closure: float = 0.0
    rhythm_cooldown: float = 0.0
    silence_momentum: float = 0.0
    low_information: float = 0.0
    uncertainty: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReplyDecision:
    """单轮是否回复的完整决策。"""

    action: DecisionAction
    source: DecisionSource
    score: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    sub_actor_used: bool = False

    @property
    def should_respond(self) -> bool:
        """返回是否应进入回复流程。"""
        return self.action == DecisionAction.RESPOND


_ACTION_TEXT = {
    DecisionAction.RESPOND: "回复",
    DecisionAction.SILENT: "暂不回复",
}
_SOURCE_TEXT = {
    DecisionSource.DISABLED: "已关闭判断",
    DecisionSource.HARD_RULE: "直接唤醒",
    DecisionSource.LOCAL: "本地判断",
    DecisionSource.SUB_ACTOR: "子决策模型",
    DecisionSource.FALLBACK: "故障回退",
}
_REASON_TEXT = {
    "decision_disabled": "已关闭自然参与判断",
    "empty_messages": "没有可处理的消息",
    "private_chat": "私聊消息",
    "reply_to_bot": "回复了我的消息",
    "mention_bot": "@了我",
    "nickname_address": "称呼了我",
    "distracted": "开启了偶尔走神",
    "question_or_request": "包含问题或请求",
    "directed_elsewhere": "消息主要面向其他成员",
    "multi_party_fast_flow": "多人连续对话中",
    "topic_closure": "话题已经收尾",
    "rhythm_cooldown": "我刚回复过，暂缓插话",
    "low_information": "消息信息较少",
    "semantic_unknown": "语义相关性暂不可用",
    "conflicting_signals": "对话信号存在冲突",
    "recent_reply_fallback_suppressed": "刚回复过，避免故障回退时重复插话",
    "sub_actor_fallback": "子决策模型不可用",
    "closure_low_information": "话题收尾且没有新信息",
    "directed_elsewhere_unrelated": "消息面向其他成员且与我无关",
    "fast_flow_no_clear_entry": "多人快速对话中没有明确介入入口",
    "cooldown_low_value": "刚回复过且当前消息价值较低",
    "contextual_question_for_bot": "问题与当前话题或我的发言连续",
    "contextual_nickname_address": "明确称呼了我，且当前没有明显冲突信号",
    "strong_contextual_followup": "高连续性且内容具有可贡献信息",
    "local_score_reply": "本地评分达到回复边界",
    "local_score_silent": "本地评分达到静默边界",
}


def describe_decision(decision: ReplyDecision) -> tuple[str, str, str]:
    """将内部决策枚举与原因转换为中文运行日志文本。"""
    action = _ACTION_TEXT.get(decision.action, "未知")
    source = _SOURCE_TEXT.get(decision.source, "未知方式")
    reasons = [
        _REASON_TEXT.get(reason, "其他判断信号")
        for reason in decision.reasons[-2:]
    ]
    return action, source, "、".join(reasons) or "未提供原因"

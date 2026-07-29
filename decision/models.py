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

"""本地回复倾向与置信区间计算。"""

from __future__ import annotations

import random
from typing import Any

from .models import DecisionAction, DecisionFeatures, DecisionSource, ReplyDecision


def _clamp(value: float) -> float:
    """将数值限制在 [-1, 1]。"""
    return max(-1.0, min(1.0, value))


def _weights(config: Any) -> dict[str, float]:
    """构建本地线性评分权重。"""
    return {
        "direct_address": float(config.weight_direct_address),
        "semantic_continuity": float(config.weight_semantic_continuity),
        "bot_history_continuity": float(config.weight_bot_history_continuity),
        "participation_momentum": float(config.weight_participation_momentum),
        "question_or_request": float(config.weight_question_or_request),
        "contribution_value": float(config.weight_contribution_value),
        "directed_elsewhere": -abs(float(config.weight_directed_elsewhere)),
        "interruption_cost": -abs(float(config.weight_interruption_cost)),
        "topic_closure": -abs(float(config.weight_topic_closure)),
        "rhythm_cooldown": -abs(float(config.weight_rhythm_cooldown)),
        "silence_momentum": -abs(float(config.weight_silence_momentum)),
        "low_information": -abs(float(config.weight_low_information)),
    }


def _score_and_interval(
    features: DecisionFeatures,
    config: Any,
    *,
    rng: random.Random | None = None,
) -> tuple[float, float, float, float]:
    """计算分数、区间与置信度。"""
    score = sum(
        float(getattr(features, name)) * weight
        for name, weight in _weights(config).items()
    )
    randomness = max(0.0, min(0.1, float(config.gray_zone_randomness)))
    if randomness:
        source = rng or random.Random()
        score += source.uniform(-randomness, randomness)
    score = _clamp(score)
    uncertainty = max(
        0.0,
        min(0.75, float(config.base_uncertainty) + float(features.uncertainty)),
    )
    lower = _clamp(score - uncertainty)
    upper = _clamp(score + uncertainty)
    confidence = max(0.0, min(1.0, 1.0 - uncertainty))
    return score, lower, upper, confidence


def _local_decision(
    action: DecisionAction,
    reason: str,
    features: DecisionFeatures,
    score: float,
    lower: float,
    upper: float,
    confidence: float,
) -> ReplyDecision:
    """构造带组合规则原因的本地决策。"""
    return ReplyDecision(
        action,
        DecisionSource.LOCAL,
        score=score,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
        reasons=[reason, *features.reasons],
    )


def _combined_decision(
    features: DecisionFeatures,
    score: float,
    lower: float,
    upper: float,
    confidence: float,
) -> ReplyDecision | None:
    """处理明确可解释的本地回复或静默组合。"""
    no_bot_entry = (
        features.directed_elsewhere <= 0.0
        and features.topic_closure <= 0.0
        and features.rhythm_cooldown <= 0.0
        and features.interruption_cost < 1.0
    )

    if features.topic_closure >= 1.0 and features.low_information >= 1.0:
        return _local_decision(
            DecisionAction.SILENT,
            "closure_low_information",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    if (
        features.directed_elsewhere >= 1.0
        and features.semantic_continuity < 0.55
        and features.bot_history_continuity < 0.55
    ):
        return _local_decision(
            DecisionAction.SILENT,
            "directed_elsewhere_unrelated",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    if features.interruption_cost >= 1.0 and (
        features.low_information >= 1.0
        or features.topic_closure >= 1.0
        or (
            features.semantic_continuity < 0.40
            and features.bot_history_continuity < 0.40
        )
    ):
        return _local_decision(
            DecisionAction.SILENT,
            "fast_flow_no_clear_entry",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    if features.rhythm_cooldown >= 1.0 and (
        features.low_information >= 1.0
        or features.topic_closure >= 1.0
        or (
            features.contribution_value < 0.20
            and features.question_or_request <= 0.0
        )
    ):
        return _local_decision(
            DecisionAction.SILENT,
            "cooldown_low_value",
            features,
            score,
            lower,
            upper,
            confidence,
        )

    if features.direct_address >= 1.0 and no_bot_entry:
        return _local_decision(
            DecisionAction.RESPOND,
            "contextual_nickname_address",
            features,
            score,
            lower,
            upper,
            confidence,
        )

    if (
        features.question_or_request >= 1.0
        and (
            features.semantic_continuity >= 0.60
            or features.bot_history_continuity >= 0.65
        )
        and no_bot_entry
    ):
        return _local_decision(
            DecisionAction.RESPOND,
            "contextual_question_for_bot",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    if (
        features.semantic_continuity >= 0.75
        and features.bot_history_continuity >= 0.55
        and features.contribution_value >= 0.25
        and no_bot_entry
    ):
        return _local_decision(
            DecisionAction.RESPOND,
            "strong_contextual_followup",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    return None


def score_features(
    features: DecisionFeatures,
    config: Any,
    *,
    rng: random.Random | None = None,
) -> ReplyDecision | None:
    """计算本地分数，明确组合与区间命中时直接决策。"""
    score, lower, upper, confidence = _score_and_interval(features, config, rng=rng)
    combined = _combined_decision(features, score, lower, upper, confidence)
    if combined is not None:
        return combined
    if lower >= float(config.local_reply_lower_bound):
        return _local_decision(
            DecisionAction.RESPOND,
            "local_score_reply",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    if upper <= float(config.local_silent_upper_bound):
        return _local_decision(
            DecisionAction.SILENT,
            "local_score_silent",
            features,
            score,
            lower,
            upper,
            confidence,
        )
    return None


def interval_summary(features: DecisionFeatures, config: Any) -> tuple[float, float, float]:
    """计算无随机扰动的分数与区间，供子决策模型参考。"""
    class _NoRandom:
        def uniform(self, _low: float, _high: float) -> float:
            """消除随机扰动。"""
            return 0.0

    score, lower, upper, _ = _score_and_interval(features, config, rng=_NoRandom())
    return score, lower, upper

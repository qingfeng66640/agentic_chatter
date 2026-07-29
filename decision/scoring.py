"""本地回复倾向与置信区间计算。"""

from __future__ import annotations

import random
from typing import Any

from .models import DecisionAction, DecisionFeatures, DecisionSource, ReplyDecision


def _clamp(value: float) -> float:
    """将数值限制在 [-1, 1]。"""
    return max(-1.0, min(1.0, value))


def score_features(
    features: DecisionFeatures,
    config: Any,
    *,
    rng: random.Random | None = None,
) -> ReplyDecision | None:
    """计算本地分数；灰区返回 None。"""
    weights = {
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
    score = sum(
        float(getattr(features, name)) * weight for name, weight in weights.items()
    )
    randomness = max(0.0, min(0.1, float(config.gray_zone_randomness)))
    if randomness:
        source = rng or random.Random()
        score += source.uniform(-randomness, randomness)
    score = _clamp(score)

    uncertainty = max(
        0.0,
        min(
            0.75,
            float(config.base_uncertainty) + float(features.uncertainty),
        ),
    )
    lower = _clamp(score - uncertainty)
    upper = _clamp(score + uncertainty)
    confidence = max(0.0, min(1.0, 1.0 - uncertainty))

    if lower >= float(config.local_reply_lower_bound):
        return ReplyDecision(
            DecisionAction.RESPOND,
            DecisionSource.LOCAL,
            score=score,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence,
            reasons=list(features.reasons),
        )
    if upper <= float(config.local_silent_upper_bound):
        return ReplyDecision(
            DecisionAction.SILENT,
            DecisionSource.LOCAL,
            score=score,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence,
            reasons=list(features.reasons),
        )
    return None


def interval_summary(features: DecisionFeatures, config: Any) -> tuple[float, float, float]:
    """计算无随机扰动的分数与区间，供 sub_actor 参考。"""
    class _NoRandom:
        def uniform(self, _low: float, _high: float) -> float:
            return 0.0

    decision = score_features(features, config, rng=_NoRandom())
    if decision is not None:
        return decision.score, decision.lower_bound, decision.upper_bound

    weights = {
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
    score = _clamp(
        sum(float(getattr(features, name)) * weight for name, weight in weights.items())
    )
    uncertainty = min(
        0.75,
        max(0.0, float(config.base_uncertainty) + float(features.uncertainty)),
    )
    return score, _clamp(score - uncertainty), _clamp(score + uncertainty)

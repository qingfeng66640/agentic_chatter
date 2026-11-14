"""情绪推断与衰减测试。"""

from __future__ import annotations

import time

from ..global_mind.store import MoodState
from ..humanize.mood import (
    describe_mood_for_prompt,
    infer_mood_delta,
)


def test_positive_signals_raise_mood() -> None:
    """积极情绪词应产生正向增量并提供非空原因。"""
    delta, reason = infer_mood_delta("谢谢你，太好了，哈哈哈")
    assert delta > 0
    assert reason


def test_negative_signals_lower_mood() -> None:
    """消极情绪词应产生负向增量并提供非空原因。"""
    delta, reason = infer_mood_delta("好烦，闭嘴，讨厌")
    assert delta < 0
    assert reason


def test_neutral_text_produces_no_change() -> None:
    """中性文本不应改变情绪，也不应生成变化原因。"""
    delta, reason = infer_mood_delta("今天几点开会")
    assert delta == 0.0
    assert reason == ""


def test_balanced_signals_cancel_out() -> None:
    """数量相当的正负情绪信号应相互抵消为零增量。"""
    delta, _ = infer_mood_delta("谢谢，但是好烦")
    assert delta == 0.0


def test_empty_text_produces_no_change() -> None:
    """空文本与纯空白文本都应返回零增量和空原因。"""
    assert infer_mood_delta("") == (0.0, "")
    assert infer_mood_delta("   ") == (0.0, "")


def test_delta_is_clamped() -> None:
    """大量重复的积极词语不应使单次情绪增量超过 0.35。"""
    delta, _ = infer_mood_delta("谢谢 " * 100)
    assert delta <= 0.35


def test_mood_decays_toward_baseline() -> None:
    """情绪必须随时间回落，否则一次不快会永久影响 bot。"""
    past = time.time() - 600
    mood = MoodState(value=0.8, updated_at=past)

    decayed = mood.decayed(decay_per_minute=0.05)
    assert decayed < 0.8
    assert decayed >= 0.0


def test_negative_mood_decays_upward() -> None:
    """负面情绪应随时间向零回升，但不越过中性基线。"""
    past = time.time() - 600
    mood = MoodState(value=-0.8, updated_at=past)

    decayed = mood.decayed(decay_per_minute=0.05)
    assert decayed > -0.8
    assert decayed <= 0.0


def test_decay_does_not_cross_baseline() -> None:
    """足够长时间后的正负情绪都应衰减到零，而非越过基线。"""
    past = time.time() - 100000
    assert MoodState(value=0.5, updated_at=past).decayed(1.0) == 0.0
    assert MoodState(value=-0.5, updated_at=past).decayed(1.0) == 0.0


def test_describe_returns_empty_near_baseline() -> None:
    """情绪处于零值或接近中性时，不应生成提示词描述。"""
    assert describe_mood_for_prompt(0.0) == ""
    assert describe_mood_for_prompt(0.1) == ""


def test_describe_reflects_strong_moods() -> None:
    """明显的正负情绪应分别体现轻快和烦躁的描述。"""
    assert "轻快" in describe_mood_for_prompt(0.8)
    assert "烦躁" in describe_mood_for_prompt(-0.8)


def test_mood_state_describe_includes_label() -> None:
    """情绪状态描述应同时包含强度措辞和最近情绪原因。"""
    mood = MoodState(value=0.7, label="刚才聊得挺愉快")
    text = mood.describe(decay_per_minute=0.0)
    assert "心情很好" in text
    assert "愉快" in text

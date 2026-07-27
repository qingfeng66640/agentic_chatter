"""情绪推断与衰减测试。"""

from __future__ import annotations

import time

from ..global_mind.store import MoodState
from ..humanize.mood import (
    describe_mood_for_prompt,
    infer_mood_delta,
)


def test_positive_signals_raise_mood() -> None:
    delta, reason = infer_mood_delta("谢谢你，太好了，哈哈哈")
    assert delta > 0
    assert reason


def test_negative_signals_lower_mood() -> None:
    delta, reason = infer_mood_delta("好烦，闭嘴，讨厌")
    assert delta < 0
    assert reason


def test_neutral_text_produces_no_change() -> None:
    delta, reason = infer_mood_delta("今天几点开会")
    assert delta == 0.0
    assert reason == ""


def test_balanced_signals_cancel_out() -> None:
    delta, _ = infer_mood_delta("谢谢，但是好烦")
    assert delta == 0.0


def test_empty_text_produces_no_change() -> None:
    assert infer_mood_delta("") == (0.0, "")
    assert infer_mood_delta("   ") == (0.0, "")


def test_delta_is_clamped() -> None:
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
    past = time.time() - 600
    mood = MoodState(value=-0.8, updated_at=past)

    decayed = mood.decayed(decay_per_minute=0.05)
    assert decayed > -0.8
    assert decayed <= 0.0


def test_decay_does_not_cross_baseline() -> None:
    past = time.time() - 100000
    assert MoodState(value=0.5, updated_at=past).decayed(1.0) == 0.0
    assert MoodState(value=-0.5, updated_at=past).decayed(1.0) == 0.0


def test_describe_returns_empty_near_baseline() -> None:
    assert describe_mood_for_prompt(0.0) == ""
    assert describe_mood_for_prompt(0.1) == ""


def test_describe_reflects_strong_moods() -> None:
    assert "轻快" in describe_mood_for_prompt(0.8)
    assert "烦躁" in describe_mood_for_prompt(-0.8)


def test_mood_state_describe_includes_label() -> None:
    mood = MoodState(value=0.7, label="刚才聊得挺愉快")
    text = mood.describe(decay_per_minute=0.0)
    assert "心情很好" in text
    assert "愉快" in text

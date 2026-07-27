"""跨流全局心智测试。

守护「不同群聊人格割裂」的修复，同时确保跨流信息不会膨胀到
稀释模型对当前对话的注意力。
"""

from __future__ import annotations

import time

from ..global_mind.render import render_global_awareness
from ..global_mind.store import GlobalMind


def test_mood_is_shared_across_streams(fresh_mind: GlobalMind) -> None:
    """在 A 流产生的情绪，在 B 流也应当可见。"""
    fresh_mind.nudge_mood(0.5, "在A群聊得开心")
    assert fresh_mind.get_mood().value > 0


def test_stream_moods_are_isolated_when_not_shared(fresh_mind: GlobalMind) -> None:
    fresh_mind.nudge_stream_mood("stream_a", 0.8, "A群很开心")

    assert fresh_mind.get_stream_mood("stream_a").value > 0
    assert fresh_mind.get_stream_mood("stream_b").value == 0.0
    assert fresh_mind.get_mood().value == 0.0


def test_mood_is_clamped(fresh_mind: GlobalMind) -> None:
    for _ in range(50):
        fresh_mind.nudge_mood(0.5, decay_per_minute=0.0)
    assert fresh_mind.get_mood().value <= 1.0

    for _ in range(100):
        fresh_mind.nudge_mood(-0.5, decay_per_minute=0.0)
    assert fresh_mind.get_mood().value >= -1.0


def test_reset_mood_returns_to_baseline(fresh_mind: GlobalMind) -> None:
    fresh_mind.nudge_mood(0.8)
    fresh_mind.reset_mood()
    assert fresh_mind.get_mood().value == 0.0


def test_digest_excludes_current_stream(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("stream_a", stream_name="A群", topic="聊游戏")
    fresh_mind.update_digest("stream_b", stream_name="B群", topic="聊工作")

    digests = fresh_mind.list_digests(exclude_stream_id="stream_a")
    assert [item.stream_id for item in digests] == ["stream_b"]


def test_digest_filters_stale_streams(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("old", stream_name="旧群", topic="旧话题")
    fresh_mind._digests["old"].updated_at = time.time() - 3600 * 5

    assert fresh_mind.list_digests(stale_minutes=120.0) == []


def test_digest_sorted_by_recency(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("first", stream_name="先")
    time.sleep(0.01)
    fresh_mind.update_digest("second", stream_name="后")

    digests = fresh_mind.list_digests()
    assert digests[0].stream_id == "second"


def test_digest_respects_max_streams(fresh_mind: GlobalMind) -> None:
    for index in range(20):
        fresh_mind.update_digest(f"s{index}", stream_name=f"群{index}")

    assert len(fresh_mind.list_digests(max_streams=3)) == 3


def test_digest_topic_is_truncated(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("s", topic="话" * 200)
    assert len(fresh_mind.list_digests()[0].topic) <= 41


def test_empty_stream_id_is_ignored(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("", stream_name="无效")
    assert fresh_mind.list_digests() == []


def test_notes_deduplicate(fresh_mind: GlobalMind) -> None:
    fresh_mind.add_note("答应了小明看代码")
    fresh_mind.add_note("答应了小明看代码")
    assert len(fresh_mind.list_notes()) == 1


def test_notes_respect_limit(fresh_mind: GlobalMind) -> None:
    for index in range(20):
        fresh_mind.add_note(f"要闻{index}", limit=5)
    assert len(fresh_mind.list_notes(limit=99)) <= 5


def test_render_respects_max_chars(fresh_mind: GlobalMind) -> None:
    """这是防止跨流信息稀释注意力的关键约束。"""
    fresh_mind.nudge_mood(0.9, "很开心")
    for index in range(30):
        fresh_mind.update_digest(
            f"s{index}", stream_name=f"很长的群名称{index}", topic="聊了很多东西"
        )
    for index in range(30):
        fresh_mind.add_note(f"这是一条比较长的近期要闻记录 {index}", limit=30)

    text = render_global_awareness(
        fresh_mind, current_stream_id="none", max_chars=300, max_streams=20
    )

    assert len(text) <= 300


def test_render_returns_empty_when_nothing_to_show(fresh_mind: GlobalMind) -> None:
    text = render_global_awareness(fresh_mind, current_stream_id="s")
    assert text == ""


def test_render_returns_empty_with_zero_budget(fresh_mind: GlobalMind) -> None:
    fresh_mind.nudge_mood(0.9)
    assert render_global_awareness(fresh_mind, current_stream_id="s", max_chars=0) == ""


def test_render_includes_other_streams(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("other", stream_name="隔壁群", topic="聊周末爬山")

    text = render_global_awareness(fresh_mind, current_stream_id="current")
    assert "隔壁群" in text
    assert "爬山" in text


def test_render_can_disable_mood(fresh_mind: GlobalMind) -> None:
    fresh_mind.nudge_mood(0.9, "开心")

    text = render_global_awareness(
        fresh_mind, current_stream_id="s", share_mood=False
    )
    assert "心情" not in text


def test_drop_digest(fresh_mind: GlobalMind) -> None:
    fresh_mind.update_digest("s", stream_name="群")
    assert fresh_mind.drop_digest("s")
    assert not fresh_mind.drop_digest("s")

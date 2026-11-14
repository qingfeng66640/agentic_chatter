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
    """独立流情绪更新不应影响其他聊天流或全局情绪。"""
    fresh_mind.nudge_stream_mood("stream_a", 0.8, "A群很开心")

    assert fresh_mind.get_stream_mood("stream_a").value > 0
    assert fresh_mind.get_stream_mood("stream_b").value == 0.0
    assert fresh_mind.get_mood().value == 0.0


def test_mood_is_clamped(fresh_mind: GlobalMind) -> None:
    """反复施加正负情绪增量后，全局情绪值仍应处于上下限内。"""
    for _ in range(50):
        fresh_mind.nudge_mood(0.5, decay_per_minute=0.0)
    assert fresh_mind.get_mood().value <= 1.0

    for _ in range(100):
        fresh_mind.nudge_mood(-0.5, decay_per_minute=0.0)
    assert fresh_mind.get_mood().value >= -1.0


def test_reset_mood_returns_to_baseline(fresh_mind: GlobalMind) -> None:
    """重置全局情绪后，情绪值应回到零基线。"""
    fresh_mind.nudge_mood(0.8)
    fresh_mind.reset_mood()
    assert fresh_mind.get_mood().value == 0.0


def test_digest_excludes_current_stream(fresh_mind: GlobalMind) -> None:
    """列举跨流摘要时应排除指定的当前聊天流。"""
    fresh_mind.update_digest("stream_a", stream_name="A群", topic="聊游戏")
    fresh_mind.update_digest("stream_b", stream_name="B群", topic="聊工作")

    digests = fresh_mind.list_digests(exclude_stream_id="stream_a")
    assert [item.stream_id for item in digests] == ["stream_b"]


def test_digest_filters_stale_streams(fresh_mind: GlobalMind) -> None:
    """更新时间超过过期阈值的聊天流摘要不应出现在结果中。"""
    fresh_mind.update_digest("old", stream_name="旧群", topic="旧话题")
    fresh_mind._digests["old"].updated_at = time.time() - 3600 * 5

    assert fresh_mind.list_digests(stale_minutes=120.0) == []


def test_digest_sorted_by_recency(fresh_mind: GlobalMind) -> None:
    """跨流摘要应优先返回最近更新的聊天流。"""
    fresh_mind.update_digest("first", stream_name="先")
    time.sleep(0.01)
    fresh_mind.update_digest("second", stream_name="后")

    digests = fresh_mind.list_digests()
    assert digests[0].stream_id == "second"


def test_digest_respects_max_streams(fresh_mind: GlobalMind) -> None:
    """跨流摘要数量应受调用方指定的聊天流上限约束。"""
    for index in range(20):
        fresh_mind.update_digest(f"s{index}", stream_name=f"群{index}")

    assert len(fresh_mind.list_digests(max_streams=3)) == 3


def test_digest_topic_is_truncated(fresh_mind: GlobalMind) -> None:
    """超长话题应截断为不超过 41 字的摘要。"""
    fresh_mind.update_digest("s", topic="话" * 200)
    assert len(fresh_mind.list_digests()[0].topic) <= 41


def test_empty_stream_id_is_ignored(fresh_mind: GlobalMind) -> None:
    """空聊天流标识不应产生可列举的摘要。"""
    fresh_mind.update_digest("", stream_name="无效")
    assert fresh_mind.list_digests() == []


def test_notes_deduplicate(fresh_mind: GlobalMind) -> None:
    """重复添加相同要闻后，应只保留一条记录。"""
    fresh_mind.add_note("答应了小明看代码")
    fresh_mind.add_note("答应了小明看代码")
    assert len(fresh_mind.list_notes()) == 1


def test_notes_respect_limit(fresh_mind: GlobalMind) -> None:
    """持续添加要闻时，实际保存的记录数不应超过写入上限。"""
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
    """没有情绪、摘要或要闻可展示时，全局感知渲染应返回空文本。"""
    text = render_global_awareness(fresh_mind, current_stream_id="s")
    assert text == ""


def test_render_returns_empty_with_zero_budget(fresh_mind: GlobalMind) -> None:
    """字符预算为零时，即使存在明显情绪也应返回空文本。"""
    fresh_mind.nudge_mood(0.9)
    assert render_global_awareness(fresh_mind, current_stream_id="s", max_chars=0) == ""


def test_render_includes_other_streams(fresh_mind: GlobalMind) -> None:
    """全局感知文本应包含其他聊天流的名称和话题。"""
    fresh_mind.update_digest("other", stream_name="隔壁群", topic="聊周末爬山")

    text = render_global_awareness(fresh_mind, current_stream_id="current")
    assert "隔壁群" in text
    assert "爬山" in text


def test_render_can_disable_mood(fresh_mind: GlobalMind) -> None:
    """关闭情绪共享后，全局感知文本不应包含心情描述。"""
    fresh_mind.nudge_mood(0.9, "开心")

    text = render_global_awareness(fresh_mind, current_stream_id="s", share_mood=False)
    assert "心情" not in text


def test_drop_digest(fresh_mind: GlobalMind) -> None:
    """首次删除已有摘要应成功，再次删除同一摘要应返回失败。"""
    fresh_mind.update_digest("s", stream_name="群")
    assert fresh_mind.drop_digest("s")
    assert not fresh_mind.drop_digest("s")

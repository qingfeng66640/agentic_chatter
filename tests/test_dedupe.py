"""软去重测试。

守护本插件对 DFC 硬去重的修复：重复调用不应被粗暴拒绝，
否则会打击模型本就不高的工具调用意愿。
"""

from __future__ import annotations

from ..tooling.dedupe import CallDeduper, build_call_key


def test_build_call_key_ignores_reason_field() -> None:
    """reason 只是模型的自述理由，不应影响去重语义。"""
    first = build_call_key("tool-search", {"q": "天气", "reason": "想查天气"})
    second = build_call_key("tool-search", {"q": "天气", "reason": "完全不同的理由"})
    assert first == second


def test_build_call_key_distinguishes_different_args() -> None:
    first = build_call_key("tool-search", {"q": "天气"})
    second = build_call_key("tool-search", {"q": "股价"})
    assert first != second


def test_build_call_key_handles_unserializable_args() -> None:
    key = build_call_key("tool-x", {"obj": object()})
    assert key.startswith("tool-x:")


def test_off_mode_never_blocks() -> None:
    deduper = CallDeduper(mode="off")
    for _ in range(10):
        assert deduper.check("tool-a", {"x": 1}).allow


def test_hard_mode_blocks_second_call() -> None:
    deduper = CallDeduper(mode="hard")
    assert deduper.check("tool-a", {"x": 1}).allow

    decision = deduper.check("tool-a", {"x": 1})
    assert not decision.allow
    assert decision.note


def test_soft_mode_allows_repeats_up_to_limit() -> None:
    """soft 模式的核心行为：有限重复放行，不打击调用意愿。"""
    deduper = CallDeduper(mode="soft", soft_limit=3)

    assert deduper.check("tool-a", {"x": 1}).allow
    assert deduper.check("tool-a", {"x": 1}).allow
    assert deduper.check("tool-a", {"x": 1}).allow

    decision = deduper.check("tool-a", {"x": 1})
    assert not decision.allow
    assert "重复" in decision.note


def test_soft_mode_echoes_previous_result() -> None:
    deduper = CallDeduper(mode="soft", soft_limit=3)

    deduper.check("tool-a", {"x": 1})
    deduper.record_result("tool-a", {"x": 1}, "今天晴天")

    decision = deduper.check("tool-a", {"x": 1})
    assert decision.allow
    assert "今天晴天" in decision.note


def test_record_result_truncates_long_output() -> None:
    deduper = CallDeduper(mode="soft")
    deduper.check("tool-a", {"x": 1})
    deduper.record_result("tool-a", {"x": 1}, "长" * 1000)

    decision = deduper.check("tool-a", {"x": 1})
    assert len(decision.note) < 400


def test_record_result_serializes_dict() -> None:
    deduper = CallDeduper(mode="soft")
    deduper.check("tool-a", {"x": 1})
    deduper.record_result("tool-a", {"x": 1}, {"temp": 25})

    decision = deduper.check("tool-a", {"x": 1})
    assert "temp" in decision.note


def test_reset_clears_state() -> None:
    deduper = CallDeduper(mode="hard")
    deduper.check("tool-a", {"x": 1})
    deduper.reset()

    assert deduper.check("tool-a", {"x": 1}).allow


def test_different_tools_do_not_interfere() -> None:
    deduper = CallDeduper(mode="hard")
    assert deduper.check("tool-a", {"x": 1}).allow
    assert deduper.check("tool-b", {"x": 1}).allow

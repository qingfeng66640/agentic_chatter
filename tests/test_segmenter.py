"""消息分段与打字节奏测试。"""

from __future__ import annotations

from ..humanize.segmenter import (
    clean_reply_text,
    segment_reply,
)


def test_clean_strips_thought_prefix() -> None:
    assert clean_reply_text("内心：我该回复他") == "我该回复他"
    assert clean_reply_text("[思考] 嗯") == "嗯"


def test_clean_strips_suspend_marker() -> None:
    assert clean_reply_text("__SUSPEND__") == ""


def test_clean_strips_wrapping_quotes() -> None:
    assert clean_reply_text('"你好呀"') == "你好呀"
    assert clean_reply_text("「在的」") == "在的"


def test_clean_returns_empty_for_blank() -> None:
    assert clean_reply_text("") == ""
    assert clean_reply_text("   ") == ""


def test_segmentation_disabled_returns_single_segment() -> None:
    segments = segment_reply("啊" * 200, enabled=False)
    assert len(segments) == 1
    assert segments[0].delay == 0.0


def test_segments_split_at_sentence_boundary() -> None:
    text = "今天天气不错。我们去爬山吧！你觉得怎么样？"
    segments = segment_reply(text, max_segment_chars=10, max_segments=4, typing_cps=0)

    assert len(segments) > 1
    assert segments[0].text.endswith("。")


def test_first_segment_has_no_delay() -> None:
    segments = segment_reply(
        "第一句话。第二句话。第三句话。",
        max_segment_chars=6,
        max_segments=4,
        typing_cps=8.0,
    )

    assert segments[0].delay == 0.0
    if len(segments) > 1:
        assert segments[1].delay > 0


def test_delay_is_capped() -> None:
    segments = segment_reply(
        "短。" + "长" * 500,
        max_segment_chars=400,
        max_segments=2,
        typing_cps=1.0,
        max_typing_delay=3.0,
    )

    assert all(segment.delay <= 3.0 for segment in segments)


def test_zero_cps_produces_no_delay() -> None:
    segments = segment_reply(
        "一句。两句。三句。",
        max_segment_chars=4,
        max_segments=4,
        typing_cps=0.0,
    )

    assert all(segment.delay == 0.0 for segment in segments)


def test_max_segments_is_respected() -> None:
    text = "。".join(f"句子{index}" for index in range(20))
    segments = segment_reply(text, max_segment_chars=5, max_segments=3, typing_cps=0)

    assert len(segments) <= 3


def test_empty_input_returns_no_segments() -> None:
    assert segment_reply("") == []
    assert segment_reply("   ") == []


def test_hard_split_when_no_punctuation() -> None:
    segments = segment_reply(
        "啊" * 100,
        max_segment_chars=20,
        max_segments=3,
        typing_cps=0,
    )

    assert len(segments) > 1
    assert all(segment.text for segment in segments)

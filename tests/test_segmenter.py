"""消息分段与打字节奏测试。"""

from __future__ import annotations

from ..humanize.segmenter import (
    clean_reply_text,
    clean_reply_text_with_metadata,
    detect_provider_error_text,
    detect_reply_decision_json_text,
    is_framework_message_line,
    segment_reply,
)


def test_detect_provider_error_text_matches_google_policy_response() -> None:
    text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]. "
        "Try rephrasing the prompt."
    )
    assert detect_provider_error_text(text) == "google_prompt_policy_block"


def test_detect_provider_error_text_normalizes_case_and_whitespace() -> None:
    text = (
        "THE PROMPT COULD NOT BE SUBMITTED.\n"
        "The prompt contains sensitive content.\n"
        "Try rephrasing the prompt."
    )
    assert detect_provider_error_text(text) == "google_prompt_policy_block"


def test_detect_provider_error_text_does_not_match_generic_discussion() -> None:
    assert detect_provider_error_text("请解释 content policy 和 API error 的区别。") is None
    assert detect_provider_error_text("The prompt could not be submitted.") is None
    assert detect_provider_error_text("Try rephrasing the prompt.") is None


def test_detect_provider_error_text_requires_complete_provider_signal() -> None:
    assert detect_provider_error_text(
        "The prompt could not be submitted. The prompt contains sensitive words."
    ) is None


def test_detect_reply_decision_json_text_matches_complete_schema() -> None:
    text = (
        '{"action":"silent","confidence":0.8,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":["not_directed_to_bot"],'
        '"brief_reason":"未指向 bot"}'
    )
    assert detect_reply_decision_json_text(text) == "silent"


def test_detect_reply_decision_json_text_matches_markdown_json_fence() -> None:
    text = '''```json
{"action":"silent","confidence":0.8,"addressee":"other","interrupt_cost":0.0,"reason_codes":[],"brief_reason":"未指向 bot"}
```'''
    assert detect_reply_decision_json_text(text) == "silent"
def test_detect_reply_decision_json_text_rejects_incomplete_or_normal_json() -> None:
    assert detect_reply_decision_json_text('{"action":"silent"}') is None
    assert detect_reply_decision_json_text(
        '{"action":"silent","confidence":0.8,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":[],"brief_reason":1}'
    ) is None
    assert detect_reply_decision_json_text('{"message":"我会保持沉默"}') is None


def test_detect_reply_decision_json_text_rejects_invalid_numbers() -> None:
    text = (
        '{"action":"silent","confidence":true,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":[],"brief_reason":"x"}'
    )
    assert detect_reply_decision_json_text(text) is None
    text = (
        '{"action":"silent","confidence":1.1,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":[],"brief_reason":"x"}'
    )
    assert detect_reply_decision_json_text(text) is None


def test_clean_metadata_contains_removed_thoughts() -> None:
    result = clean_reply_text_with_metadata(
        "<think>先分析语气</think>[思考]再确认一下[/思考]你好"
    )

    assert result.text == "你好"
    assert result.removed_thoughts == ("先分析语气", "再确认一下")


def test_clean_metadata_keeps_structured_reasoning_separate() -> None:
    result = clean_reply_text_with_metadata("普通回复")

    assert result.text == "普通回复"
    assert result.removed_thoughts == ()


def test_clean_removes_tagged_thought_blocks() -> None:
    assert clean_reply_text("<think>内部推理</think>你好") == "你好"
    assert clean_reply_text("<ANALYSIS data-x='1'>多行\n分析</ANALYSIS>答案") == "答案"
    assert clean_reply_text("<analysis>步骤一</analysis><reasoning>步骤二</reasoning>结果") == "结果"


def test_clean_removes_bracketed_thought_blocks() -> None:
    assert clean_reply_text("[思考]内部推理[/思考]你好") == "你好"
    assert clean_reply_text("【内心】不该发送【/内心】在的") == "在的"
    assert clean_reply_text("[OS]internal[/OS]收到") == "收到"


def test_clean_keeps_reply_after_explicit_thought_section() -> None:
    assert clean_reply_text("[思考]先判断语气\n[最终回复]你好呀") == "你好呀"
    assert clean_reply_text("分析过程：先看上下文\n回复：可以的") == "可以的"


def test_clean_returns_empty_for_thought_only_output() -> None:
    assert clean_reply_text("<think>只有推理</think>") == ""
    assert clean_reply_text("【思考】只有独白【/思考】") == ""
    assert segment_reply("<analysis>只有分析</analysis>") == []


def test_clean_does_not_remove_unclosed_or_mismatched_tags() -> None:
    assert clean_reply_text("<think>未闭合内容\n最终回复") == "<think>未闭合内容\n最终回复"
    assert clean_reply_text("<think>内容</analysis>回答") == "<think>内容</analysis>回答"


def test_clean_preserves_normal_visible_reasoning_words() -> None:
    assert clean_reply_text("我想先确认一下需求。") == "我想先确认一下需求。"
    assert clean_reply_text("我觉得这个方案可行。") == "我觉得这个方案可行。"
    assert clean_reply_text("我的分析：问题来自配置缺失。") == "我的分析：问题来自配置缺失。"


def test_clean_strips_suspend_marker() -> None:
    assert clean_reply_text("__SUSPEND__") == ""


def test_clean_strips_wrapping_quotes() -> None:
    assert clean_reply_text('"你好呀"') == "你好呀"
    assert clean_reply_text("「在的」") == "在的"


def test_clean_returns_empty_for_blank() -> None:
    assert clean_reply_text("") == ""
    assert clean_reply_text("   ") == ""


def test_framework_message_line_detection_requires_complete_single_line() -> None:
    assert is_framework_message_line(
        "【13:47】<机器人> [6264745991384149877] 我是打工蝶😭 ："
    )
    assert is_framework_message_line(
        "【13:47】<成员> [user-id] 小豆 [message-id]：你好"
    )
    assert not is_framework_message_line("机器人说：你好")
    assert not is_framework_message_line("说明： 【13:47】<机器人> [1] 小蝶 ：你好")
    assert not is_framework_message_line("【13:47】<机器人> [1] 小蝶 ：你好\n补充说明")
    assert not is_framework_message_line("【13:47】<机器人> [1] 小蝶: 你好")


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

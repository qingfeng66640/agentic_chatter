"""消息分段与打字节奏测试。"""

from __future__ import annotations

import pytest

from ..humanize.grapheme import find_units, reconstruct, split_into_atoms
from ..humanize.segmenter import (
    _assert_units_intact,
    clean_reply_text,
    clean_reply_text_with_metadata,
    detect_provider_error_text,
    detect_reply_decision_json_text,
    is_framework_message_line,
    segment_reply,
)


def test_detect_provider_error_text_matches_google_policy_response() -> None:
    """完整的 Google 政策拒绝响应被识别为提示词政策拦截。"""
    text = (
        "The prompt could not be submitted. The prompt contains sensitive words "
        "that violate Google's [Generative AI Prohibited Use Policy]. "
        "Try rephrasing the prompt."
    )
    assert detect_provider_error_text(text) == "google_prompt_policy_block"


def test_detect_provider_error_text_matches_gemini_filter_response() -> None:
    """Gemini 过滤器拒绝响应及其大写换行形式均返回过滤拦截原因。"""
    text = (
        "This request was blocked by Gemini's filters. They can occasionally trigger "
        "by mistake on safe coding, security, or biology-related queries. Please try "
        "rephrasing your prompt. You can [send feedback] or read more about our policies here."
    )
    assert detect_provider_error_text(text) == "gemini_filter_block"
    assert (
        detect_provider_error_text(
            "THIS REQUEST WAS BLOCKED BY GEMINI'S FILTERS.\n"
            "Please try rephrasing your prompt."
        )
        == "gemini_filter_block"
    )


def test_detect_provider_error_text_normalizes_case_and_whitespace() -> None:
    """Google 拒绝信号的大写及换行变体仍能被识别。"""
    text = (
        "THE PROMPT COULD NOT BE SUBMITTED.\n"
        "The prompt contains sensitive content.\n"
        "Try rephrasing the prompt."
    )
    assert detect_provider_error_text(text) == "google_prompt_policy_block"


def test_detect_provider_error_text_does_not_match_generic_discussion() -> None:
    """普通政策讨论与孤立的拒绝提示片段不会误报供应商拦截。"""
    assert (
        detect_provider_error_text("请解释 content policy 和 API error 的区别。")
        is None
    )
    assert detect_provider_error_text("The prompt could not be submitted.") is None
    assert detect_provider_error_text("Try rephrasing the prompt.") is None
    assert detect_provider_error_text("Gemini's filters can block a request.") is None
    assert detect_provider_error_text("Please try rephrasing your prompt.") is None


def test_detect_provider_error_text_requires_complete_provider_signal() -> None:
    """缺少改写提示的 Google 拒绝片段不构成完整拦截信号。"""
    assert (
        detect_provider_error_text(
            "The prompt could not be submitted. The prompt contains sensitive words."
        )
        is None
    )


def test_detect_reply_decision_json_text_matches_complete_schema() -> None:
    """字段完整且类型正确的决策 JSON 被识别为静默决策。"""
    text = (
        '{"action":"silent","confidence":0.8,"addressee":"other",'
        '"interrupt_cost":0.0,"reason_codes":["not_directed_to_bot"],'
        '"brief_reason":"未指向 bot"}'
    )
    assert detect_reply_decision_json_text(text) == "silent"


def test_detect_reply_decision_json_text_matches_markdown_json_fence() -> None:
    """Markdown JSON 围栏中的完整决策仍被识别为静默决策。"""
    text = """```json
{"action":"silent","confidence":0.8,"addressee":"other","interrupt_cost":0.0,"reason_codes":[],"brief_reason":"未指向 bot"}
```"""
    assert detect_reply_decision_json_text(text) == "silent"


def test_detect_reply_decision_json_text_rejects_incomplete_or_normal_json() -> None:
    """字段不全、理由类型错误及普通消息 JSON 均不被视为决策。"""
    assert detect_reply_decision_json_text('{"action":"silent"}') is None
    assert (
        detect_reply_decision_json_text(
            '{"action":"silent","confidence":0.8,"addressee":"other",'
            '"interrupt_cost":0.0,"reason_codes":[],"brief_reason":1}'
        )
        is None
    )
    assert detect_reply_decision_json_text('{"message":"我会保持沉默"}') is None


def test_detect_reply_decision_json_text_rejects_invalid_numbers() -> None:
    """布尔值及超出有效范围的置信度不能通过决策识别。"""
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
    """清洗结果保留可见回复，并按顺序记录两段被移除的思考内容。"""
    result = clean_reply_text_with_metadata(
        "<think>先分析语气</think>[思考]再确认一下[/思考]你好"
    )

    assert result.text == "你好"
    assert result.removed_thoughts == ("先分析语气", "再确认一下")


def test_clean_metadata_keeps_structured_reasoning_separate() -> None:
    """普通回复保持原文，移除思考的元数据为空元组。"""
    result = clean_reply_text_with_metadata("普通回复")

    assert result.text == "普通回复"
    assert result.removed_thoughts == ()


def test_clean_removes_tagged_thought_blocks() -> None:
    """成对的思考、分析和推理标签连同内容被移除，保留后续回复。"""
    assert clean_reply_text("<think>内部推理</think>你好") == "你好"
    assert clean_reply_text("<ANALYSIS data-x='1'>多行\n分析</ANALYSIS>答案") == "答案"
    assert (
        clean_reply_text("<analysis>步骤一</analysis><reasoning>步骤二</reasoning>结果")
        == "结果"
    )


def test_clean_removes_bracketed_thought_blocks() -> None:
    """中英文方括号思考标记及其内部内容被移除，保留可见回复。"""
    assert clean_reply_text("[思考]内部推理[/思考]你好") == "你好"
    assert clean_reply_text("【内心】不该发送【/内心】在的") == "在的"
    assert clean_reply_text("[OS]internal[/OS]收到") == "收到"


def test_clean_keeps_reply_after_explicit_thought_section() -> None:
    """显式最终回复分隔符之前的思考被移除，之后的回答被保留。"""
    assert clean_reply_text("[思考]先判断语气\n[最终回复]你好呀") == "你好呀"
    assert clean_reply_text("分析过程：先看上下文\n回复：可以的") == "可以的"


def test_clean_returns_empty_for_thought_only_output() -> None:
    """仅含思考块的输出清洗为空文本，分段结果也为空列表。"""
    assert clean_reply_text("<think>只有推理</think>") == ""
    assert clean_reply_text("【思考】只有独白【/思考】") == ""
    assert segment_reply("<analysis>只有分析</analysis>") == []


def test_clean_does_not_remove_unclosed_or_mismatched_tags() -> None:
    """未闭合或首尾不匹配的标签内容保持原样，避免误删正文。"""
    assert (
        clean_reply_text("<think>未闭合内容\n最终回复") == "<think>未闭合内容\n最终回复"
    )
    assert (
        clean_reply_text("<think>内容</analysis>回答") == "<think>内容</analysis>回答"
    )


def test_clean_preserves_normal_visible_reasoning_words() -> None:
    """正文中的思考、判断和分析措辞不会被当作内部推理删除。"""
    assert clean_reply_text("我想先确认一下需求。") == "我想先确认一下需求。"
    assert clean_reply_text("我觉得这个方案可行。") == "我觉得这个方案可行。"
    assert (
        clean_reply_text("我的分析：问题来自配置缺失。")
        == "我的分析：问题来自配置缺失。"
    )


def test_clean_strips_suspend_marker() -> None:
    """单独的挂起标记被清洗为空文本。"""
    assert clean_reply_text("__SUSPEND__") == ""


def test_clean_strips_wrapping_quotes() -> None:
    """包围回复的英文双引号与直角引号被剥离。"""
    assert clean_reply_text('"你好呀"') == "你好呀"
    assert clean_reply_text("「在的」") == "在的"


def test_clean_returns_empty_for_blank() -> None:
    """空字符串与纯空白回复均清洗为空文本。"""
    assert clean_reply_text("") == ""
    assert clean_reply_text("   ") == ""


def test_framework_message_line_detection_requires_complete_single_line() -> None:
    """仅完整单行框架消息被识别，普通文本、前缀说明和格式不完整的行不匹配。"""
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
    """关闭分段时长回复只产生一个无延迟片段。"""
    segments = segment_reply("啊" * 200, enabled=False)
    assert len(segments) == 1
    assert segments[0].delay == 0.0


def test_segments_split_at_sentence_boundary() -> None:
    """句子超过分段长度时拆为多段，首段保留句号边界。"""
    text = "今天天气不错。我们去爬山吧！你觉得怎么样？"
    segments = segment_reply(text, max_segment_chars=10, max_segments=4, typing_cps=0)

    assert len(segments) > 1
    assert segments[0].text.endswith("。")


def test_first_segment_has_no_delay() -> None:
    """首段没有打字延迟，存在后续段时其延迟为正。"""
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
    """慢速打字与长片段组合下，每段延迟仍不超过三秒上限。"""
    segments = segment_reply(
        "短。" + "长" * 500,
        max_segment_chars=400,
        max_segments=2,
        typing_cps=1.0,
        max_typing_delay=3.0,
    )

    assert all(segment.delay <= 3.0 for segment in segments)


def test_zero_cps_produces_no_delay() -> None:
    """打字速度设为零时所有片段都不增加延迟。"""
    segments = segment_reply(
        "一句。两句。三句。",
        max_segment_chars=4,
        max_segments=4,
        typing_cps=0.0,
    )

    assert all(segment.delay == 0.0 for segment in segments)


def test_max_segments_is_respected() -> None:
    """多句长回复的分段数量不超过配置的三段上限。"""
    text = "。".join(f"句子{index}" for index in range(20))
    segments = segment_reply(text, max_segment_chars=5, max_segments=3, typing_cps=0)

    assert len(segments) <= 3


def test_empty_input_returns_no_segments() -> None:
    """空文本与纯空白文本均不生成消息片段。"""
    assert segment_reply("") == []
    assert segment_reply("   ") == []


def test_hard_split_when_no_punctuation() -> None:
    """无标点的长文本会被硬切为多个非空片段。"""
    segments = segment_reply(
        "啊" * 100,
        max_segment_chars=20,
        max_segments=3,
        typing_cps=0,
    )

    assert len(segments) > 1
    assert all(segment.text for segment in segments)


# --- 不可拆单元（颜文字与表情） -------------------------------------------------

# 覆盖各类书写风格：半角/全角/尖角/方括号/花括号/波浪号、网络缩写、尾巴外挂。
INDIVISIBLE_UNITS = (
    "(´･ω･`)",
    "（＾▽＾）",
    "<(=･ω･=)>",
    "[･ω･]",
    "{・ω・}",
    "~(´ω｀)~",
    "〜(´ω｀〜)",
    "(´;ω;`)",
    "(:﹏﹏:)",
    "(๑•̀ㅂ•́)و✧",
    "(((((*´ω`*) ))))",
    "QAQ",
    "\U0001f600",
    "\U0001f44b\U0001f3fd",
    "\U0001f468‍\U0001f469‍\U0001f467",
    "1️⃣",
    "\U0001f469‍\U0001f4bb",
)

# 带尾部填充，供裸颜文字用例复用
FILLER_TAIL = "今天我们讨论了很多关于项目排期和技术选型的问题整体进展还算顺利甲甲甲甲甲"


@pytest.mark.parametrize("unit", INDIVISIBLE_UNITS)
def test_units_survive_every_split_position(unit: str) -> None:
    """切点落在单元任意位置时，单元都必须完整落在同一段。

    修复前硬切与标点两条路径都会把单元切碎，因此这里逐一扫描
    单元前后每个可能的切点。
    """
    text = FILLER_TAIL + unit + "乙" * 25
    base = len(FILLER_TAIL)

    for cut in range(base, base + len(unit) + 1):
        segments = segment_reply(
            text, max_segment_chars=cut, max_segments=2, typing_cps=0
        )
        if len(segments) < 2:
            continue
        assert any(unit in segment.text for segment in segments), (
            f"单元 {unit!r} 在切点 {cut} 处被拆散"
        )


def test_kaomoji_with_chinese_punctuation_is_not_split() -> None:
    """颜文字内的中文标点不得被当作语义边界。"""
    text = "好的 (，´ω`)，我这就去办，你看行吗？今天应该能搞定 (¯τωτ)"
    segments = segment_reply(text, max_segment_chars=16, max_segments=4, typing_cps=0)

    for unit in ("(，´ω`)", "(¯τωτ)"):
        assert any(unit in segment.text for segment in segments)


def test_segments_join_back_to_original() -> None:
    """分段拼接必须与原文完全一致。"""
    text = "今天天气不错 (´･ω･`) 。我们去爬山吧！👨‍👩‍👧 你觉得怎么样？"
    segments = segment_reply(text, max_segment_chars=10, max_segments=4, typing_cps=0)

    assert "".join(segment.text for segment in segments) == text


def test_single_unit_longer_than_limit_is_not_broken() -> None:
    """单个单元超过上限时宁可整条发出，也不拆散它。"""
    unit = "(((((*´ω`*) ))))"
    text = "甲" * 30 + unit + "乙" * 30
    segments = segment_reply(text, max_segment_chars=10, max_segments=4, typing_cps=0)

    assert any(unit in segment.text for segment in segments)
    assert "".join(segment.text for segment in segments) == text


def test_plain_bracketed_text_is_not_treated_as_unit() -> None:
    """普通括号内容不应被误判为颜文字而改变切分行为。"""
    text = "这个方案（重要）需要（注意）评审，大家（abc）先看一下"
    segments = segment_reply(text, max_segment_chars=12, max_segments=4, typing_cps=0)

    assert "".join(segment.text for segment in segments) == text


def test_find_units_reports_kind_and_span() -> None:
    """表情与颜文字的类型、顺序及原文位置区间均正确返回。"""
    text = "开心 \U0001f600 呀 (´･ω･`) 真的"
    units = find_units(text)

    assert [(unit.text, unit.kind) for unit in units] == [
        ("\U0001f600", "emoji"),
        ("(´･ω･`)", "kaomoji"),
    ]
    for unit in units:
        assert text[unit.start : unit.end] == unit.text


def test_split_into_atoms_round_trips() -> None:
    """原子拆分结果包含不可拆单元标记，并能无损重建原文。"""
    text = "你好 \U0001f600 (´･ω･`) 结束"
    atoms = split_into_atoms(text, find_units(text))

    assert reconstruct(atoms) == text
    assert any(is_unit for _, is_unit in atoms)


def test_grapheme_backend_fallback_keeps_emoji_whole() -> None:
    """regex 缺席的降级路径下，ZWJ 序列仍需完整保留。"""
    text = "今天 \U0001f468‍\U0001f469‍\U0001f467 很好"

    assert [unit.text for unit in find_units(text)] == [
        "\U0001f468‍\U0001f469‍\U0001f467"
    ]


def test_bare_kaomoji_without_brackets_is_recognized() -> None:
    """中点包裹的裸颜文字同样不可拆。"""
    text = "今天真开心 、´ω´、 明天继续"

    assert [unit.text for unit in find_units(text)] == ["、´ω´、"]


def test_bare_kaomoji_survives_split() -> None:
    """扫描裸颜文字周围的切点时，颜文字始终完整保留在某一段中。"""
    text = FILLER_TAIL + "、´ω´、" + "乙" * 20
    base = len(FILLER_TAIL)

    for cut in range(base, base + 6):
        segments = segment_reply(
            text, max_segment_chars=cut, max_segments=2, typing_cps=0
        )
        if len(segments) < 2:
            continue
        assert any("、´ω´、" in segment.text for segment in segments)


def test_integrity_check_rejects_split_units() -> None:
    """自检必须在单元被拆散时报警，防止损坏的消息被静默发出。"""
    # 拼接结果与原文一致，但颜文字被拆到两段：仍属损坏
    with pytest.raises(AssertionError, match="单元"):
        _assert_units_intact("甲乙(´･ω･`)丙", ["甲乙(´･", "ω･`)丙"])


def test_integrity_check_rejects_content_loss() -> None:
    """自检必须在内容丢失时报警。"""
    with pytest.raises(AssertionError, match="不一致"):
        _assert_units_intact("甲乙丙", ["甲乙"])

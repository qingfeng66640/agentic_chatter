"""Agentic Chatter 自定义系统提示词区块测试。"""

from __future__ import annotations

from types import SimpleNamespace

from ..chatter import _AgenticChatterBase
from ..prompts import (
    DEFAULT_HOW_YOU_ACT,
    DEFAULT_HOW_YOU_SPEAK,
    DEFAULT_WHEN_TO_STOP,
    system_prompt,
)


def test_prompt_section_uses_default_for_missing_or_blank_values() -> None:
    """缺失、空字符串和空白值均回退内置区块。"""
    for value, default, tag in (
        (None, DEFAULT_HOW_YOU_SPEAK, "how_you_speak"),
        ("", DEFAULT_HOW_YOU_ACT, "how_you_act"),
        ("  \n", DEFAULT_WHEN_TO_STOP, "when_to_stop"),
    ):
        assert _AgenticChatterBase._prompt_section(value, default, tag) == default


def test_prompt_section_wraps_only_custom_body() -> None:
    """自定义内容只替换对应标签内部正文并保留格式。"""
    custom = "  第一行\n\n    第二行  "

    rendered = _AgenticChatterBase._prompt_section(
        custom,
        DEFAULT_HOW_YOU_SPEAK,
        "how_you_speak",
    )

    assert rendered == f"<how_you_speak>\n{custom}\n</how_you_speak>"
    assert rendered.count("<how_you_speak>") == 1
    assert rendered.count("</how_you_speak>") == 1
    assert "第一行\n\n    第二行  " in rendered


def test_prompt_section_defaults_are_independent() -> None:
    """单独覆盖一个区块时，另外两个仍可使用原始区块。"""
    custom_speak = _AgenticChatterBase._prompt_section(
        "只说重点",
        DEFAULT_HOW_YOU_SPEAK,
        "how_you_speak",
    )

    assert "只说重点" in custom_speak
    assert DEFAULT_HOW_YOU_ACT.startswith("<how_you_act>")
    assert DEFAULT_WHEN_TO_STOP.startswith("<when_to_stop>")


def test_default_prompts_distinguish_direct_text_from_say_action() -> None:
    """内置提示必须明确正文直出与 action-say 是互斥的发送路径。"""
    assert "普通回复直接输出正文" in DEFAULT_HOW_YOU_SPEAK
    assert "额外发送一条消息" in DEFAULT_HOW_YOU_SPEAK
    assert "本次模型响应不要再输出正文" in DEFAULT_HOW_YOU_SPEAK
    assert "不要先直出一句、再用 action-say 发送同义或补全后的版本" in DEFAULT_HOW_YOU_ACT
    assert "没有引用或 @ 需求时，不要调用 action-say" in DEFAULT_HOW_YOU_ACT


def test_system_prompt_keeps_send_channel_rule_outside_custom_sections() -> None:
    """发送通道约束必须保留在系统模板固定区，不被 persona 覆盖。"""
    assert "<message_channel_rules>" in system_prompt
    assert "普通回复直接输出正文，框架会自动发送" in system_prompt
    assert "本次响应不要再输出正文" in system_prompt
    assert "必须填写 target_type（reply 或 mention）和非空 target_id" in system_prompt
    assert "不要先直接输出正文，再用 action-say 重发或改写同一内容" in system_prompt


def test_persona_fields_are_available_with_default_values() -> None:
    """配置 persona 默认字段为空，交由构建层回退内置模板。"""
    persona = SimpleNamespace(how_you_speak="", how_you_act="", when_to_stop="")

    assert persona.how_you_speak == ""
    assert persona.how_you_act == ""
    assert persona.when_to_stop == ""


__all__ = []

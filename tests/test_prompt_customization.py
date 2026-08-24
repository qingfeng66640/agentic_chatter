"""Agentic Chatter 自定义系统提示词区块测试。"""

from __future__ import annotations

from types import SimpleNamespace

from ..chatter import _AgenticChatterBase
from ..prompts import (
    DEFAULT_HOW_YOU_ACT,
    DEFAULT_HOW_YOU_SPEAK,
    DEFAULT_WHEN_TO_STOP,
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


def test_persona_fields_are_available_with_default_values() -> None:
    """配置 persona 默认字段为空，交由构建层回退内置模板。"""
    persona = SimpleNamespace(how_you_speak="", how_you_act="", when_to_stop="")

    assert persona.how_you_speak == ""
    assert persona.how_you_act == ""
    assert persona.when_to_stop == ""


__all__ = []

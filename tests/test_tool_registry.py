"""工具分层与渐进披露测试。

这些测试守护本插件对「工具调用率低」的核心修复：
海量工具必须被折叠，拟人化核心工具必须常驻置顶。
"""

from __future__ import annotations

from ..tooling.registry import (
    build_encouragement_prompt,
    build_tool_layout,
    is_blacklisted_component,
    signature_matches,
)


def _make_component(signature: str, short_name: str) -> type:
    """构造一个带签名的假组件类。

    Args:
        signature: 组件签名。
        short_name: 组件短名。

    Returns:
        type: 假组件类。
    """

    class _Fake:
        tool_name = short_name

        @classmethod
        def get_signature(cls) -> str:
            return signature

        @classmethod
        def to_schema(cls) -> dict:
            return {"function": {"name": f"tool-{short_name}"}}

    _Fake.__name__ = f"Fake_{short_name}"
    return _Fake


def test_signature_matches_supports_wildcard() -> None:
    assert signature_matches("onebot_expand:tool:get_group_info", ["onebot_expand:tool:*"])
    assert not signature_matches("emoji_sender:action:send", ["onebot_expand:tool:*"])
    assert not signature_matches("", ["*"])


def test_blacklist_hides_component_entirely() -> None:
    usables = [_make_component("bad_plugin:tool:danger", "danger")]

    layout = build_tool_layout(
        usables,
        always_visible=[],
        collapsed=[],
        blacklist=["bad_plugin:tool:*"],
        max_exposed=10,
    )

    assert layout.exposed == []
    assert layout.collapsed_signatures == {}


def test_is_blacklisted_component_uses_signature_patterns() -> None:
    component = _make_component("bad_plugin:tool:danger", "danger")

    assert is_blacklisted_component(component, ["bad_plugin:tool:*"])
    assert not is_blacklisted_component(component, ["good_plugin:tool:*"])


def test_always_visible_components_are_pinned_first() -> None:
    usables = [
        _make_component("other:tool:filler_1", "filler_1"),
        _make_component("other:tool:filler_2", "filler_2"),
        _make_component("emoji_sender:action:send_emoji", "send_emoji"),
    ]

    layout = build_tool_layout(
        usables,
        always_visible=["emoji_sender:action:*"],
        collapsed=[],
        blacklist=[],
        max_exposed=10,
    )

    first = layout.exposed[0]
    assert first.get_signature() == "emoji_sender:action:send_emoji"


def test_collapsed_components_do_not_consume_exposure_budget() -> None:
    """折叠工具不应占用暴露预算 —— 这是防止 schema 洪水的关键。"""
    usables = [
        _make_component(f"onebot_expand:tool:api_{index}", f"api_{index}")
        for index in range(200)
    ]
    usables.append(_make_component("emoji_sender:action:send_emoji", "send_emoji"))

    layout = build_tool_layout(
        usables,
        always_visible=["emoji_sender:action:*"],
        collapsed=["onebot_expand:tool:*"],
        blacklist=[],
        max_exposed=25,
    )

    assert len(layout.exposed) == 1
    assert len(layout.collapsed_signatures) == 200
    assert "onebot_expand" in layout.collapsed_categories


def test_max_exposed_truncates_normal_components() -> None:
    usables = [_make_component(f"p:tool:t{index}", f"t{index}") for index in range(50)]

    layout = build_tool_layout(
        usables,
        always_visible=[],
        collapsed=[],
        blacklist=[],
        max_exposed=10,
    )

    assert len(layout.exposed) == 10
    assert layout.dropped_count == 40


def test_max_exposed_zero_means_unlimited() -> None:
    usables = [_make_component(f"p:tool:t{index}", f"t{index}") for index in range(50)]

    layout = build_tool_layout(
        usables,
        always_visible=[],
        collapsed=[],
        blacklist=[],
        max_exposed=0,
    )

    assert len(layout.exposed) == 50
    assert layout.dropped_count == 0


def test_describe_categories_renders_hint() -> None:
    usables = [
        _make_component(f"onebot_expand:tool:api_{index}", f"api_{index}")
        for index in range(12)
    ]

    layout = build_tool_layout(
        usables,
        always_visible=[],
        collapsed=["onebot_expand:tool:*"],
        blacklist=[],
        max_exposed=25,
    )

    text = layout.describe_categories()
    assert "explore_tools" in text
    assert "onebot_expand" in text


def test_describe_categories_empty_when_nothing_collapsed() -> None:
    layout = build_tool_layout(
        [_make_component("p:tool:t", "t")],
        always_visible=[],
        collapsed=[],
        blacklist=[],
        max_exposed=25,
    )

    assert layout.describe_categories() == ""


def test_encouragement_prompt_mentions_tool_usage() -> None:
    text = build_encouragement_prompt()
    assert "查" in text
    assert "表情包" in text

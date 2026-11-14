"""工具分层与渐进披露测试。

这些测试守护本插件对「工具调用率低」的核心修复：
海量工具必须被折叠，拟人化核心工具必须常驻置顶。
"""

from __future__ import annotations

from ..chatter import AgenticChatter
from ..config import AgenticChatterConfig
from ..tooling.registry import (
    build_encouragement_prompt,
    build_tool_layout,
    is_blacklisted_component,
    signature_matches,
)


def _make_component(signature: str, short_name: str) -> type[object]:
    """构造一个带签名的假组件类。

    Args:
        signature: 组件签名。
        short_name: 组件短名。

    Returns:
        type: 假组件类。
    """

    class _Fake:
        """保留传入签名和短名的组件替身，用于工具筛选与排序。"""

        name = short_name

        @classmethod
        def get_signature(cls) -> str:
            """返回构造时传入的组件签名，供匹配和黑名单过滤使用。"""
            return signature

        @classmethod
        def to_schema(cls) -> dict[str, dict[str, str]]:
            """用组件短名构造最小工具结构，供工具布局识别名称。"""
            return {"function": {"name": f"tool-{short_name}"}}

    _Fake.__name__ = f"Fake_{short_name}"
    return _Fake


def test_signature_matches_supports_wildcard() -> None:
    """通配签名应匹配指定插件工具，并排除其他插件和空签名。"""
    assert signature_matches(
        "onebot_expand:tool:get_group_info", ["onebot_expand:tool:*"]
    )
    assert not signature_matches("emoji_sender:action:send", ["onebot_expand:tool:*"])
    assert not signature_matches("", ["*"])


def test_blacklist_hides_component_entirely() -> None:
    """命中黑名单的组件既不得直接暴露，也不得进入折叠目录。"""
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
    """组件黑名单判断应依据所属插件的签名模式。"""
    component = _make_component("bad_plugin:tool:danger", "danger")

    assert is_blacklisted_component(component, ["bad_plugin:tool:*"])
    assert not is_blacklisted_component(component, ["good_plugin:tool:*"])


def test_always_visible_components_are_pinned_first() -> None:
    """常驻可见的表情动作应排在普通工具之前。"""
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
    """普通工具超过暴露预算时，应保留限额并记录丢弃数量。"""
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
    """暴露预算为零时应保留所有普通工具，不记录丢弃项。"""
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
    """折叠分类提示应包含探索工具入口和所属插件类别。"""
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
    """不存在折叠工具时，分类提示应为空字符串。"""
    layout = build_tool_layout(
        [_make_component("p:tool:t", "t")],
        always_visible=[],
        collapsed=[],
        blacklist=[],
        max_exposed=25,
    )

    assert layout.describe_categories() == ""


def test_tool_call_mode_defaults_to_planning() -> None:
    """默认配置和缺失配置都应选择逐个提交的规划模式。"""
    config = AgenticChatterConfig()
    chatter = AgenticChatter(stream_id="tool-mode", plugin=object())

    assert config.tools.tool_call_mode == "planning"
    assert chatter._tool_call_mode(config) == "planning"
    assert chatter._tool_call_mode(None) == "planning"


def test_tool_call_mode_invalid_value_falls_back_to_planning() -> None:
    """配置中的未知工具调用模式应按规划模式处理。"""
    config = AgenticChatterConfig()
    config.tools.tool_call_mode = "invalid"
    chatter = AgenticChatter(stream_id="tool-mode-invalid", plugin=object())

    assert chatter._tool_call_mode(config) == "planning"


def test_tool_call_mode_guidance_describes_planning_and_batch() -> None:
    """模式指引应说明规划等待结果，以及批量调度不保证并行和固定顺序。"""
    chatter = AgenticChatter(stream_id="tool-mode-guidance", plugin=object())

    planning = chatter._tool_call_mode_guidance("planning")
    batch = chatter._tool_call_mode_guidance("batch")

    assert "规划模式" in planning
    assert "逐个提交普通 Tool Call" in planning
    assert "等待 Tool Result" in planning
    assert "批量调度模式" in batch
    assert "整批交给 MoFox Core 调度" in batch
    assert "绝对并行" in batch
    assert "固定执行顺序" in batch


def test_encouragement_prompt_mentions_tool_usage() -> None:
    """工具鼓励提示应强调实际调用、结果依赖、独立批次和新信息发言约束。"""
    text = build_encouragement_prompt()
    assert "优先调用实际可用的工具" in text
    assert "查询、读取、计算、记录或执行动作" in text
    assert "表情包" in text
    assert "排序、排队和调度" in text
    assert "真实结果、生成的 ID、查询内容或执行状态" in text
    assert "不要在同一轮预先发出后一个调用" in text
    assert "互相独立且没有顺序要求的工具可以在同一轮组合调用" in text
    assert "只有工具结果带来新信息时才补充正文" in text

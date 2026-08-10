"""工具分层与渐进披露。

本模块解决 DFC 的核心问题之一：工具 schema 洪水稀释模型注意力。

现状是全局注册表中有 271 个可调用组件，仅 ``onebot_expand`` 一家就
注册了 206 个 Tool。全部注入给 LLM 会导致模型注意力被稀释，反而
表现为「几乎不调用任何工具」。

本模块把工具分成三层：

- **常驻层**：拟人化最关键的工具（表情包、记忆、日程等），永远可见且置顶。
- **普通层**：其余工具，按上限截断后可见。
- **折叠层**：数量庞大的平台 API 工具集，收进分类目录，
  由模型调用 ``explore_tools`` 按需展开。
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

# 折叠工具按签名的 plugin_name 分类时，单个分类展示的最大工具数
MAX_TOOLS_PER_CATEGORY = 60


def signature_matches(signature: str, patterns: list[str]) -> bool:
    """判断组件签名是否命中任一通配模式。

    Args:
        signature: 组件签名，形如 ``plugin_name:tool:tool_name``。
        patterns: 通配模式列表，支持 ``*``，例如 ``onebot_expand:tool:*``。

    Returns:
        bool: 命中任一模式时返回 True。
    """
    target = str(signature or "").strip()
    if not target:
        return False
    return any(fnmatch.fnmatch(target, str(pattern).strip()) for pattern in patterns if pattern)


@dataclass
class ToolLayout:
    """一次工具分层的结果。

    Attributes:
        exposed: 本轮实际注入给 LLM 的组件类列表，常驻工具排在最前。
        collapsed_categories: 折叠工具的分类目录，键为分类名，值为工具短名列表。
        collapsed_signatures: 折叠工具的签名到组件类的映射，供 explore_tools 展开使用。
        dropped_count: 因超出上限而被丢弃的工具数量。
    """

    exposed: list[type] = field(default_factory=list)
    collapsed_categories: dict[str, list[str]] = field(default_factory=dict)
    collapsed_classes: dict[str, list[type]] = field(default_factory=dict)
    collapsed_signatures: dict[str, type] = field(default_factory=dict)
    dropped_count: int = 0

    def describe_categories(self) -> str:
        """将折叠分类目录渲染为提示词文本。

        Returns:
            str: 分类目录描述；无折叠工具时返回空字符串。
        """
        if not self.collapsed_categories:
            return ""

        lines = [
            "你还有一批未展开的工具，按类别收纳如下。"
            "需要用到时先调用 explore_tools 展开对应类别，再调用具体工具："
        ]
        for category in sorted(self.collapsed_categories):
            names = self.collapsed_categories[category]
            preview = "、".join(names[:8])
            suffix = f" 等 {len(names)} 个" if len(names) > 8 else ""
            lines.append(f"- {category}：{preview}{suffix}")
        return "\n".join(lines)


def _extract_signature(component_cls: type) -> str:
    """尽力提取组件类的签名。

    Args:
        component_cls: 组件类。

    Returns:
        str: 组件签名；无法提取时返回空字符串。
    """
    getter = getattr(component_cls, "get_signature", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return ""


def is_blacklisted_component(component_cls: type, blacklist: list[str]) -> bool:
    """判断组件类是否命中工具黑名单。"""
    return bool(blacklist and signature_matches(_extract_signature(component_cls), blacklist))


def _category_of(signature: str) -> str:
    """从组件签名推导折叠分类名。

    Args:
        signature: 组件签名。

    Returns:
        str: 分类名，取签名的 plugin_name 段；无法解析时返回 ``其他``。
    """
    head = str(signature or "").split(":", 1)[0].strip()
    return head or "其他"


def _short_name(component_cls: type, signature: str) -> str:
    """提取组件的可读短名。

    Args:
        component_cls: 组件类。
        signature: 组件签名。

    Returns:
        str: 组件短名。
    """
    for attr in ("tool_name", "action_name", "agent_name"):
        value = getattr(component_cls, attr, "")
        if value:
            return str(value)
    parts = str(signature or "").split(":")
    if len(parts) == 3:
        return parts[2]
    return component_cls.__name__


def build_tool_layout(
    usables: list[type],
    *,
    always_visible: list[str],
    collapsed: list[str],
    blacklist: list[str],
    max_exposed: int,
) -> ToolLayout:
    """将可用组件列表分层，产出本轮的工具布局。

    分层规则按优先级依次判定：

    1. 命中 ``blacklist`` 的组件直接丢弃，对模型完全不可见。
    2. 命中 ``always_visible`` 的组件进入常驻层，置顶且不受上限约束。
    3. 命中 ``collapsed`` 的组件进入折叠层，仅在分类目录中露出名字。
    4. 其余组件进入普通层，按 ``max_exposed`` 余量截断。

    Args:
        usables: 候选组件类列表，通常来自 ``modify_llm_usables`` 的输出。
        always_visible: 常驻工具的签名通配模式列表。
        collapsed: 折叠工具的签名通配模式列表。
        blacklist: 禁用工具的签名通配模式列表。
        max_exposed: 暴露给 LLM 的工具数量上限；小于等于 0 表示不限制。

    Returns:
        ToolLayout: 分层结果。
    """
    layout = ToolLayout()

    pinned: list[type] = []
    normal: list[type] = []

    for component_cls in usables:
        signature = _extract_signature(component_cls)

        if blacklist and signature_matches(signature, blacklist):
            continue

        if always_visible and signature_matches(signature, always_visible):
            pinned.append(component_cls)
            continue

        if collapsed and signature_matches(signature, collapsed):
            category = _category_of(signature)
            bucket = layout.collapsed_categories.setdefault(category, [])
            class_bucket = layout.collapsed_classes.setdefault(category, [])
            if len(bucket) < MAX_TOOLS_PER_CATEGORY:
                bucket.append(_short_name(component_cls, signature))
                class_bucket.append(component_cls)
            if signature:
                layout.collapsed_signatures[signature] = component_cls
            continue

        normal.append(component_cls)

    if max_exposed <= 0:
        layout.exposed = pinned + normal
        return layout

    layout.exposed = pinned[:max_exposed]
    remaining = max_exposed - len(layout.exposed)
    if remaining > 0:
        layout.exposed.extend(normal[:remaining])
        layout.dropped_count = max(0, len(normal) - remaining)
    else:
        layout.dropped_count = len(normal)

    return layout


def build_encouragement_prompt() -> str:
    """构建鼓励工具使用的提示词片段。

    这段提示词针对 DFC 观察到的「工具调用意愿低」问题，明确告诉模型
    哪些情况下应当主动调用工具，而不是仅凭上下文臆测作答。

    Returns:
        str: 提示词文本。
    """
    return (
        "关于工具：你的文本输出会直接作为你说的话发送出去，"
        "所以工具不是用来「说话」的，而是用来真正做事和查证的。\n"
        "以下情况你应当主动调用工具，而不是凭印象作答：\n"
        "- 你不确定某个事实、时间、数据时，去查，不要编。\n"
        "- 对方提到值得记住的事（喜好、约定、近况）时，记下来。\n"
        "- 你想表达情绪、活跃气氛时，发表情包往往比纯文字更自然。\n"
        "- 你需要了解某人或某个群的情况时，去查资料而不是猜。\n"
        "你可以在说话的同时并行调用多个工具，这很正常，也更接近真人的行为。"
    )


def to_schema_safe(component_cls: type) -> dict[str, Any] | None:
    """安全地生成组件 schema。

    Args:
        component_cls: 组件类。

    Returns:
        dict[str, Any] | None: schema 字典；生成失败时返回 None。
    """
    getter = getattr(component_cls, "to_schema", None)
    if not callable(getter):
        return None
    try:
        schema = getter()
    except Exception:
        return None
    return schema if isinstance(schema, dict) else None

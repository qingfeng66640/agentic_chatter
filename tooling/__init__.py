"""工具治理模块。

提供工具分层、渐进披露与软去重能力，用于提升模型的工具调用意愿。
"""

from .dedupe import CallDeduper, DedupeDecision, DedupeMode, build_call_key
from .explore import ExploreToolsTool
from .manage_tasks import ManageTasksTool
from .registry import (
    ToolLayout,
    build_encouragement_prompt,
    build_tool_layout,
    signature_matches,
)

__all__ = [
    "CallDeduper",
    "DedupeDecision",
    "DedupeMode",
    "ExploreToolsTool",
    "ManageTasksTool",
    "ToolLayout",
    "build_call_key",
    "build_encouragement_prompt",
    "build_tool_layout",
    "signature_matches",
]

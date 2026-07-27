"""回复管线模块。

提供 agent 主循环、阶段定义与回合状态。
"""

from .loop import (
    append_no_op_nudge,
    append_tool_result,
    build_speak_segments,
    classify_calls,
    deliver_segments,
    should_continue_loop,
)
from .stages import (
    STAGE_ACT,
    STAGE_PERCEIVE,
    STAGE_PLAN,
    STAGE_REFLECT,
    PerceiveStage,
    PipelineStage,
    PlanStage,
    ReflectStage,
    resolve_stage_order,
)
from .state import TurnOutcome, TurnState

__all__ = [
    "STAGE_ACT",
    "STAGE_PERCEIVE",
    "STAGE_PLAN",
    "STAGE_REFLECT",
    "PerceiveStage",
    "PipelineStage",
    "PlanStage",
    "ReflectStage",
    "TurnOutcome",
    "TurnState",
    "append_no_op_nudge",
    "append_tool_result",
    "build_speak_segments",
    "classify_calls",
    "deliver_segments",
    "resolve_stage_order",
    "should_continue_loop",
]

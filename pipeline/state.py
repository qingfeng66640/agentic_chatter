"""回合状态定义。

``TurnState`` 承载一轮对话从开始到结束的全部中间状态，在管线各
阶段之间传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from ..decision.models import ReplyDecision
from ..tooling.dedupe import CallDeduper


class TerminationReason(StrEnum):
    """Agent 回合结束原因。"""

    STOP_REQUESTED = "stop_requested"
    END_TURN_REQUESTED = "end_turn_requested"
    TEXT_WITHOUT_TOOL = "text_without_tool"
    DUPLICATE_TEXT_WITHOUT_TOOL = "duplicate_text_without_tool"
    MAX_DUPLICATE_STREAK = "max_duplicate_streak"
    MAX_POST_SPEECH = "max_post_speech"
    MAX_NO_PROGRESS = "max_no_progress"
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True)
class LoopDecision:
    """一次 act 迭代的继续或终止裁决。"""

    should_continue: bool
    reason: TerminationReason | None = None
    wait_seconds: float | None = None
    stop_seconds: float = 0.0


@dataclass(frozen=True)
class ToolExecutionRecord:
    """一次工具调用的有界运行记录。"""

    iteration: int
    call_id: str | None
    name: str
    outcome: Literal["success", "failure", "skipped"]
    result_capture: Literal["captured", "missing", "ambiguous", "not_applicable"]
    result_preview: str
    counts_as_progress: bool


@dataclass
class TurnOutcome:
    """一轮 agent 循环的最终结果。"""

    should_wait: bool = True
    wait_seconds: float | None = None
    should_stop: bool = False
    stop_seconds: float = 0.0
    spoke: bool = False
    iterations: int = 0
    tool_calls: list[str] = field(default_factory=list)
    topic: str = ""


@dataclass
class TurnState:
    """单轮对话的运行时状态。"""

    stream_id: str
    unread_texts: str = ""
    deduper: CallDeduper = field(default_factory=CallDeduper)
    iterations: int = 0
    spoke: bool = False
    sent_texts: list[str] = field(default_factory=list)
    duplicate_text_streak: int = 0
    no_progress_iterations: int = 0
    visible_text_emissions: int = 0
    input_confirmed: bool = False
    interrupted_by_new_input: bool = False
    post_speech_iterations: int = 0
    tool_calls: list[str] = field(default_factory=list)
    tool_ledger: list[ToolExecutionRecord] = field(default_factory=list)
    termination: LoopDecision | None = None
    perceived_topic: str = ""
    plan_note: str = ""
    decision: ReplyDecision | None = None
    failed: bool = False
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_outcome(self) -> TurnOutcome:
        """将当前状态收敛为一轮的最终结果。

        终止裁决（含等待/停止时长）只来自 termination；
        should_stop 语义为「本轮以停止收尾而非等待下一轮输入」。
        """
        decision = self.termination
        should_stop = decision is not None and decision.stop_seconds > 0
        return TurnOutcome(
            should_wait=not should_stop,
            wait_seconds=decision.wait_seconds if decision is not None else None,
            should_stop=should_stop,
            stop_seconds=decision.stop_seconds if decision is not None else 0.0,
            spoke=self.spoke,
            iterations=self.iterations,
            tool_calls=list(self.tool_calls),
            topic=self.perceived_topic,
        )

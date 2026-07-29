"""回合状态定义。

``TurnState`` 承载一轮对话从开始到结束的全部中间状态，在管线各
阶段之间传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..decision.models import ReplyDecision
from ..tooling.dedupe import CallDeduper


@dataclass
class TurnOutcome:
    """一轮 agent 循环的最终结果。

    Attributes:
        should_wait: 是否进入等待。
        wait_seconds: 等待秒数；None 表示等待新消息。
        should_stop: 是否进入冷却。
        stop_seconds: 冷却秒数。
        spoke: 本轮是否真的说过话。
        iterations: 实际执行的迭代次数。
        tool_calls: 本轮执行过的工具调用名列表。
        topic: 本轮对话的一句话主题，用于写回全局心智。
    """

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
    """单轮对话的运行时状态。

    Attributes:
        stream_id: 当前聊天流 ID。
        unread_texts: 本轮处理的未读消息文本，用于情绪推断与摘要。
        deduper: 工具调用去重器。
        iterations: 已执行的迭代次数。
        spoke: 本轮是否已经说过话。
        tool_calls: 已执行的工具调用名列表。
        end_turn_requested: 模型是否已请求结束本轮。
        end_turn_seconds: 结束本轮时请求的等待秒数。
        stop_requested: 模型是否已请求冷却。
        stop_minutes: 冷却分钟数。
        perceived_topic: 感知阶段产出的话题摘要。
        plan_note: 规划阶段产出的意图说明。
        extras: 供自定义阶段存放任意数据。
    """

    stream_id: str
    unread_texts: str = ""
    deduper: CallDeduper = field(default_factory=CallDeduper)
    iterations: int = 0
    spoke: bool = False
    sent_texts: list[str] = field(default_factory=list)
    duplicate_text_streak: int = 0
    no_progress_iterations: int = 0
    visible_text_emissions: int = 0
    post_speech_iterations: int = 0
    tool_calls: list[str] = field(default_factory=list)
    end_turn_requested: bool = False
    end_turn_seconds: float = 0.0
    stop_requested: bool = False
    stop_minutes: float = 0.0
    perceived_topic: str = ""
    plan_note: str = ""
    decision: ReplyDecision | None = None
    failed: bool = False
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_outcome(self) -> TurnOutcome:
        """将当前状态收敛为一轮的最终结果。

        Returns:
            TurnOutcome: 本轮结果。
        """
        return TurnOutcome(
            should_wait=not self.stop_requested,
            wait_seconds=self.end_turn_seconds if self.end_turn_seconds > 0 else None,
            should_stop=self.stop_requested,
            stop_seconds=max(0.0, self.stop_minutes * 60.0),
            spoke=self.spoke,
            iterations=self.iterations,
            tool_calls=list(self.tool_calls),
            topic=self.perceived_topic,
        )

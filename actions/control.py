"""对话流程控制动作。

这两个动作是 agent 循环的显式出口。与 DFC 不同的是，本 chatter
**不会**在模型说完话后自动挂起——模型必须主动调用 ``end_turn``
才会进入等待。这样模型在说完话之后仍然有机会继续查证、记录或
补充行动，工具调用不会被过早截断。
"""

from __future__ import annotations

from typing import Annotated

from src.core.components.base.action import BaseAction

# 冷却常量与钳制函数实现在 pipeline.control_constants。
# 此处重新导出以保留既有导入路径，同时避免 pipeline -> actions 的循环导入。
from ..pipeline.control_constants import (
    DEFAULT_STOP_MINUTES,
    MAX_STOP_MINUTES,
    clamp_stop_minutes,
)

__all__ = [
    "DEFAULT_STOP_MINUTES",
    "MAX_STOP_MINUTES",
    "EndTurnAction",
    "StopConversationAction",
    "clamp_stop_minutes",
]


class EndTurnAction(BaseAction):
    """结束当前回合，等待对方回应。"""

    name = "end_turn"
    description = (
        "结束这一轮，等待对方的下一条消息。"
        "当你该说的已经说完、该做的已经做完时调用它。"
        "如果传入 seconds，则不管对方有没有回应，都会在指定秒数后重新唤起你，"
        "适合「等一会儿再追问」这类场景。"
    )
    primary_action = False
    associated_types = ["text"]

    # 核心动作执行前事件仍读取历史字段，保留同值别名。
    action_name = name
    action_description = description

    async def execute(
        self,
        seconds: Annotated[
            float,
            "等待秒数。留空或传 0 表示一直等到对方发新消息为止。",
        ] = 0.0,
    ) -> tuple[bool, str]:
        """登记一个回合结束点。

        实际的等待由 chatter 主循环处理，本动作只负责记录意图。

        Args:
            seconds: 主动唤醒的等待秒数；小于等于 0 表示等待新消息。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        wait_seconds = max(0.0, float(seconds or 0.0))
        if wait_seconds > 0:
            return True, f"本轮结束，{wait_seconds:.0f} 秒后会重新唤起你"
        return True, "本轮结束，等待对方的新消息"


class StopConversationAction(BaseAction):
    """结束对话并进入冷却。"""

    name = "stop_conversation"
    description = (
        "结束当前话题并进入冷却，冷却期内即使有新消息也不会唤起你。"
        "当这个话题确实聊完了、或者你暂时不想继续时使用。"
        "注意这比 end_turn 更彻底，不确定时优先用 end_turn。"
    )
    primary_action = False
    associated_types = ["text"]

    # 核心动作执行前事件仍读取历史字段，保留同值别名。
    action_name = name
    action_description = description

    async def execute(
        self,
        minutes: Annotated[
            float,
            "冷却分钟数，建议 3-15 分钟。",
        ] = DEFAULT_STOP_MINUTES,
    ) -> tuple[bool, str]:
        """登记一个对话冷却点。

        Args:
            minutes: 冷却分钟数，会被钳制到合法区间。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        cooldown = clamp_stop_minutes(minutes)
        return True, f"对话已结束；{cooldown:.0f} 分钟后，收到新消息时才会重新开启"

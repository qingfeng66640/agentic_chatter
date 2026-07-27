"""管线阶段定义。

管线由若干阶段组成，按配置中的 ``stage_order`` 依次执行。每个阶段
都是一个独立的、可被替换的处理单元，插件开发者可以通过
``PipelineService`` 注册自定义阶段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .state import TurnState

# 内置阶段名
STAGE_PERCEIVE = "perceive"
STAGE_PLAN = "plan"
STAGE_ACT = "act"
STAGE_REFLECT = "reflect"

# act 阶段是管线的核心，不可省略
REQUIRED_STAGES = (STAGE_ACT,)


class PipelineStage(ABC):
    """管线阶段基类。

    自定义阶段需继承本类并实现 :meth:`run`。阶段之间通过
    :class:`TurnState` 传递数据。

    Class Attributes:
        stage_name: 阶段名，需与配置中 ``stage_order`` 的条目对应。
    """

    stage_name: str = ""

    @abstractmethod
    async def run(self, state: TurnState, context: dict[str, Any]) -> None:
        """执行本阶段的逻辑。

        Args:
            state: 当前回合状态，阶段可读写。
            context: 运行时上下文，包含 chatter、config、chat_stream 等。
        """
        ...


class PerceiveStage(PipelineStage):
    """情境感知阶段。

    用小模型对当前对话做一句话摘要，写入 ``state.perceived_topic``。
    该摘要同时会被写回全局心智，供其他聊天流感知本流的动态。
    """

    stage_name = STAGE_PERCEIVE

    async def run(self, state: TurnState, context: dict[str, Any]) -> None:
        """产出当前对话的话题摘要。

        Args:
            state: 当前回合状态。
            context: 运行时上下文，需包含 ``summarize`` 可调用对象。
        """
        summarize = context.get("summarize")
        if not callable(summarize) or not state.unread_texts.strip():
            return

        topic = await summarize(state.unread_texts)
        if topic:
            state.perceived_topic = topic


class PlanStage(PipelineStage):
    """意图规划阶段。

    在正式行动前产出一句话的行动意图，写入 ``state.plan_note``。
    适合需要多步推理的复杂场景，简单闲聊时建议关闭。
    """

    stage_name = STAGE_PLAN

    async def run(self, state: TurnState, context: dict[str, Any]) -> None:
        """产出本轮的行动意图。

        Args:
            state: 当前回合状态。
            context: 运行时上下文，需包含 ``plan`` 可调用对象。
        """
        planner = context.get("plan")
        if not callable(planner) or not state.unread_texts.strip():
            return

        note = await planner(state.unread_texts)
        if note:
            state.plan_note = note


class ReflectStage(PipelineStage):
    """回合反思阶段。

    在本轮结束时更新情绪状态与跨流摘要。关闭本阶段会导致全局心智
    无法获得本流的最新动态。
    """

    stage_name = STAGE_REFLECT

    async def run(self, state: TurnState, context: dict[str, Any]) -> None:
        """更新情绪与全局心智。

        Args:
            state: 当前回合状态。
            context: 运行时上下文，需包含 ``reflect`` 可调用对象。
        """
        reflector = context.get("reflect")
        if callable(reflector):
            await reflector(state)


def resolve_stage_order(
    configured: list[str],
    *,
    enable_perceive: bool,
    enable_plan: bool,
    enable_reflect: bool,
) -> list[str]:
    """根据配置解析出实际要执行的阶段序列。

    未启用的内置阶段会被剔除；``act`` 阶段若被遗漏会被自动补上，
    因为没有它管线无法产出任何回复。

    Args:
        configured: 配置中声明的阶段顺序。
        enable_perceive: 是否启用感知阶段。
        enable_plan: 是否启用规划阶段。
        enable_reflect: 是否启用反思阶段。

    Returns:
        list[str]: 实际执行的阶段名序列。
    """
    toggles = {
        STAGE_PERCEIVE: enable_perceive,
        STAGE_PLAN: enable_plan,
        STAGE_REFLECT: enable_reflect,
        STAGE_ACT: True,
    }

    resolved: list[str] = []
    for name in configured or []:
        key = str(name or "").strip()
        if not key or key in resolved:
            continue
        if toggles.get(key, True):
            resolved.append(key)

    for required in REQUIRED_STAGES:
        if required not in resolved:
            resolved.append(required)

    return resolved

"""注意力模拟：走神与打断。

两种拟人化行为：

- **走神**：真人不会盯着每一条群消息都回。启用后有小概率直接略过本轮。
- **打断**：正在组织回复时对方又发了新消息，真人会改口而不是把
  过时的话发出去。启用后检测到新未读会中止当前生成并重新规划。
"""

from __future__ import annotations

import random


def should_get_distracted(
    *,
    enabled: bool,
    probability: float,
    is_direct: bool,
    rng: random.Random | None = None,
) -> bool:
    """判断本轮是否「走神」略过。

    被直接点名或私聊时永远不走神——真人可能漏看群消息，
    但不会漏看专门找自己的消息。

    Args:
        enabled: 是否启用走神。
        probability: 走神概率，有效范围 0.0-1.0。
        is_direct: 本轮是否为私聊或被 @ 的直接互动。
        rng: 随机数发生器；为 None 时使用模块级默认。

    Returns:
        bool: 为 True 表示本轮应当略过不回复。
    """
    if not enabled or is_direct:
        return False

    chance = max(0.0, min(1.0, float(probability)))
    if chance <= 0:
        return False

    generator = rng or random
    return generator.random() < chance


def should_interrupt(
    *,
    enabled: bool,
    new_unread_count: int,
) -> bool:
    """判断是否应当打断当前生成并重新规划。

    Args:
        enabled: 是否启用打断重规划。
        new_unread_count: 生成期间新到达的未读消息数。

    Returns:
        bool: 为 True 表示应当中止当前回复并基于新消息重新规划。
    """
    if not enabled:
        return False
    return new_unread_count > 0

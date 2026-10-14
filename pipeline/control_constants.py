"""对话流程控制的共享常量与工具。

本模块被 ``actions.control``（模型可见的 Action）与 ``pipeline.loop``
（agent 主循环）共同依赖，因此不放在 ``actions`` 包内：若放在那里，
``pipeline`` 导入时会连带拉起整个 ``actions`` 包，而 ``actions.dispatch``
又需要回引 ``chatter``，形成 ``chatter -> pipeline -> actions -> chatter``
的循环导入。

放在中立的 ``pipeline`` 包下可避免该环，同时保留
``actions.control.clamp_stop_minutes`` 的历史导入路径。
"""

from __future__ import annotations

# 结束对话的默认冷却分钟数
DEFAULT_STOP_MINUTES = 5.0
# 冷却时间的合法上限，防止模型设置过长导致长时间失联
MAX_STOP_MINUTES = 120.0


def clamp_stop_minutes(raw: object) -> float:
    """把模型传入的冷却分钟数解析并钳制到合法区间。

    非法输入（None/非数值）回退到默认值。

    Args:
        raw: 模型传入的原始分钟数。

    Returns:
        float: 介于 0 与 MAX_STOP_MINUTES 之间的冷却分钟数。
    """
    try:
        minutes = float(raw if raw is not None else DEFAULT_STOP_MINUTES)
    except (TypeError, ValueError):
        return DEFAULT_STOP_MINUTES
    if minutes <= 0:
        minutes = DEFAULT_STOP_MINUTES
    return min(MAX_STOP_MINUTES, minutes)

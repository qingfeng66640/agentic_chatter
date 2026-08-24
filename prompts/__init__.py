"""提示词模板模块。"""

from .templates import (
    DEFAULT_HOW_YOU_ACT,
    DEFAULT_HOW_YOU_SPEAK,
    DEFAULT_WHEN_TO_STOP,
    perceive_prompt,
    plan_prompt,
    reply_decision_prompt,
    system_prompt,
    user_prompt,
)

__all__ = [
    "DEFAULT_HOW_YOU_ACT",
    "DEFAULT_HOW_YOU_SPEAK",
    "DEFAULT_WHEN_TO_STOP",
    "perceive_prompt",
    "plan_prompt",
    "reply_decision_prompt",
    "system_prompt",
    "user_prompt",
]

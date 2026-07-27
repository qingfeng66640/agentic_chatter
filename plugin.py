"""AgenticChatter 插件入口。

注册聊天器、控制动作、工具探索器与管线服务，并在插件加载时
向提示词管理器登记本插件使用的模板。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.plugin import BasePlugin
from src.core.components.loader import register_plugin
from src.core.config import get_core_config
from src.core.prompt import get_prompt_manager, min_len, optional, wrap

from .actions import EndTurnAction, SayAction, StopConversationAction
from .chatter import AgenticChatter
from .config import AgenticChatterConfig
from .prompts import perceive_prompt, plan_prompt, system_prompt, user_prompt
from .service import PipelineService
from .tooling import ExploreToolsTool

logger = get_logger("agentic_chatter")


@register_plugin
class AgenticChatterPlugin(BasePlugin):
    """Agentic Chatter 插件。"""

    plugin_name = "agentic_chatter"
    plugin_version = "0.1.0"
    plugin_author = "MoFox Team"
    plugin_description = (
        "Agent 式回复流程聊天器：纯文本即回复、分层工具暴露、"
        "跨流全局心智、可 TOML 编排的回复管线"
    )
    configs = [AgenticChatterConfig]

    async def on_plugin_loaded(self) -> None:
        """注册本插件使用的提示词模板。"""
        try:
            personality = get_core_config().personality
        except RuntimeError:
            personality = SimpleNamespace(
                nickname="",
                personality_core="",
                personality_side="",
                identity="",
                background_story="",
                reply_style="",
                safety_guidelines=[],
                negative_behaviors=[],
            )

        get_prompt_manager().get_or_create(
            name="agentic_chatter_system",
            template=system_prompt,
            policies={
                "nickname": optional(personality.nickname),
                "personality_core": optional(personality.personality_core),
                "personality_side": optional(personality.personality_side),
                "identity": optional(personality.identity),
                "background_story": optional(personality.background_story)
                .then(min_len(10))
                .then(
                    wrap(
                        "# 背景故事\n",
                        "\n（以上是背景知识，理解并作为行动依据即可，不要在对话中复述。）",
                    )
                ),
                "reply_style": optional(personality.reply_style),
                "safety_guidelines": optional("\n".join(personality.safety_guidelines)),
                "negative_behaviors": optional("\n".join(personality.negative_behaviors)),
                "theme_guide": optional(""),
                "tool_encouragement": optional(""),
                "collapsed_tools": optional(""),
                "global_awareness": optional(""),
                "mood_guidance": optional(""),
                "system_prompt_extra": optional(""),
            },
        )

        get_prompt_manager().get_or_create(
            name="agentic_chatter_user",
            template=user_prompt,
            policies={
                "stream_name": optional("未知对话"),
                "current_time": optional("未知时间"),
                "platform": optional("未知平台"),
                "chat_type": optional("未知类型"),
                "platform_name": optional("未知"),
                "platform_id": optional("未知ID"),
                "history": optional("")
                .then(min_len(2))
                .then(wrap("# 之前的对话\n", "\n（以上是历史记录，供你了解来龙去脉，不必复述）")),
                "unreads": optional("")
                .then(min_len(2))
                .then(wrap("# 新消息\n", "")),
                "extra": optional("").then(min_len(2)).then(wrap("# 补充\n", "")),
            },
        )

        get_prompt_manager().get_or_create(
            name="agentic_chatter_perceive",
            template=perceive_prompt,
            policies={"conversation": optional("")},
        )

        get_prompt_manager().get_or_create(
            name="agentic_chatter_plan",
            template=plan_prompt,
            policies={"conversation": optional("")},
        )

        logger.info("agentic_chatter 插件已加载")

    async def on_plugin_unloaded(self) -> None:
        """插件卸载前的清理。"""
        logger.info("agentic_chatter 插件已卸载")

    def get_components(self) -> list[type]:
        """返回插件提供的全部组件类。

        Returns:
            list[type]: 组件类列表。
        """
        return [
            AgenticChatter,
            PipelineService,
            ExploreToolsTool,
            SayAction,
            EndTurnAction,
            StopConversationAction,
        ]

"""受控发送动作。

在本 chatter 中，模型的文本输出会被自动发送出去，因此绝大多数情况
不需要任何发送工具。``SayAction`` 只用于文本输出无法表达的场景：
需要引用回复某条消息，或需要 @ 某个人。
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from src.app.plugin_system.api import send_api
from src.app.plugin_system.api.adapter_api import get_bot_info_by_platform
from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.action import BaseAction
from src.core.models.message import Message, MessageType

logger = get_logger("agentic_chatter")


class SayAction(BaseAction):
    """引用回复或 @ 某人时使用的发送动作。"""

    action_name = "say"
    action_description = (
        "仅在需要「引用回复某条消息」或「在群聊中 @ 某个人」时使用。"
        "target_type 必须为 reply 或 mention，target_id 必须提供对应的消息 ID 或用户 ID。"
        "它会独立额外发送一条消息；普通回复直接输出正文即可。"
        "调用本动作时，把完整回复只写在 content 中，不要在同一响应再输出正文，"
        "也不要重发或改写本轮已输出的正文。"
    )
    primary_action = False
    associated_types = ["text"]

    async def execute(
        self,
        content: Annotated[str, "你要说的话，只写正文，不要加引号或前缀"],
        target_type: Annotated[
            Literal["reply", "mention"],
            "发送模式：reply 引用回复消息；mention 在群聊中 @ 用户。",
        ],
        target_id: Annotated[
            str,
            "必填目标 ID：reply 时填写上下文中方括号里的消息 ID；mention 时填写用户的平台 ID。",
        ],
    ) -> tuple[bool, str]:
        """发送一条引用回复或带 @ 的消息。

        Args:
            content: 消息正文。
            target_type: 发送模式，reply 表示引用消息，mention 表示 @ 群聊用户。
            target_id: 必填的消息 ID 或用户平台 ID。

        Returns:
            tuple[bool, str]: (是否发送成功, 结果说明)。
        """
        text = str(content or "").strip()
        if not text:
            return False, "content 为空，没有内容可发送"

        mode = str(target_type or "").strip().lower()
        if mode not in {"reply", "mention"}:
            return False, "target_type 必须是 reply 或 mention"

        target = str(target_id or "").strip()
        if not target:
            return False, "target_id 不能为空"

        stream_id = str(getattr(self.chat_stream, "stream_id", "") or "")
        if not stream_id:
            return False, "无法解析当前对话流，发送失败"

        if mode == "mention":
            chat_type = str(getattr(self.chat_stream, "chat_type", "") or "")
            if chat_type != "group":
                return False, "mention 仅支持群聊发送"

            platform = str(getattr(self.chat_stream, "platform", "") or "")
            bot_info = await get_bot_info_by_platform(platform)
            message = Message(
                message_id=f"action_{self.name}_{uuid4().hex}",
                content=text,
                processed_plain_text=text,
                message_type=MessageType.TEXT,
                sender_id=str((bot_info or {}).get("bot_id", "")),
                sender_name=str((bot_info or {}).get("bot_name", "Bot")),
                platform=platform,
                chat_type=chat_type,
                stream_id=stream_id,
            )
            message.extra["at_user_id"] = target
            context_message = self._get_context_message_for_target()
            if context_message is not None:
                target_group_id = context_message.extra.get("group_id")
                target_group_name = context_message.extra.get("group_name")
                if target_group_id:
                    message.extra["target_group_id"] = str(target_group_id)
                if target_group_name:
                    message.extra["target_group_name"] = str(target_group_name)

            try:
                ok = await send_api.send_message(message)
            except Exception as exc:
                logger.error(f"引用或提及发送动作失败：{exc}")
                return False, f"发送失败: {exc}"
        else:
            try:
                ok = await send_api.send_text(
                    content=text,
                    stream_id=stream_id,
                    reply_to=target,
                )
            except Exception as exc:
                logger.error(f"引用或提及发送动作失败：{exc}")
                return False, f"发送失败: {exc}"

        if not ok:
            return False, "发送失败，适配器返回未成功"

        self._last_message = text
        return True, "已发送"

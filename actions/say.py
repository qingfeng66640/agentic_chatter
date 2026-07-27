"""受控发送动作。

在本 chatter 中，模型的文本输出会被自动发送出去，因此绝大多数情况
不需要任何发送工具。``SayAction`` 只用于文本输出无法表达的场景：
需要引用回复某条消息，或需要 @ 某个人。
"""

from __future__ import annotations

from typing import Annotated

from src.app.plugin_system.api import send_api
from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.action import BaseAction

logger = get_logger("agentic_chatter")


class SayAction(BaseAction):
    """引用回复或 @ 某人时使用的发送动作。"""

    action_name = "say"
    action_description = (
        "仅在需要「引用回复某条消息」或「@某个人」时使用。"
        "普通说话不需要这个工具——你直接输出文本就会被发送出去。"
        "content 只写你要说的话本身，不要写理由或旁白。"
    )
    primary_action = False
    associated_types = ["text"]

    async def execute(
        self,
        content: Annotated[str, "你要说的话，只写正文，不要加引号或前缀"],
        reply_to: Annotated[
            str,
            "要引用回复的消息 ID，取自消息记录中方括号里的那个 ID。不引用时留空。",
        ] = "",
        at: Annotated[
            str,
            "要 @ 的对象的平台 ID。不需要 @ 时留空。",
        ] = "",
    ) -> tuple[bool, str]:
        """发送一条带引用或 @ 的消息。

        Args:
            content: 消息正文。
            reply_to: 要引用的消息 ID；留空表示不引用。
            at: 要 @ 的平台 ID；留空表示不 @。

        Returns:
            tuple[bool, str]: (是否发送成功, 结果说明)。
        """
        text = str(content or "").strip()
        if not text:
            return False, "content 为空，没有内容可发送"

        stream_id = getattr(self.chat_stream, "stream_id", "")
        if not stream_id:
            return False, "无法解析当前对话流，发送失败"

        at_target = str(at or "").strip()
        if at_target:
            text = f"@{at_target} {text}"

        try:
            ok = await send_api.send_text(
                content=text,
                stream_id=stream_id,
                reply_to=str(reply_to or "").strip() or None,
            )
        except Exception as exc:
            logger.error(f"say 动作发送失败: {exc}")
            return False, f"发送失败: {exc}"

        if not ok:
            return False, "发送失败，适配器返回未成功"

        self._last_message = text
        return True, "已发送"

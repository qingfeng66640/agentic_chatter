"""技能目录管理员刷新命令。"""

from __future__ import annotations

from src.app.plugin_system.api.send_api import send_text
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel

from .skills import refresh_skills


class AgenticSkillCommand(BaseCommand):
    """仅允许所有者重新扫描本插件的技能目录。"""

    name = "agentic_skill"
    description = "刷新 Agentic Chatter 的只读技能目录"
    permission_level = PermissionLevel.OWNER

    @cmd_route("refresh")
    async def handle_refresh(self) -> tuple[bool, str]:
        """从当前配置路径重新扫描技能。"""
        config = self.plugin.config
        section = getattr(config, "skills", None)
        count = refresh_skills(
            list(getattr(section, "paths", ["skill"])),
            enabled=bool(getattr(section, "enabled", True)),
            max_description_chars=int(getattr(section, "max_description_chars", 500)),
        )
        message = f"技能目录已刷新：{count} 个。"
        await send_text(message, stream_id=self.stream_id)
        return True, message

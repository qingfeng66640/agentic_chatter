"""任务工具权限策略。"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum


class ToolPermission(StrEnum):
    """工具调用权限结果。"""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass(slots=True)
class ToolPolicy:
    """以白名单为主的任务工具权限策略。"""

    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    require_confirmation_tools: tuple[str, ...] = ()
    default_allow: bool = False
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    @staticmethod
    def _matches(name: str, patterns: tuple[str, ...]) -> bool:
        """判断工具名是否匹配通配规则。"""
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns if pattern)

    def permission(self, name: str) -> ToolPermission:
        """计算工具调用权限。"""
        normalized = str(name or "").strip()
        if self._matches(normalized, self.denied_tools):
            return ToolPermission.DENY
        if self._matches(normalized, self.require_confirmation_tools):
            return ToolPermission.CONFIRM
        if self.allowed_tools:
            return ToolPermission.ALLOW if self._matches(normalized, self.allowed_tools) else ToolPermission.DENY
        return ToolPermission.ALLOW if self.default_allow else ToolPermission.DENY

    def check_and_count(self, name: str, maximum: int) -> bool:
        """记录工具调用并判断是否超出单签名上限。"""
        key = str(name or "")
        count = self._counts.get(key, 0)
        if maximum > 0 and count >= maximum:
            return False
        self._counts[key] = count + 1
        return True

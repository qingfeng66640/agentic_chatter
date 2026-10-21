"""定时任务管理工具。

主 Agent 的 LLM 通过以下四个工具管理定时任务：

- ``cron_create``：创建定时任务
- ``cron_read``：查询任务列表 / 详情 / 运行历史
- ``cron_update``：修改任务或暂停恢复
- ``cron_delete``：删除任务（保留运行历史）

设计对齐业界 cron 的成熟做法：``schedule`` / ``after`` / ``at`` 三选一互斥、
时区显式声明、会话策略创建后固定。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any

from src.core.components.base.tool import BaseTool

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from ..task_runtime.schedule import ScheduleManager

# 任务状态的中文标签
_STATE_LABELS = {
    "active": "运行中",
    "completed": "已完成",
    "expired": "已过期",
    "cancelled": "已删除",
}

# 会话策略说明
_SESSION_HINT = {
    "new": "每次触发都是全新上下文，看不到上次结果",
    "session": "复用该任务专属上下文，后续运行可引用此前结果",
}


def _get_manager() -> "ScheduleManager | None":
    """获取全局定时任务管理器。"""
    from ..task_runtime.schedule import get_schedule_manager

    return get_schedule_manager()


def _format_time(value: Any) -> str:
    """把毫秒时间戳渲染为可读文本。"""
    if not value:
        return "无"
    moment = datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).astimezone()
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _describe_schedule(record: dict[str, Any]) -> str:
    """渲染任务的调度描述。"""
    kind = str(record.get("schedule_kind", ""))
    if kind == "cron":
        return f"cron `{record.get('cron_expression', '')}`（时区 {record.get('timezone_name') or '本地'}）"
    if kind == "after":
        seconds = int(float(record.get("duration_seconds", 0) or 0))
        return f"一次性，延迟 {seconds // 60} 分钟"
    return f"一次性，{_format_time(record.get('absolute_at_ms'))}"


def _format_detail(record: dict[str, Any]) -> str:
    """渲染单个任务的详情。"""
    lines = [
        f"定时任务「{record.get('cron_name', '')}」（ID {str(record.get('cron_id', ''))[:8]}）",
        f"指令：{str(record.get('prompt', ''))[:120]}",
        f"调度：{_describe_schedule(record)}",
        f"状态：{_STATE_LABELS.get(str(record.get('state', '')), record.get('state', ''))}"
        f"{'（已暂停）' if not record.get('enabled') else ''}",
        f"会话策略：{_SESSION_HINT.get(str(record.get('session_mode', 'new')), '未知')}",
        f"下次触发：{_format_time(record.get('next_run_at_ms'))}",
        f"累计运行：{record.get('run_count', 0)} 次",
    ]
    return "\n".join(lines)


class CreateCronTool(BaseTool):
    """创建定时任务。"""

    tool_name = "cron_create"
    tool_description = (
        "创建一个定时任务，到点自动执行一段指令。"
        "schedule/after/at 三者必须且只能提供一个："
        "schedule 是五段 cron 表达式（如 0 9 * * 1-5 表示工作日早九点）；"
        "after 是一次性相对延迟（如 30m、2h）；at 是一次性绝对时间。"
        "timezone 建议显式声明，如 Asia/Shanghai，否则按服务器本地时区。"
        "session_mode=new 表示每次全新上下文；session 表示复用该任务专属上下文，"
        "后续运行可引用此前结果（session_target 留空则由任务自行累积历史，me 非法）。"
        "用户说“每天早上…”“每隔一小时…”“30分钟后…”时调用。"
    )

    async def execute(
        self,
        cron_name: Annotated[str, "任务标题，例如「工作日晨报」"],
        prompt: Annotated[str, "到点执行的任务指令"],
        schedule: Annotated[str, "五段 cron 表达式，例如 0 9 * * 1-5"] = "",
        after: Annotated[str, "一次性相对延迟，例如 30m、2h"] = "",
        at: Annotated[str, "一次性绝对时间，Unix 毫秒时间戳或日期时间字符串"] = "",
        timezone_name: Annotated[str, "IANA 时区名，例如 Asia/Shanghai"] = "",
        session_mode: Annotated[str, "会话策略：new（每次新上下文，默认）或 session（复用专属上下文）"] = "new",
        session_target: Annotated[str, "session 模式下要复用的历史标识；留空表示由任务自行累积，me 非法"] = "",
    ) -> tuple[bool, str]:
        """创建定时任务。

        Args:
            cron_name: 任务标题。
            prompt: 任务指令。
            schedule: 五段 cron 表达式。
            after: 一次性相对延迟。
            at: 一次性绝对时间。
            timezone_name: IANA 时区名。
            session_mode: 会话策略。
            session_target: session 模式下的历史标识。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        manager = _get_manager()
        if manager is None:
            return False, "定时任务功能尚未启用"

        stream_id = self.get_current_stream_id()
        if not stream_id:
            return False, "无法确定当前聊天流，暂不能创建定时任务"

        from ..task_runtime.schedule import ScheduleError

        try:
            record = manager.create(
                cron_name=cron_name,
                prompt=prompt,
                stream_id=stream_id,
                session_mode=session_mode,
                session_target=session_target,
                schedule=schedule or None,
                after=after or None,
                at=at or None,
                timezone_name=timezone_name or None,
            )
        except ScheduleError as exc:
            return False, f"创建失败：{exc}"

        return True, _format_detail(record)


class ReadCronTool(BaseTool):
    """查询定时任务。"""

    tool_name = "cron_read"
    tool_description = (
        "查询定时任务。view=list 列出全部任务；view=detail 查询单个任务详情；"
        "view=runs 查询某任务的运行历史。detail 与 runs 需要 cron_ref，"
        "可填任务ID或唯一名称前缀。用户问“我设了哪些定时任务/下次什么时候跑”时调用。"
    )

    async def execute(
        self,
        view: Annotated[str, "视图：list（默认）/ detail / runs"] = "list",
        cron_ref: Annotated[str, "任务ID或唯一名称前缀，detail 与 runs 必填"] = "",
        limit: Annotated[int, "runs 视图返回的条数上限，默认 10"] = 10,
    ) -> tuple[bool, str]:
        """查询定时任务。

        Args:
            view: list / detail / runs。
            cron_ref: 任务引用。
            limit: 运行历史条数上限。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        manager = _get_manager()
        if manager is None:
            return False, "定时任务功能尚未启用"

        from ..task_runtime.schedule import ScheduleError

        action = str(view or "").strip().lower() or "list"
        if action not in ("list", "detail", "runs"):
            return False, f"未知的 view：{view}（可用 list / detail / runs）"
        if action == "list":
            records = manager.read_list()
            if not records:
                return True, "当前没有定时任务。"
            return True, "\n\n".join(_format_detail(record) for record in records)

        if not str(cron_ref or "").strip():
            return False, f"view={action} 需要提供 cron_ref"
        try:
            record = manager.resolve_id(cron_ref)
        except ScheduleError as exc:
            return False, str(exc)
        if record is None:
            return False, f"没有找到匹配「{cron_ref}」的定时任务"

        if action == "detail":
            return True, _format_detail(record)
        runs = manager.read_runs(str(record["cron_id"]), limit=max(1, int(limit)))
        if not runs:
            return True, "该定时任务还没有运行记录。"
        return True, "\n".join(runs)


class UpdateCronTool(BaseTool):
    """修改定时任务。"""

    tool_name = "cron_update"
    tool_description = (
        "修改定时任务。可改标题、指令、调度表达式、时区，或用 enabled 暂停/恢复。"
        "会话策略在创建后固定，无法通过本工具更改。"
        "用户说“把每天早上的任务改成九点”“暂停那个定时任务”时调用。"
    )

    async def execute(
        self,
        cron_ref: Annotated[str, "任务ID或唯一名称前缀"],
        cron_name: Annotated[str, "新的任务标题，留空表示不改"] = "",
        prompt: Annotated[str, "新的任务指令，留空表示不改"] = "",
        schedule: Annotated[str, "新的五段 cron 表达式，留空表示不改"] = "",
        after: Annotated[str, "新的相对延迟，留空表示不改"] = "",
        at: Annotated[str, "新的绝对时间，留空表示不改"] = "",
        timezone_name: Annotated[str, "新的时区名，留空表示不改"] = "",
        enabled: Annotated[str, "enabled=false 暂停，enabled=true 恢复；留空表示不改"] = "",
    ) -> tuple[bool, str]:
        """修改定时任务。

        Args:
            cron_ref: 任务引用。
            cron_name: 新标题。
            prompt: 新指令。
            schedule: 新 cron 表达式。
            after: 新相对延迟。
            at: 新绝对时间。
            timezone_name: 新时区。
            enabled: ``false`` 暂停、``true`` 恢复。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        manager = _get_manager()
        if manager is None:
            return False, "定时任务功能尚未启用"

        from ..task_runtime.schedule import ScheduleError

        try:
            record = manager.resolve_id(cron_ref)
        except ScheduleError as exc:
            return False, str(exc)
        if record is None:
            return False, f"没有找到匹配「{cron_ref}」的定时任务"

        enabled_value: bool | None = None
        enabled_text = str(enabled or "").strip().lower()
        if enabled_text:
            if enabled_text in ("false", "0", "off", "no", "暂停", "关"):
                enabled_value = False
            elif enabled_text in ("true", "1", "on", "yes", "恢复", "开"):
                enabled_value = True
            else:
                return False, f"enabled 只接受 true/false，收到：{enabled}"

        try:
            updated = manager.update(
                str(record["cron_id"]),
                cron_name=cron_name or None,
                prompt=prompt or None,
                schedule=schedule or None,
                after=after or None,
                at=at or None,
                timezone_name=timezone_name or None,
                enabled=enabled_value,
            )
        except ScheduleError as exc:
            return False, f"修改失败：{exc}"

        if updated is None:
            return False, "任务不存在"
        return True, _format_detail(updated)


class DeleteCronTool(BaseTool):
    """删除定时任务。"""

    tool_name = "cron_delete"
    tool_description = (
        "删除定时任务，停止其后续运行。运行历史会保留，仍可用 cron_read 的 runs 视图查询。"
        "用户明确要求取消或删除某个定时任务时调用。"
    )

    async def execute(
        self,
        cron_ref: Annotated[str, "任务ID或唯一名称前缀"],
    ) -> tuple[bool, str]:
        """删除定时任务。

        Args:
            cron_ref: 任务引用。

        Returns:
            tuple[bool, str]: (是否成功, 结果说明)。
        """
        manager = _get_manager()
        if manager is None:
            return False, "定时任务功能尚未启用"

        from ..task_runtime.schedule import ScheduleError

        try:
            record = manager.resolve_id(cron_ref)
        except ScheduleError as exc:
            return False, str(exc)
        if record is None:
            return False, f"没有找到匹配「{cron_ref}」的定时任务"

        manager.delete(str(record["cron_id"]))
        return True, f"已删除定时任务「{record.get('cron_name', '')}」，运行历史仍可通过 cron_read 的 runs 视图查看。"

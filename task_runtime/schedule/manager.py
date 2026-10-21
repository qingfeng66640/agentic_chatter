"""定时任务管理器。

对外提供创建 / 查询 / 修改 / 删除与后台扫描循环。扫描循环注册到内核
``UnifiedScheduler`` 上，每分钟唤醒一次，按 SQLite 中记录的 ``next_run_at_ms``
认领到期任务并提交到 ``task_runtime``。

可靠性约定（对齐业界 cron 的成熟做法）：

- **catch-up 不回放**：宕机期间错过多个触发点时只补触发一次，随后直接
  重同步到下一个未来时刻，避免补跑放大成本。
- **失败不禁用**：单次运行失败只记录，下一次触发天然是重试。
- **单次认领**：运行记录与到期认领在同一事务内写入，配合唯一约束
  保证同一触发点不会被执行两次。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

from ..models import TaskStatus
from .expression import Schedule, ScheduleError, parse_schedule
from .store import (
    RUN_DELIVERED,
    RUN_FAILED,
    STATE_ACTIVE,
    STATE_COMPLETED,
    STATE_EXPIRED,
    ScheduleStore,
    describe_run,
)

logger = get_logger("agentic_chatter")

# 扫描间隔（秒）；内核 UnifiedScheduler 以 recurring 方式调用
SCAN_INTERVAL_SECONDS = 60

# 固定线程注入上下文时保留的历史条数
THREAD_HISTORY_ENTRIES = 6

# 一次性任务触发后停留的状态
_ONESHOT_COMPLETED = STATE_COMPLETED


def _now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    import time

    return int(time.time() * 1000)


def _parse_again(record: dict[str, Any]) -> Schedule:
    """从存储记录还原调度定义。"""
    kind = str(record.get("schedule_kind", ""))
    if kind == "cron":
        return Schedule(
            kind="cron",
            cron_expression=str(record.get("cron_expression", "")),
            timezone_name=str(record.get("timezone_name", "")),
        )
    if kind == "after":
        return Schedule(kind="after", duration_seconds=float(record.get("duration_seconds", 0.0)))
    return Schedule(kind="at", absolute_at_ms=int(record.get("absolute_at_ms", 0)))


class ScheduleManager:
    """定时任务管理器。"""

    def __init__(
        self,
        store: ScheduleStore,
        submitter: Callable[..., Any] | None = None,
    ) -> None:
        """初始化管理器。

        Args:
            store: SQLite 存储。
            submitter: 提交任务的回调，签名为 ``submitter(stream_id, prompt) -> task_id``。
        """
        self._store = store
        self._submitter = submitter
        self._scanning = False

    def recover_stale_claims(self) -> int:
        """恢复上次进程中断遗留的已认领任务。

        Returns:
            int: 恢复的任务数量。
        """
        try:
            recovered = self._store.reset_stale_claims()
        except Exception as exc:  # noqa: BLE001 - 恢复失败不应阻断启动
            logger.warning(f"定时任务认领恢复失败：{exc}")
            return 0
        if recovered:
            logger.warning(f"已恢复 {recovered} 个中断遗留的定时任务认领")
        return recovered

    # ==================== 外部接口 ====================

    def create(
        self,
        *,
        cron_name: str,
        prompt: str,
        stream_id: str,
        session_mode: str = "new",
        session_target: str = "",
        schedule: str | None = None,
        after: str | None = None,
        at: str | None = None,
        timezone_name: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """创建一个定时任务。

        Args:
            cron_name: 任务标题。
            prompt: 任务指令。
            stream_id: 目标聊天流。
            session_mode: ``new`` 或 ``session``。
            session_target: 固定线程模式下的绑定标识，可留空。
            schedule: 五段 cron 表达式。
            after: 相对延迟。
            at: 一次性绝对时间。
            timezone_name: IANA 时区名。
            enabled: 是否立即启用。

        Returns:
            dict[str, Any]: 新建的记录。

        Raises:
            ScheduleError: 参数非法时抛出。
        """
        title = str(cron_name or "").strip()
        instruction = str(prompt or "").strip()
        if not title:
            raise ScheduleError("cron_name 不能为空")
        if not instruction:
            raise ScheduleError("prompt 不能为空")
        target_stream = str(stream_id or "").strip()
        if not target_stream:
            raise ScheduleError("缺少目标聊天流")

        normalized_mode = _normalize_session_mode(session_mode)
        target = str(session_target or "").strip()
        if normalized_mode == "session" and target.lower() == "me":
            raise ScheduleError("me 不是合法的 session 目标")
        if normalized_mode == "new":
            target = ""

        parsed, first_run = parse_schedule(
            schedule=schedule, after=after, at=at, timezone_name=timezone_name
        )
        next_run_at_ms = int(first_run.timestamp() * 1000) if first_run else None
        record = self._store.create(
            {
                "cron_id": uuid.uuid4().hex,
                "cron_name": title,
                "prompt": instruction,
                "stream_id": target_stream,
                "session_mode": normalized_mode,
                "session_target": target,
                "schedule_kind": parsed.kind,
                "cron_expression": parsed.cron_expression,
                "duration_seconds": parsed.duration_seconds,
                "absolute_at_ms": parsed.absolute_at_ms,
                "timezone_name": parsed.timezone_name,
                "enabled": bool(enabled),
                "state": STATE_ACTIVE,
                "next_run_at_ms": next_run_at_ms,
            }
        )
        logger.info(
            f"已创建定时任务「{title}」，下次触发 {_format_ms(next_run_at_ms)}"
        )
        return record

    def read_list(self, *, limit: int = 50) -> tuple[dict[str, Any], ...]:
        """列出全部定时任务。"""
        return self._store.list_all()

    def read_detail(self, cron_id: str) -> dict[str, Any] | None:
        """读取单个定时任务。"""
        return self._store.get(cron_id)

    def read_runs(self, cron_id: str, *, limit: int = 20) -> tuple[str, ...]:
        """读取某任务的运行历史（可读文本）。"""
        return tuple(describe_run(run) for run in self._store.list_runs(cron_id, limit=limit))

    def update(
        self,
        cron_id: str,
        *,
        cron_name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        after: str | None = None,
        at: str | None = None,
        timezone_name: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        """修改定时任务。

        会话策略在创建后固定，此处不可更改。

        Raises:
            ScheduleError: 参数非法时抛出。
        """
        record = self._store.get(cron_id)
        if record is None:
            return None

        changes: dict[str, Any] = {}
        if cron_name is not None:
            cleaned = str(cron_name).strip()
            if not cleaned:
                raise ScheduleError("cron_name 不能为空")
            changes["cron_name"] = cleaned
        if prompt is not None:
            cleaned = str(prompt).strip()
            if not cleaned:
                raise ScheduleError("prompt 不能为空")
            changes["prompt"] = cleaned

        timing_changed = any(
            value is not None for value in (schedule, after, at)
        ) or timezone_name is not None
        if timing_changed:
            if sum(
                1 for value in (schedule, after, at) if str(value or "").strip()
            ) > 1:
                raise ScheduleError("schedule、after、at 必须且只能提供一个")
            # 未改动的那一段沿用原值，保证三选一语义不被破坏
            effective_schedule = schedule
            effective_after = after
            effective_at = at
            if schedule is None and after is None and at is None:
                kind = str(record.get("schedule_kind", ""))
                if kind == "cron":
                    effective_schedule = str(record.get("cron_expression", ""))
                elif kind == "after":
                    effective_after = f"{int(float(record.get('duration_seconds', 0)))}s"
                else:
                    effective_at = str(record.get("absolute_at_ms", 0))
            parsed, first_run = parse_schedule(
                schedule=effective_schedule,
                after=effective_after,
                at=effective_at,
                timezone_name=(
                    timezone_name
                    if timezone_name is not None
                    else str(record.get("timezone_name", "")) or None
                ),
            )
            changes.update(
                {
                    "schedule_kind": parsed.kind,
                    "cron_expression": parsed.cron_expression,
                    "duration_seconds": parsed.duration_seconds,
                    "absolute_at_ms": parsed.absolute_at_ms,
                    "timezone_name": parsed.timezone_name,
                    "next_run_at_ms": int(first_run.timestamp() * 1000) if first_run else None,
                }
            )

        if enabled is not None:
            changes["enabled"] = bool(enabled)
            if not enabled:
                changes["next_run_at_ms"] = None
            elif record.get("next_run_at_ms") is None:
                # 恢复时若已无待执行时刻，需要重算
                parsed = _parse_again({**record, **changes})
                moment = datetime.now().astimezone()
                nxt = parsed.next_after(moment)
                changes["next_run_at_ms"] = int(nxt.timestamp() * 1000) if nxt else None

        if timing_changed:
            # 表达式变更时递增世代，防止调度器使用过期时间表
            self._store.update(cron_id, changes)
            self._store.bump_generation(cron_id)
        elif changes:
            self._store.update(cron_id, changes)
        return self._store.get(cron_id)

    def delete(self, cron_id: str) -> bool:
        """删除定时任务，保留运行历史。"""
        return self._store.soft_delete(cron_id)

    def resolve_id(self, reference: str) -> dict[str, Any] | None:
        """按完整 ID 或唯一名称前缀解析任务。"""
        text = str(reference or "").strip()
        if not text:
            return None
        exact = self._store.get(text)
        if exact is not None:
            return exact
        matches = [
            record
            for record in self._store.list_all()
            if str(record.get("cron_name", "")).startswith(text)
            or str(record.get("cron_id", "")).startswith(text)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ScheduleError(
                f"「{text}」匹配到多个任务，请改用完整 cron_id"
            )
        return None

    # ==================== 调度循环 ====================

    async def scan_once(self) -> int:
        """扫描一次到期任务并提交。

        Returns:
            int: 本次提交的运行数量。
        """
        self._scanning = True
        try:
            claimed = self._store.claim_due(_now_ms())
        except Exception as exc:  # noqa: BLE001 - 存储异常不应中断循环
            logger.error(f"定时任务扫描失败：{exc}")
            return 0
        finally:
            self._scanning = False

        for record in claimed:
            await self._dispatch(record)
        return len(claimed)

    async def run_forever(self, *, interval: float = SCAN_INTERVAL_SECONDS) -> None:
        """在后台循环中持续扫描。

        Args:
            interval: 扫描间隔秒数。
        """
        logger.info(f"定时任务调度循环已启动，扫描间隔 {interval} 秒")
        import asyncio

        try:
            while True:
                await self.scan_once()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:  # pragma: no cover - 取消路径
            logger.info("定时任务调度循环已停止")
            raise

    async def _dispatch(self, record: dict[str, Any]) -> None:
        """把一次到期触发提交为实际任务。"""
        cron_id = str(record.get("cron_id", ""))
        stream_id = str(record.get("stream_id", ""))
        prompt = str(record.get("prompt", ""))
        one_shot = str(record.get("schedule_kind", "")) != "cron"

        self._reschedule(record)

        if self._submitter is None:
            logger.warning(f"定时任务「{record.get('cron_name')}」缺少提交回调，已跳过")
            self._store.mark_run(cron_id, status=RUN_FAILED, error="缺少提交回调")
            return

        try:
            context = self._build_context(record) if self._uses_thread(record) else ""
            task_id = await self._invoke_submitter(
                stream_id=stream_id,
                prompt=prompt,
                context=context,
                cron_id=cron_id,
                record=record,
            )
        except Exception as exc:  # noqa: BLE001 - 单次失败不禁用
            logger.error(f"定时任务「{record.get('cron_name')}」执行失败：{exc}")
            self._store.mark_run(cron_id, status=RUN_FAILED, error=str(exc)[:300])
            return

        if one_shot:
            self._store.finish_state(cron_id, _ONESHOT_COMPLETED)
        if task_id:
            self._store.mark_run(
                cron_id, status=RUN_DELIVERED, task_id=str(task_id)
            )
            if self._uses_thread(record):
                self._store.append_thread_entry(cron_id, prompt, max_entries=THREAD_HISTORY_ENTRIES)

    async def _invoke_submitter(
        self,
        *,
        stream_id: str,
        prompt: str,
        context: str,
        cron_id: str,
        record: dict[str, Any],
    ) -> str:
        """调用注入的提交回调，兼容同步与异步实现。"""
        import inspect

        submitter = self._submitter
        assert submitter is not None  # 已由调用方保证
        outcome = submitter(
            stream_id=stream_id,
            prompt=prompt,
            context=context,
            cron_id=cron_id,
            record=record,
        )
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return str(outcome or "")

    def _uses_thread(self, record: dict[str, Any]) -> bool:
        """是否处于固定线程（有状态）模式。"""
        return str(record.get("session_mode", "new")) != "new"

    def _build_context(self, record: dict[str, Any]) -> str:
        """拼接固定线程模式的历史上下文。"""
        cron_id = str(record.get("cron_id", ""))
        entries = self._store.list_thread_entries(cron_id, limit=THREAD_HISTORY_ENTRIES)
        if not entries:
            return ""
        lines = [
            f"- [{_format_ms(entry.get('created_at_ms'))}] {entry.get('summary', '')}"
            for entry in entries
        ]
        return "以下是本任务此前运行的记录：\n" + "\n".join(lines)

    def _reschedule(self, record: dict[str, Any]) -> None:
        """推进下一次触发时刻。

        循环任务按 cron 重算；过期的一次性任务置为 expired。
        """
        cron_id = str(record.get("cron_id", ""))
        if str(record.get("schedule_kind", "")) != "cron":
            # 一次性任务触发后不再排程；已过期则标记 expired
            self._store.finish_state(
                cron_id, STATE_EXPIRED if _is_expired(record) else _ONESHOT_COMPLETED
            )
            return

        parsed = _parse_again(record)
        moment = datetime.now().astimezone()
        following = parsed.next_after(moment)
        if following is None:
            # 表达式在四年内不再命中，视为不可达
            self._store.finish_state(cron_id, STATE_COMPLETED)
            return
        # 直接跳到下一个未来点：错过的触发点不回放
        self._store.set_next_run(cron_id, int(following.timestamp() * 1000))

    def is_busy(self) -> bool:
        """扫描是否正在进行中。"""
        return self._scanning

    def thread_context(self, cron_id: str) -> str:
        """对外暴露固定线程历史，供外部查询。"""
        return self._build_context({"cron_id": cron_id})


def _is_expired(record: dict[str, Any]) -> bool:
    """一次性任务是否已过期。"""
    target = int(record.get("absolute_at_ms", 0) or 0)
    if target:
        return target < _now_ms()
    duration = float(record.get("duration_seconds", 0.0) or 0.0)
    return duration <= 0


# ==================== 全局单例 ====================

_manager: ScheduleManager | None = None


def init_schedule_manager(
    store: ScheduleStore, submitter: Callable[..., Any] | None = None
) -> ScheduleManager:
    """初始化全局定时任务管理器。

    Args:
        store: SQLite 存储。
        submitter: 提交任务的回调。

    Returns:
        ScheduleManager: 全局管理器。
    """
    global _manager
    _manager = ScheduleManager(store, submitter=submitter)
    return _manager


def get_schedule_manager() -> ScheduleManager | None:
    """获取全局定时任务管理器。

    Returns:
        ScheduleManager | None: 未初始化时返回 None。
    """
    return _manager


def shutdown_schedule_manager() -> None:
    """释放全局管理器引用。"""
    global _manager
    _manager = None



def _normalize_session_mode(mode: str) -> str:
    """归一化会话策略名称。"""
    text = str(mode or "new").strip().lower()
    if text in {"new", ""}:
        return "new"
    if text in {"session", "sessionid", "session_id", "fixed"}:
        return "session"
    raise ScheduleError(f"未知的会话策略：{mode}")


def _format_ms(value: int | None) -> str:
    """把毫秒时间戳渲染为可读文本。"""
    if not value:
        return "无"
    moment = datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).astimezone()
    return moment.strftime("%Y-%m-%d %H:%M:%S")


# 供调度循环使用的默认间隔别名
DEFAULT_INTERVAL = timedelta(seconds=SCAN_INTERVAL_SECONDS).total_seconds()

# 暴露任务状态枚举便于外部对齐语义
TASK_STATUS_VALUES = tuple(status.value for status in TaskStatus)

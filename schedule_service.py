"""定时任务服务：把调度触发接到复杂任务运行时。

定时任务到点后不在这里直接跑 LLM，而是复用 :func:`service._submit_task`
的既有通路：在目标聊天流创建一个受预算与工具策略约束的复杂任务，
由 task_runtime 在后台执行，行为与用户手动触发的任务完全一致。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import ROLE, LLMPayload, Text
from src.kernel.scheduler import TriggerType, get_unified_scheduler

from .task_runtime.schedule import (
    SCAN_INTERVAL_SECONDS,
    ScheduleStore,
    get_schedule_manager,
    init_schedule_manager,
    shutdown_schedule_manager,
)

logger = get_logger("agentic_chatter")

# 调度循环注册到内核调度器时使用的任务名
SCHEDULER_TASK_NAME = "agentic_chatter_schedule_scan"


def build_schedule_directory(config: Any) -> str | None:
    """按插件配置解析定时任务数据库目录。

    显式配置的 ``database_directory`` 直接作为数据库所在目录使用；
    留空时才回落到任务检查点目录下的 ``schedules`` 子目录，避免与任务
    检查点 JSON 混放。

    Args:
        config: 插件配置对象。

    Returns:
        str | None: 数据库目录；未启用时返回 None。
    """
    schedule_config = getattr(config, "schedule", None)
    if schedule_config is None:
        return None
    if not bool(getattr(schedule_config, "enabled", True)):
        return None
    directory = str(getattr(schedule_config, "database_directory", "") or "").strip()
    if directory:
        return directory
    tasks_config = getattr(config, "tasks", None)
    checkpoint_dir = str(
        getattr(tasks_config, "checkpoint_directory", "") or ""
    ).strip()
    if not checkpoint_dir:
        return None
    import os

    return os.path.join(checkpoint_dir, "schedules")


def _make_submitter(plugin: Any) -> Any:
    """构造把定时触发转为复杂任务的回调。"""

    async def submit(
        *,
        stream_id: str,
        prompt: str,
        context: str,
        cron_id: str,
        record: dict[str, Any],
    ) -> str:
        """提交一次定时触发的任务。

        Args:
            stream_id: 目标聊天流。
            prompt: 任务指令。
            context: 固定线程模式下的历史上下文，可能为空。
            cron_id: 定时任务 ID。
            record: 任务定义记录。

        Returns:
            str: 任务 ID；提交失败时为空串。
        """
        from .chatter import AgenticChatter
        from .task_runtime import TaskRequest
        from .task_runtime.coordinator import start_task
        from .task_runtime.models import TaskType
        from .task_runtime.runtime import get_task_runtime_manager

        manager = get_task_runtime_manager()
        tasks_config = getattr(plugin, "config", None)
        cap = max(
            1,
            int(
                getattr(getattr(tasks_config, "tasks", None), "max_concurrent_tasks", 2)
                or 2
            ),
        )
        if len(manager.get_active_tasks(stream_id)) >= cap:
            logger.warning(f"定时任务触发被跳过：该聊天流活动任务数已达上限（{cap}）")
            return ""

        try:
            runtime = manager.create(
                stream_id,
                prompt,
                task_type=TaskType.GENERAL,
            )
        except ValueError as exc:
            logger.warning(f"定时任务触发失败：{exc}")
            return ""
        runtime.start()

        # 固定线程模式：把此前运行记录作为初始上下文注入本次执行
        initial_context: tuple[LLMPayload, ...] = ()
        if context:
            initial_context = (LLMPayload(ROLE.USER, Text(context)),)

        chatter = AgenticChatter(stream_id, plugin)
        start_task(
            chatter,
            runtime,
            TaskRequest(
                objective=prompt,
                metadata={
                    "cron_id": cron_id,
                    "cron_name": str(record.get("cron_name", "")),
                },
                initial_context=initial_context,
                report_events=True,
            ),
            None,
        )
        logger.info(
            f"定时任务已触发 事件=定时任务已触发 "
            f"任务ID={runtime.state.task_id} cron={str(cron_id)[:8]} "
            f"有状态={bool(context)}"
        )
        return runtime.state.task_id

    return submit


async def start_schedule_service(plugin: Any) -> None:
    """启动定时任务调度。

    Args:
        plugin: 插件实例。
    """
    directory = build_schedule_directory(getattr(plugin, "config", None))
    if not directory:
        logger.info("定时任务未启用（未配置数据库目录）")
        return

    try:
        store = ScheduleStore(directory)
    except Exception as exc:  # noqa: BLE001 - 存储不可用不应阻断插件加载
        logger.error(f"定时任务数据库初始化失败，功能未启用：{exc}")
        return

    manager = init_schedule_manager(store, submitter=_make_submitter(plugin))
    manager.recover_stale_claims()

    scheduler = get_unified_scheduler()
    try:
        await scheduler.create_schedule(
            callback=manager.scan_once,
            trigger_type=TriggerType.TIME,
            trigger_config={"interval_seconds": SCAN_INTERVAL_SECONDS},
            is_recurring=True,
            task_name=SCHEDULER_TASK_NAME,
        )
    except Exception as exc:  # noqa: BLE001 - 内核调度器不可用时降级
        logger.error(f"定时任务调度注册失败，功能未启用：{exc}")
        shutdown_schedule_manager()
        return

    logger.info(f"定时任务功能已启用，数据库位于 {directory}")


async def stop_schedule_service() -> None:
    """停止定时任务调度并释放管理器。"""
    manager = get_schedule_manager()
    if manager is not None:
        scheduler = get_unified_scheduler()
        try:
            await scheduler.remove_schedule_by_name(SCHEDULER_TASK_NAME)
        except Exception as exc:  # noqa: BLE001 - 卸载路径不因清理失败而中断
            logger.warning(f"定时任务调度注销失败：{exc}")
    shutdown_schedule_manager()

"""管线扩展服务。

暴露给其他插件的接口，用于：

- 注册自定义管线阶段
- 读写跨流全局心智（例如让记忆插件写入「近期要闻」）

注意：``BaseService`` 每次通过 ``service_api.get_service()`` 获取
都会返回新实例（框架行为），因此本服务不持有任何跨调用状态，
所有状态都存放在模块级的注册表与全局心智单例中。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.service import BaseService

from .global_mind import get_global_mind
from .pipeline.stages import PipelineStage
from .task_runtime import (
    TaskBudget,
    TaskRequest,
    TaskResult,
    TaskRuntime,
    TaskStateStore,
    TaskStatus,
    TaskType,
    get_task_runtime_manager,
)
from .task_runtime.coordinator import start_task

logger = get_logger("agentic_chatter")

# 模块级阶段注册表，跨 Service 实例共享
_CUSTOM_STAGES: dict[str, PipelineStage] = {}


def get_custom_stage(name: str) -> PipelineStage | None:
    """获取已注册的自定义阶段。

    Args:
        name: 阶段名。

    Returns:
        PipelineStage | None: 阶段实例；未注册时返回 None。
    """
    return _CUSTOM_STAGES.get(str(name or "").strip())


def list_custom_stages() -> list[str]:
    """列出所有已注册的自定义阶段名。

    Returns:
        list[str]: 阶段名列表。
    """
    return sorted(_CUSTOM_STAGES)


async def _submit_task(
    self: Any,
    stream_id: str,
    objective: str,
    *,
    task_type: TaskType = TaskType.GENERAL,
    budget: dict[str, Any] | None = None,
    allowed_tools: tuple[str, ...] | None = None,
    result_schema: dict[str, Any] | None = None,
    validation_steps: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> TaskResult | None:
    """提交一个受预算和工具策略约束的复杂任务。

    供其他插件或组件把长耗时请求转入独立任务运行时。任务使用
    插件配置的默认预算与工具白名单，结果通过任务运行时返回。

    Args:
        stream_id: 目标聊天流 ID。
        objective: 任务目标描述。
        task_type: 任务类型，决定默认模板。
        budget: 覆盖默认预算的字段。
        allowed_tools: 任务工具白名单。
        result_schema: 最终结果 JSON schema。
        validation_steps: 需要工具证据支持的验证步骤。
        metadata: 额外任务元数据。

    Returns:
        TaskResult | None: 提交失败返回 None；否则返回初始结果占位。
    """
    from .chatter import AgenticChatter

    manager = get_task_runtime_manager()
    if manager.get_active(stream_id) is not None:
        logger.warning("提交任务失败：该聊天流已有活动任务")
        return None
    budget_kwargs = dict(budget or {})
    try:
        runtime: TaskRuntime = manager.create(
            stream_id,
            objective,
            task_type=task_type,
            allowed_tools=allowed_tools or None,
            budget=TaskBudget(**budget_kwargs) if budget_kwargs else None,
        )
    except ValueError as exc:
        logger.warning(f"提交任务失败：{exc}")
        return None
    runtime.start()
    chatter = AgenticChatter(stream_id, self.plugin)
    start_task(
        chatter,
        runtime,
        TaskRequest(
            objective=objective,
            result_schema=result_schema,
            validation_steps=validation_steps,
            metadata=metadata or {},
        ),
        _task_store(self),
    )
    return runtime.state.to_result() or TaskResult(TaskStatus.RUNNING, "任务已提交")


def _task_store(service: Any) -> TaskStateStore | None:
    """按服务所属插件配置构建任务状态存储。"""
    config = getattr(service, "plugin", None)
    directory = str(getattr(getattr(config, "tasks", None), "checkpoint_directory", "") or "").strip()
    return TaskStateStore(directory) if directory else None


def _get_task_status(self: Any, task_id: str) -> TaskResult | None:
    """按 task_id 查询任务当前结构化状态。"""
    runtime = get_task_runtime_manager().get(task_id)
    if runtime is None:
        return None
    return runtime.state.to_result() or TaskResult(
        runtime.state.status,
        f"任务状态：{runtime.state.status.value}",
    )


class PipelineService(BaseService):
    """AgenticChatter 管线扩展服务。"""

    service_name = "pipeline"
    service_description = "注册自定义回复管线阶段，并读写跨流全局心智"
    version = "0.1.0"

    submit_task = _submit_task
    get_task_status = _get_task_status

    def register_stage(self, stage: PipelineStage) -> bool:
        """注册一个自定义管线阶段。

        注册后，用户在配置的 ``stage_order`` 中写入该阶段名即可启用。

        Args:
            stage: 阶段实例，其 ``stage_name`` 不能为空。

        Returns:
            bool: 是否注册成功。
        """
        name = str(getattr(stage, "stage_name", "") or "").strip()
        if not name:
            logger.warning("注册管线阶段失败：stage_name 为空")
            return False

        _CUSTOM_STAGES[name] = stage
        logger.info(f"已注册自定义管线阶段: {name}")
        return True

    def unregister_stage(self, name: str) -> bool:
        """注销一个自定义管线阶段。

        Args:
            name: 阶段名。

        Returns:
            bool: 是否确实注销了一个阶段。
        """
        return _CUSTOM_STAGES.pop(str(name or "").strip(), None) is not None

    def add_global_note(self, note: str, limit: int = 8) -> None:
        """向全局心智写入一条跨流可见的要闻。

        适合记忆类插件调用，例如记下「答应了小明晚上帮他看代码」，
        这条信息在其他聊天流中也会被 bot 感知到。

        Args:
            note: 要闻内容，过长会被截断。
            limit: 保留的要闻条数上限。
        """
        get_global_mind().add_note(note, limit=limit)

    def get_mood_value(self) -> float:
        """读取当前的全局情绪值。

        Returns:
            float: 情绪值，取值范围 [-1.0, 1.0]，0 为基线。
        """
        return get_global_mind().get_mood().value

    def nudge_mood(self, delta: float, reason: str = "") -> float:
        """调整全局情绪。

        Args:
            delta: 情绪增量，正数偏愉悦、负数偏烦躁。
            reason: 变化原因简述。

        Returns:
            float: 调整后的情绪值。
        """
        return get_global_mind().nudge_mood(delta, reason)

    def snapshot(self) -> dict[str, Any]:
        """导出全局心智的当前快照，便于调试。

        Returns:
            dict[str, Any]: 包含情绪、流摘要与要闻的快照。
        """
        mind = get_global_mind()
        mood = mind.get_mood()
        return {
            "mood": {"value": mood.value, "label": mood.label},
            "streams": [
                {
                    "stream_id": digest.stream_id,
                    "stream_name": digest.stream_name,
                    "topic": digest.topic,
                    "age_minutes": round(digest.age_minutes(), 1),
                }
                for digest in mind.list_digests(max_streams=32)
            ],
            "notes": mind.list_notes(limit=32),
            "custom_stages": list_custom_stages(),
        }


class TaskRuntimeService(BaseService):
    """复杂任务提交服务，供其他插件转入独立任务运行时。"""

    service_name = "tasks"
    service_description = "提交受预算与工具策略约束的复杂任务并查询任务状态"
    version = "0.1.0"

    submit_task = _submit_task
    get_task_status = _get_task_status

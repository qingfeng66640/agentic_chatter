"""复杂任务的受限 LLM/tool 执行器。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.api import llm_api, send_api
from src.kernel.llm import LLMPayload, ROLE, Text

from ..humanize.segmenter import clean_reply_text_with_metadata
from ..pipeline.loop import END_TURN_CALL, STOP_CALL
from ..tooling.dedupe import build_call_key
from .models import TaskResult, TaskStatus
from .persistence import TaskStateStore
from .policy import ToolPermission
from .runtime import TaskRuntime

# 可恢复执行的任务状态；与 coordinator.RESUMABLE_TASK_STATUSES 保持同一语义
RESUMABLE_TASK_STATUSES = (
    TaskStatus.PAUSED,
    TaskStatus.WAITING_USER,
    TaskStatus.FAILED,
)


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """任务执行请求。"""

    objective: str
    task_name: str = "actor"
    result_schema: dict[str, Any] | None = None
    validation_steps: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    trigger_message: Any = None
    initial_context: tuple[LLMPayload, ...] = ()
    subtasks: tuple[Any, ...] = ()
    collaboration_concurrency: int = 2
    collaboration_limit: int | None = None
    deliver_final_text: bool = True
    report_events: bool = False


@dataclass(frozen=True, slots=True)
class TaskCommand:
    """在安全边界消费的任务控制命令。"""

    kind: str
    payload: str = ""


class TaskExecutor:
    """复用 Chatter 能力执行一个有界任务。"""

    def __init__(
        self,
        chatter: Any,
        runtime: TaskRuntime,
        checkpoint_store: TaskStateStore | None = None,
    ) -> None:
        """创建任务执行器。"""
        self.chatter = chatter
        self.runtime = runtime
        self.checkpoint_store = checkpoint_store
        self.commands: asyncio.Queue[TaskCommand] = asyncio.Queue()

    async def send_command(self, command: TaskCommand) -> None:
        """提交控制或补充输入命令。"""
        await self.commands.put(command)

    async def _consume_commands(self) -> bool:
        """在执行边界消费控制命令。"""
        should_continue = True
        while not self.commands.empty():
            command = await self.commands.get()
            if command.kind == "pause" and self.runtime.is_active():
                self.runtime.pause()
                should_continue = False
            elif command.kind == "resume" and self.runtime.state.status in (
                RESUMABLE_TASK_STATUSES
            ):
                self.runtime.resume()
            elif command.kind == "cancel":
                self.runtime.cancel()
                should_continue = False
            elif command.kind == "input" and command.payload:
                self.runtime.state.metadata["pending_input"] = command.payload
                if self.runtime.state.status == TaskStatus.WAITING_USER:
                    self.runtime.resume()
        return should_continue

    def _visible_tools(self, registry: Any) -> Any:
        """根据任务固化权限生成局部工具注册表。"""
        from src.kernel.llm import ToolRegistry

        filtered = ToolRegistry()
        tools = getattr(registry, "_tools", None)
        classes = tools.values() if isinstance(tools, dict) else ()
        for component_cls in classes:
            getter = getattr(component_cls, "get_signature", None)
            signature = str(getter() or "") if callable(getter) else ""
            names = (
                signature,
                str(getattr(component_cls, "name", "") or ""),
                str(getattr(component_cls, "tool_name", "") or ""),
                str(getattr(component_cls, "action_name", "") or ""),
                str(getattr(component_cls, "agent_name", "") or ""),
            )
            if any(
                self.runtime.policy.permission(name) == ToolPermission.ALLOW
                for name in names
                if name
            ):
                filtered.register(component_cls)
        return filtered

    async def _inject_visible_tools(self, request: Any) -> Any:
        """仅向模型注入任务策略允许的工具。"""
        usables = await self.chatter.get_llm_usables()
        usables = await self.chatter.modify_llm_usables(usables)
        registry = self._visible_tools(llm_api.create_tool_registry(tools=usables))
        schemas = registry.get_all()
        if schemas:
            request.add_payload(LLMPayload(ROLE.TOOL, schemas))
        return registry

    @staticmethod
    def _validate_result(text: str, schema: dict[str, Any] | None) -> bool:
        """按任务结果 schema 校验最终文本。"""
        if not schema:
            return True
        try:
            value = text.strip()
            if value.startswith("```"):
                value = value.strip("`").removeprefix("json").strip()
            document = json.loads(value)
        except (TypeError, ValueError):
            return False

        def matches(value: Any, rule: dict[str, Any]) -> bool:
            expected = rule.get("type")
            type_matches = {
                "object": lambda: isinstance(value, dict),
                "array": lambda: isinstance(value, list),
                "string": lambda: isinstance(value, str),
                "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
                "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
                "boolean": lambda: isinstance(value, bool),
                "null": lambda: value is None,
            }
            if expected is not None and expected not in type_matches:
                return False
            if expected is not None and not type_matches[expected]():
                return False
            if "enum" in rule and value not in rule["enum"]:
                return False
            if isinstance(value, dict):
                required = rule.get("required", ())
                if any(field not in value for field in required):
                    return False
                properties = rule.get("properties", {})
                if any(field not in properties for field in value) and rule.get(
                    "additionalProperties", True
                ) is False:
                    return False
                if any(
                    field in value and not matches(value[field], child_rule)
                    for field, child_rule in properties.items()
                    if isinstance(child_rule, dict)
                ):
                    return False
            elif isinstance(value, list) and isinstance(rule.get("items"), dict):
                if any(not matches(item, rule["items"]) for item in value):
                    return False
            return True

        return matches(document, schema)

    async def send_notice(self, text: str) -> str:
        """向聊天流发送任务通知文本（公有入口，供 coordinator 使用）。"""
        return await self._send_final_text(text)

    async def _send_final_text(self, text: str) -> str:
        """发送清理后的最终文本。"""
        cleaned = clean_reply_text_with_metadata(text).text
        final_text = cleaned.strip()
        if not final_text:
            return ""
        try:
            sent = await send_api.send_text(
                content=final_text,
                stream_id=self.chatter.stream_id,
            )
        except Exception:
            sent = False
        return final_text if sent else ""

    def _save_checkpoint(self, summary: str) -> None:
        """在工具结果边界保存检查点。"""
        if self.checkpoint_store is None:
            return
        self.runtime.checkpoint(
            self.runtime.state.current_step
            or f"iteration-{self.runtime.state.iterations + 1}",
            summary,
        )
        self.checkpoint_store.save(self.runtime)

    async def _run_validations(
        self,
        steps: tuple[str, ...],
        executed_tools: tuple[str, ...],
    ) -> tuple[str, ...]:
        """根据实际完成的工具调用生成验证证据。"""
        tool_names = tuple(name.lower() for name in executed_tools)
        evidence: list[str] = []
        for step in steps:
            normalized = step.strip()
            if not normalized:
                continue
            if normalized == "检查任务结果":
                evidence.append(f"已验证：{normalized}")
            elif normalized.startswith("运行") and any(
                name.startswith(("test", "pytest", "check")) for name in tool_names
            ):
                evidence.append(f"已验证：{normalized}")
            elif normalized.startswith("核对") and any(
                name.startswith(("search", "query", "read", "explore"))
                for name in tool_names
            ):
                evidence.append(f"已验证：{normalized}")
            elif normalized.startswith("检查") and any(
                name.startswith(("read", "search", "list", "edit"))
                for name in tool_names
            ):
                evidence.append(f"已验证：{normalized}")
            else:
                raise ValueError(f"验证步骤未获得执行证据：{normalized}")
        return tuple(evidence)

    async def _run_collaboration(self, request: TaskRequest) -> TaskResult:
        """执行请求声明的有限子任务并合并结果。"""
        from .collaboration import TaskCollaboration, TaskExecutorFactory

        collaboration = TaskCollaboration(
            max_concurrency=request.collaboration_concurrency,
            max_subtasks=self.runtime.state.budget.max_subtasks,
        )
        factory = TaskExecutorFactory(self.chatter, self.runtime)
        async def execute_subtask(spec: Any, *, dependency_context: str = "") -> TaskResult:
            return await factory.execute(
                spec,
                dependency_context=dependency_context,
                result_schema=request.result_schema,
                validation_steps=request.validation_steps,
            )

        result = await collaboration.run(
            tuple(request.subtasks),
            execute_subtask,
            parent_budget=request.collaboration_limit,
        )
        return self.runtime.adopt_result(result)

    async def run(self, request: TaskRequest) -> TaskResult:
        """执行受限的 LLM/tool loop。"""
        if self.runtime.state.status == TaskStatus.CREATED:
            self.runtime.start()
        if request.subtasks:
            return await self._run_collaboration(request)
        response: Any = self._build_initial_request(request)
        registry = await self._inject_visible_tools(response)
        executed_tools: list[str] = []

        try:
            while self.runtime.is_active():
                if not await self._consume_commands():
                    return self.runtime.state.result or TaskResult(
                        self.runtime.state.status,
                        "任务已停止",
                    )
                self._append_pending_input(response)
                budget_result = self.runtime.check_budget()
                if budget_result is not None:
                    return budget_result
                response = await response.send(stream=True)
                await response
                calls = list(getattr(response, "call_list", None) or [])
                normal_calls, denied_calls, confirmation_calls = self._classify_calls(calls)
                if confirmation_calls:
                    names = "、".join(confirmation_calls)
                    return self.runtime.set_waiting_user(f"工具 {names} 需要用户确认")
                if denied_calls:
                    names = "、".join(denied_calls)
                    return self.runtime.fail(
                        f"任务请求了未授权工具：{names}",
                        "工具权限策略拒绝执行",
                    )
                if normal_calls:
                    budget_result = await self._run_tool_round(
                        request, normal_calls, response, registry, executed_tools
                    )
                else:
                    return await self._finalize_result(request, executed_tools, response)
                if budget_result is not None:
                    return budget_result
            return self.runtime.state.result or TaskResult(
                self.runtime.state.status,
                "任务已停止",
            )
        except asyncio.CancelledError:
            if self.runtime.state.status not in (
                *RESUMABLE_TASK_STATUSES,
                TaskStatus.CANCELLED,
            ):
                self.runtime.cancel()
            raise
        except ValueError as exc:
            return self.runtime.fail("任务验证失败", str(exc))
        except Exception as exc:
            return self.runtime.fail("任务执行失败", str(exc))

    def _build_initial_request(self, request: TaskRequest) -> Any:
        """构建首轮 LLM 请求：系统提示、初始上下文与任务目标。"""
        system = (
            "你是一个受限任务执行器，只处理任务目标，不进行闲聊。"
            "每次工具调用必须推进任务，完成后用简洁文本总结。\n"
            f"任务目标：{request.objective}"
        )
        llm_request = self.chatter.create_request(
            task=request.task_name,
            request_name=f"task:{self.runtime.state.task_id}",
        )
        llm_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system)))
        for payload in request.initial_context:
            llm_request.add_payload(payload)
        llm_request.add_payload(LLMPayload(ROLE.USER, Text(request.objective)))
        return llm_request

    def _append_pending_input(self, response: Any) -> None:
        """把用户补充输入注入下一轮请求。"""
        pending_input = str(
            self.runtime.state.metadata.pop("pending_input", "") or ""
        ).strip()
        if pending_input:
            response.add_payload(LLMPayload(ROLE.USER, Text(pending_input)))

    def _classify_calls(
        self,
        calls: list[Any],
    ) -> tuple[list[Any], list[str], list[str]]:
        """按工具权限策略把调用分为放行/拒绝/待确认三组。"""
        normal_calls: list[Any] = []
        denied_calls: list[str] = []
        confirmation_calls: list[str] = []
        for call in calls:
            name = str(getattr(call, "name", "") or "")
            if name in (END_TURN_CALL, STOP_CALL):
                continue
            signature = build_call_key(name, getattr(call, "args", {}))
            permission = self.runtime.classify_tool(name, signature)
            if permission == ToolPermission.ALLOW:
                normal_calls.append(call)
            elif permission == ToolPermission.CONFIRM:
                confirmation_calls.append(name)
            else:
                denied_calls.append(name)
        return normal_calls, denied_calls, confirmation_calls

    async def _run_tool_round(
        self,
        request: TaskRequest,
        normal_calls: list[Any],
        response: Any,
        registry: Any,
        executed_tools: list[str],
    ) -> TaskResult | None:
        """执行一轮放行的工具调用并记录预算消耗。

        Returns:
            TaskResult | None: 预算耗尽时返回终态结果，否则 None 继续循环。
        """
        outcomes = await self.chatter.run_tool_call(
            normal_calls,
            response,
            registry,
            request.trigger_message,
        )
        successful_tools = [
            str(getattr(call, "name", "") or "")
            for call, outcome in zip(normal_calls, outcomes, strict=False)
            if outcome and bool(outcome[1])
        ]
        executed_tools.extend(successful_tools)
        failures = len(normal_calls) - len(successful_tools)
        for _ in range(failures):
            budget_result = self.runtime.add_failure()
            if budget_result is not None:
                return budget_result
        self._save_checkpoint(f"已执行 {len(normal_calls)} 个工具调用")
        return self.runtime.record_iteration(failures < len(normal_calls))

    async def _finalize_result(
        self,
        request: TaskRequest,
        executed_tools: list[str],
        response: Any,
    ) -> TaskResult:
        """校验并交付模型给出的最终文本。"""
        raw_message = str(getattr(response, "message", "") or "")
        if not self._validate_result(raw_message, request.result_schema):
            return self.runtime.fail(
                "任务结果校验失败",
                "模型返回结果不符合要求",
            )
        validation = await self._run_validations(
            request.validation_steps,
            tuple(executed_tools),
        )
        if request.deliver_final_text:
            final_text = await self._send_final_text(raw_message)
            if not final_text:
                return self.runtime.fail(
                    "任务结果发送失败",
                    "最终结果为空或发送失败",
                )
        else:
            final_text = raw_message.strip()
            if not final_text:
                return self.runtime.fail(
                    "任务结果为空",
                    "模型没有返回可用的结果文本",
                )
        self.runtime.record_iteration(True)
        return self.runtime.complete(final_text, validation=validation)

    @classmethod
    def restore(
        cls,
        chatter: Any,
        store: TaskStateStore,
        task_id: str,
    ) -> "TaskExecutor | None":
        """从持久化状态恢复执行器。"""
        runtime = store.load(task_id)
        return None if runtime is None else cls(chatter, runtime, store)


async def run_task(
    chatter: Any,
    runtime: TaskRuntime,
    objective: str,
    trigger_message: Any = None,
) -> TaskResult:
    """执行一个任务的便捷入口。"""
    return await TaskExecutor(chatter, runtime).run(
        TaskRequest(objective=objective, trigger_message=trigger_message)
    )

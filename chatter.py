"""AgenticChatter 核心聊天器。

实现 agent 式的回复流程：

1. 感知（可选）：用小模型概括当前对话话题。
2. 规划（可选）：产出本轮的行动意图。
3. 行动：主循环。模型的文本输出直接作为回复分段发送，
   tool call 用于真正的行为；执行完继续迭代，直到模型显式结束。
4. 反思（可选）：更新情绪与跨流全局心智。

与 DFC 的三个关键差异：

- **文本即回复**，不需要 send_text 工具，工具不再与「说话」竞争注意力。
- **说完话不自动挂起**，模型可以继续查证和记录。
- **工具分层暴露**，避免 271 个工具的 schema 洪水稀释注意力。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api import llm_api, send_api, stream_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import (
    BaseChatter,
    Failure,
    Stop,
    Success,
    Wait,
    WaitResumeEvent,
)
from src.core.components.types import ChatType
from src.core.config import get_core_config
from src.core.prompt import get_prompt_manager
from src.core.utils.context_compression import default_chat_context_compression_handler
from src.kernel.llm import (
    LLMContextManager,
    LLMRequest,
    LLMPayload,
    ROLE,
    Text,
    ToolRegistry,
)

from .config import AgenticChatterConfig
from .decision import (
    DecisionAction,
    DecisionSource,
    ReplyDecision,
    compute_semantic_relevance,
    decide_with_sub_actor,
    describe_decision,
    extract_features,
    get_participation_store,
    hard_rule_decision,
    interval_summary,
    score_features,
)
from .global_mind import get_global_mind, render_global_awareness
from .humanize.attention import should_get_distracted, should_interrupt
from .humanize.mood import describe_mood_for_prompt, infer_mood_delta
from .humanize.qqbot_streaming import create_streaming_session
from .humanize.segmenter import (
    CleanReplyResult,
    clean_reply_text_with_metadata,
    detect_provider_error_text,
    detect_reply_decision_json_text,
    is_framework_message_line,
)
from .pipeline.mailbox import TurnClaim, get_stream_mailbox, message_key
from .pipeline.loop import (
    append_control_tool_results,
    append_interrupted_tool_results,
    append_no_op_nudge,
    append_tool_result,
    build_speak_segments,
    build_tool_execution_records,
    classify_calls,
    collect_tool_result_delta,
    decide_iteration,
    deliver_segments,
    is_repeated_reply,
    should_continue_loop,
)
from .pipeline.stages import (
    STAGE_ACT,
    STAGE_DECIDE,
    STAGE_PERCEIVE,
    STAGE_PLAN,
    STAGE_REFLECT,
    resolve_stage_order,
)
from .pipeline.state import (
    ToolExecutionRecord,
    TurnOutcome,
    TurnState,
)

from .prompts import (
    DEFAULT_HOW_YOU_ACT,
    DEFAULT_HOW_YOU_SPEAK,
    DEFAULT_WHEN_TO_STOP,
)
from .tooling.dedupe import CallDeduper, build_log_args
from .tooling.explore import (
    ExploreToolsTool,
    clear_stream_catalog,
    consume_expansion,
    set_stream_catalog,
)
from .tooling.registry import (
    build_encouragement_prompt,
    build_tool_layout,
    is_blacklisted_component,
    signature_matches,
)
from .task_runtime import (
    TaskBudget,
    TaskRequest,
    TaskStateStore,
    TaskStatus,
    TaskType,
    get_task_runtime_manager,
    get_task_templates,
)
from .task_runtime.coordinator import register_task, route_message, start_task
from .tooling.provider_error_record import (
    append_provider_error_request_record,
    build_provider_error_request_record,
)




_TASK_KEYWORDS = {
    TaskType.RESEARCH: ("查询资料", "搜索资料", "查资料", "调研"),
    TaskType.CODE_ANALYSIS: ("分析代码", "检查代码", "排查代码"),
    TaskType.CODE_CHANGE: ("修改代码", "改代码", "修复代码", "更改代码"),
    TaskType.CONTENT_EDIT: ("修改内容", "编辑内容", "润色内容"),
}


def _task_request_for_runtime(runtime: Any) -> TaskRequest:
    """根据持久化任务状态重建原任务请求。"""
    template = get_task_templates().get(runtime.state.task_type)
    return TaskRequest(
        objective=runtime.state.user_goal,
        result_schema=runtime.state.metadata.get("result_schema", template.result_schema),
        validation_steps=tuple(
            runtime.state.metadata.get("validation_steps", template.validation_steps)
        ),
    )


def _detect_task_type(text: str) -> TaskType | None:
    """识别需要独立任务预算的复杂请求。"""
    stripped = text.strip()
    if stripped.startswith("任务:") or stripped.startswith("任务："):
        return TaskType.GENERAL
    for task_type, keywords in _TASK_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return task_type
    return None


def _build_task_budget(config: AgenticChatterConfig) -> TaskBudget:
    """从插件配置构造任务预算。"""
    section = config.tasks
    return TaskBudget(
        max_iterations=max(1, int(section.max_iterations)),
        max_tool_calls=max(1, int(section.max_tool_calls)),
        max_same_signature_calls=max(1, int(section.max_same_signature_calls)),
        max_no_progress_steps=max(1, int(section.max_no_progress_steps)),
        max_failures=max(1, int(section.max_failures)),
        timeout_seconds=max(1.0, float(section.timeout_seconds)),
        max_result_size=max(100, int(section.max_result_size)),
    )


def _is_task_enabled(config: AgenticChatterConfig | None) -> bool:
    """判断复杂任务运行时是否开启。"""
    return config is not None and bool(getattr(getattr(config, "tasks", None), "enabled", False))

if TYPE_CHECKING:
    from src.core.components.types import ChatterResult
    from src.core.models.message import Message
    from src.core.models.stream import ChatStream

logger = get_logger("agentic_chatter")


@dataclass(frozen=True)
class _ClaimUnreadMatch:
    """本轮 claim 与当前 stream 未读快照的无副作用匹配结果。"""

    context: Any
    matched: tuple["Message", ...]
    history_matched: tuple["Message", ...]
    unread_before: tuple["Message", ...]
    history_before: tuple["Message", ...]
    history_attr: str


_TERMINATION_LABELS = {
    "stop_requested": "请求停止",
    "end_turn_requested": "请求结束本轮",
    "text_without_tool": "纯文本无工具调用",
    "duplicate_text_without_tool": "重复文本无工具调用",
    "max_duplicate_streak": "达到重复文本上限",
    "max_post_speech": "达到发言后迭代上限",
    "max_no_progress": "达到无进展上限",
    "max_iterations": "达到最大迭代次数",
}
_TOOL_OUTCOME_LABELS = {
    "success": "成功",
    "failure": "失败",
    "skipped": "已跳过",
}
_RESULT_CAPTURE_LABELS = {
    "captured": "已捕获",
    "missing": "缺失",
    "ambiguous": "有歧义",
    "not_applicable": "不适用",
}


def _label(mapping: dict[str, str], value: object, default: str = "未知") -> str:
    """将结构化日志枚举转换为中文说明。"""
    return mapping.get(str(value), default)


# 摘要类小模型调用的最大输出长度，防止小模型话痨
MAX_DIGEST_CHARS = 40
# INFO 思考日志的单条最大长度
MAX_THOUGHT_LOG_CHARS = 300


def _thought_log_line(stream_id: str, source: str, content: str) -> str:
    """构建单行、限长的主 Agent 思考日志。"""
    normalized = " ".join(
        "".join(character if character.isprintable() else " " for character in str(content)).split()
    )
    prefix = f"[{stream_id[:8]}] 主 Agent 思考 source={source} content="
    if len(prefix) >= MAX_THOUGHT_LOG_CHARS:
        return prefix[: MAX_THOUGHT_LOG_CHARS - 1] + "…"
    available = MAX_THOUGHT_LOG_CHARS - len(prefix)
    if len(normalized) > available:
        normalized = normalized[: max(0, available - 1)] + "…"
    return prefix + normalized


class _AgenticChatterBase(BaseChatter):
    """不同聊天类型共享的 Agent 式聊天器实现。"""

    name = "agentic_base"
    description = "Agent 式回复流程：文本即回复、工具分层暴露、跨流全局心智"

    def __init__(self, stream_id: str, plugin: Any) -> None:
        """初始化聊天器。

        Args:
            stream_id: 聊天流 ID。
            plugin: 所属插件实例。
        """
        super().__init__(stream_id=stream_id, plugin=plugin)
        self._deduper = CallDeduper()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> AgenticChatterConfig | None:
        """获取插件配置。

        Returns:
            AgenticChatterConfig | None: 配置实例；未加载时返回 None。
        """
        config = getattr(self.plugin, "config", None)
        return config if isinstance(config, AgenticChatterConfig) else None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    async def execute(self) -> AsyncGenerator[ChatterResult, WaitResumeEvent | None]:
        """执行 agent 回复流程。

        Yields:
            ChatterResult: Wait/Success/Failure/Stop 结果。
        """
        config = self.cfg
        if config is not None and not config.plugin.enabled:
            yield Success("AgenticChatter 已禁用")
            return

        try:
            async for result in self._run_turns(config):
                yield result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[{self.stream_id[:8]}] 智能体循环异常：{exc}")
            yield Failure(f"agent 循环异常: {exc}", exception=exc)

    async def _run_turns(
        self,
        config: AgenticChatterConfig | None,
    ) -> AsyncGenerator[ChatterResult, WaitResumeEvent | None]:
        """驱动一轮又一轮的对话。

        Args:
            config: 插件配置。

        Yields:
            ChatterResult: 每轮结束时的结果。
        """
        while True:
            mailbox = get_stream_mailbox(self.stream_id)
            owner = object()
            generation = await mailbox.try_acquire(owner)
            if generation is None:
                yield Wait()
                continue

            claim: TurnClaim | None = None
            owner_active = True
            turn_result: ChatterResult = Wait()
            try:
                _, unread_snapshot = await self.fetch_unreads()
                added = await mailbox.merge_snapshot(unread_snapshot)
                merge_window = max(
                    0.0,
                    float(
                        getattr(
                            getattr(config, "humanize", None),
                            "input_merge_window_seconds",
                            0.0,
                        )
                    ),
                )
                pending_count, wait_seconds = await mailbox.pending_state(merge_window)
                if added:
                    logger.info(
                        f"[{self.stream_id[:8]}] 输入已合并 event=input_merged "
                        f"generation={generation} added={added} pending={pending_count} "
                        f"window_seconds={merge_window:.3f}"
                    )
                if pending_count and wait_seconds > 0:
                    logger.info(
                        f"[{self.stream_id[:8]}] 等待合并输入 event=turn_delayed_for_input_merge "
                        f"pending={pending_count} wait_seconds={wait_seconds:.3f} "
                        f"window_seconds={merge_window:.3f}"
                    )
                    released = await mailbox.release_owner(owner, generation)
                    if not released:
                        logger.error(
                            f"[{self.stream_id[:8]}] 合并窗口等待时 owner 释放失败 "
                            f"event=owner_release_failed generation={generation}"
                        )
                    owner_active = False
                    yield Wait(time=wait_seconds)
                    continue

                claim = await mailbox.claim_pending(owner, generation)
                if claim is not None:
                    from src.core.managers import get_stream_manager

                    unread_msgs = list(claim.messages)
                    unread_text = "\n".join(
                        self.format_message_line(message) for message in unread_msgs
                    )
                    chat_stream = await get_stream_manager().get_or_create_stream(
                        stream_id=self.stream_id
                    )
                    state = TurnState(
                        stream_id=self.stream_id,
                        unread_texts=unread_text,
                        deduper=self._build_deduper(config),
                    )
                    checkpoint_dir = str(
                        getattr(getattr(config, "tasks", None), "checkpoint_directory", "") or ""
                    ).strip()
                    task_store = TaskStateStore(checkpoint_dir) if checkpoint_dir else None
                    manager = get_task_runtime_manager()
                    manager.configure_store(task_store)
                    if task_store is not None:
                        for restored in task_store.load_active():
                            if manager.get(restored.state.task_id) is None:
                                manager.register(restored)
                            if restored.state.stream_id == self.stream_id:
                                request = _task_request_for_runtime(restored)
                                if restored.state.status in (
                                    TaskStatus.CREATED,
                                    TaskStatus.PLANNING,
                                    TaskStatus.RUNNING,
                                ):
                                    start_task(self, restored, request, task_store)
                                elif restored.state.status in (
                                    TaskStatus.PAUSED,
                                    TaskStatus.WAITING_USER,
                                    TaskStatus.FAILED,
                                ):
                                    register_task(self, restored, request, task_store)

                    active_runtime = manager.get_active(self.stream_id)
                    if active_runtime is not None:
                        routed = False
                        for message in unread_msgs:
                            routed = await route_message(
                                active_runtime.state.task_id,
                                str(getattr(message, "content", "") or unread_text),
                            ) or routed
                        if routed:
                            state.input_confirmed = True
                            state.spoke = False
                            outcome_result = TurnOutcome(spoke=False)
                            task_runtime = active_runtime
                        else:
                            task_runtime = None
                    else:
                        task_runtime = None
                    task_type = _detect_task_type(unread_text) if task_runtime is None else None
                    if task_type is not None and _is_task_enabled(config):
                        task_runtime = get_task_runtime_manager().create(
                            self.stream_id,
                            unread_text,
                            task_type=task_type,
                            parent_turn_id=f"{self.stream_id}:{generation}",
                            allowed_tools=tuple(getattr(config.tasks, "default_allowed_tools", ()) or ()) or None,
                            budget=_build_task_budget(config),
                            denied_tools=tuple(getattr(config.tasks, "denied_tools", ()) or ()),
                        )
                        task_runtime.start()
                        logger.info(
                            f"[{self.stream_id[:8]}] 已创建复杂任务 event=task_created "
                            f"task_id={task_runtime.state.task_id} task_type={task_type.value}"
                        )

                    if task_runtime is not None:
                        if task_type is not None:
                            checkpoint_dir = str(
                                getattr(config.tasks, "checkpoint_directory", "") or ""
                            ).strip()
                            task_store = TaskStateStore(checkpoint_dir) if checkpoint_dir else None
                            template = get_task_templates().get(task_type)
                            task_request = TaskRequest(
                                objective=unread_text,
                                trigger_message=unread_msgs[-1] if unread_msgs else None,
                                result_schema=template.result_schema,
                                validation_steps=template.validation_steps,
                            )
                        else:
                            task_request = None
                        committed = await mailbox.commit_claim(claim)
                        if not committed:
                            logger.error(
                                f"[{self.stream_id[:8]}] 任务 claim 提交失败 "
                                f"event=task_claim_commit_failed generation={generation}"
                            )
                            raise RuntimeError("任务 claim 提交失败")
                        claim = None
                        if task_request is not None:
                            start_task(self, task_runtime, task_request, task_store)
                            logger.info(
                                f"[{self.stream_id[:8]}] 任务已转入后台 event=task_background_started "
                                f"task_id={task_runtime.state.task_id}"
                            )
                        logger.info(
                            f"[{self.stream_id[:8]}] 任务消息已确认 event=task_turn_committed "
                            f"generation={generation} messages={len(unread_msgs)} "
                            f"task_id={task_runtime.state.task_id}"
                        )
                        turn_result = Wait(time=5.0)
                    else:
                        outcome_result = await self._run_pipeline(
                            config=config,
                            chat_stream=chat_stream,
                            state=state,
                            unread_msgs=unread_msgs,
                            claim=claim,
                        )

                        if state.failed:
                            interruption_streak = await mailbox.interruption_state()
                            pending_after_release = await mailbox.pending_count()
                            released = await mailbox.release_claim(claim)
                            if not released:
                                logger.error(
                                    f"[{self.stream_id[:8]}] 回合 claim 释放失败 "
                                    f"event=claim_release_failed generation={generation}"
                                )
                            claim = None
                            logger.warning(
                                f"[{self.stream_id[:8]}] 回合已释放 event=turn_released "
                                f"generation={generation} messages={len(unread_msgs)} "
                                f"visible={state.spoke} reason_code=input_interrupt "
                                f"reason={state.error} pending={pending_after_release} "
                                f"consecutive_interruptions={interruption_streak}"
                            )
                            turn_result = Wait(time=5.0)
                        else:
                            if config is not None and config.decision.enabled:
                                get_participation_store().record(
                                    self.stream_id,
                                    responded=state.spoke,
                                    topic=state.perceived_topic,
                                    ttl_seconds=max(
                                        60.0,
                                        float(config.decision.state_ttl_minutes) * 60.0,
                                    ),
                                    max_streams=max(
                                        1,
                                        int(config.decision.max_state_streams),
                                    ),
                                )

                            match = await self._match_claim_unreads(unread_msgs)
                            if match is None:
                                raise RuntimeError(
                                    "消息确认不完整："
                                    f"expected={len(unread_msgs)} actual=0"
                                )
                            committed = await mailbox.commit_claim(
                                claim,
                                apply=lambda: self._apply_claim_unread_match(match),
                            )
                            if not committed:
                                self._restore_claim_unread_match(match)
                                logger.error(
                                    f"[{self.stream_id[:8]}] mailbox claim 提交失败 "
                                    f"event=claim_commit_failed generation={generation} "
                                    f"messages={len(unread_msgs)}"
                                )
                                raise RuntimeError("mailbox claim 提交失败")
                            claim = None
                            state.input_confirmed = True
                            logger.info(
                                f"[{self.stream_id[:8]}] 回合已确认 event=turn_committed "
                                f"generation={generation} messages={len(unread_msgs)} "
                                f"matched_unread={len(match.matched)} "
                                f"already_in_history={len(match.history_matched)} "
                                f"visible={state.spoke} "
                                "consecutive_interruptions_reset=true"
                            )
                            turn_result = (
                                Stop(outcome_result.stop_seconds)
                                if outcome_result.should_stop
                                else Wait(time=outcome_result.wait_seconds)
                            )
            except asyncio.CancelledError:
                if claim is not None:
                    released = await mailbox.release_claim(claim)
                    if not released:
                        logger.error(
                            f"[{self.stream_id[:8]}] 取消时 claim 释放失败 "
                            f"event=claim_release_failed generation={generation}"
                        )
                    claim = None
                raise
            except Exception as exc:
                if claim is not None:
                    released = await mailbox.release_claim(claim)
                    if not released:
                        logger.error(
                            f"[{self.stream_id[:8]}] 异常时 claim 释放失败 "
                            f"event=claim_release_failed generation={generation}"
                        )
                    claim = None
                logger.error(f"[{self.stream_id[:8]}] 回合执行失败：{exc}")
                turn_result = Wait(time=5.0)
            finally:
                if claim is not None:
                    released = await mailbox.release_claim(claim)
                    if not released:
                        logger.error(
                            f"[{self.stream_id[:8]}] 收尾时 claim 释放失败 "
                            f"event=claim_release_failed generation={generation}"
                        )
                if owner_active:
                    owner_released = await mailbox.release_owner(owner, generation)
                    if not owner_released:
                        logger.error(
                            f"[{self.stream_id[:8]}] mailbox owner 释放失败 "
                            f"event=owner_release_failed generation={generation}"
                        )

            yield turn_result

    def _build_deduper(self, config: AgenticChatterConfig | None) -> CallDeduper:
        """按配置构建本轮的去重器。

        Args:
            config: 插件配置。

        Returns:
            CallDeduper: 去重器实例。
        """
        if config is None:
            return CallDeduper()

        mode = str(config.tools.dedupe_mode or "soft").strip().lower()
        if mode not in ("soft", "hard", "off"):
            mode = "soft"

        return CallDeduper(
            mode=mode,  # type: ignore[arg-type]
            soft_limit=max(1, int(config.tools.soft_dedupe_limit)),
        )

    def _get_persona_values(self) -> dict[str, Any]:
        """读取记录请求体时需要变量化的人设字段。"""
        try:
            personality = get_core_config().personality
        except RuntimeError:
            return {}
        return {
            "nickname": getattr(personality, "nickname", ""),
            "alias_names": getattr(personality, "alias_names", []),
            "personality_core": getattr(personality, "personality_core", ""),
            "personality_side": getattr(personality, "personality_side", ""),
            "identity": getattr(personality, "identity", ""),
            "background_story": getattr(personality, "background_story", ""),
            "reply_style": getattr(personality, "reply_style", ""),
        }

    async def _run_pipeline(
        self,
        *,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
        unread_msgs: list["Message"],
        claim: TurnClaim | None = None,
    ) -> Any:
        """按配置的阶段顺序执行管线。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
            unread_msgs: 本轮未读消息。
            claim: mailbox 固定的本轮输入；兼容直接调用时可为空。

        Returns:
            TurnOutcome: 本轮结果。
        """
        if config is None:
            order = [STAGE_ACT]
        else:
            order = resolve_stage_order(
                list(config.pipeline.stage_order),
                enable_perceive=bool(config.pipeline.enable_perceive),
                enable_decide=bool(config.decision.enabled),
                enable_plan=bool(config.pipeline.enable_plan),
                enable_reflect=bool(config.pipeline.enable_reflect),
            )

        for stage_name in order:
            if stage_name == STAGE_PERCEIVE:
                await self._stage_perceive(config, state)
            elif stage_name == STAGE_DECIDE:
                await self._stage_decide(config, chat_stream, state, unread_msgs)
            elif stage_name == STAGE_PLAN:
                if state.decision is None or state.decision.should_respond:
                    await self._stage_plan(config, state)
            elif stage_name == STAGE_ACT:
                if state.decision is None or state.decision.should_respond:
                    await self._stage_act(
                        config,
                        chat_stream,
                        state,
                        unread_msgs,
                        claim=claim,
                    )
            elif stage_name == STAGE_REFLECT:
                await self._stage_reflect(config, chat_stream, state)
            else:
                from .service import get_custom_stage

                custom_stage = get_custom_stage(stage_name)
                if custom_stage is None:
                    logger.warning(f"[{self.stream_id[:8]}] 未注册管线阶段：{stage_name}")
                    continue
                await custom_stage.run(
                    state,
                    {
                        "chatter": self,
                        "config": config,
                        "chat_stream": chat_stream,
                        "unread_messages": unread_msgs,
                    },
                )

        return state.to_outcome()

    # ------------------------------------------------------------------
    # 阶段实现
    # ------------------------------------------------------------------

    async def _stage_perceive(
        self,
        config: AgenticChatterConfig | None,
        state: TurnState,
    ) -> None:
        """情境感知阶段：概括当前话题。

        Args:
            config: 插件配置。
            state: 回合状态。
        """
        if config is None or not state.unread_texts.strip():
            return

        topic = await self._summarize(
            state.unread_texts,
            task=str(config.pipeline.perceive_model_task or "utils_small"),
            template_name="agentic_chatter_perceive",
        )
        if topic:
            state.perceived_topic = topic

    async def _stage_decide(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
        unread_msgs: list["Message"],
    ) -> None:
        """分层判断当前轮次是否应该自然介入。"""
        if config is None or not config.decision.enabled:
            state.decision = ReplyDecision(
                DecisionAction.RESPOND,
                DecisionSource.DISABLED,
                reasons=["decision_disabled"],
            )
            self._log_reply_decision(state.decision)
            return

        is_private = str(chat_stream.chat_type or "").lower() == ChatType.PRIVATE.value
        history_messages = list(chat_stream.context.history_messages)
        bot_id = str(getattr(chat_stream, "bot_id", "") or "")
        bot_nickname = str(getattr(chat_stream, "bot_nickname", "") or "")
        nicknames = {bot_nickname}
        try:
            personality = get_core_config().personality
            nicknames.add(str(getattr(personality, "nickname", "") or ""))
            nicknames.update(
                str(alias)
                for alias in getattr(personality, "alias_names", []) or []
            )
        except RuntimeError:
            pass
        bot_nicknames = tuple(
            nickname.strip() for nickname in nicknames if nickname.strip()
        )
        bot_messages = [
            message
            for message in history_messages
            if str(getattr(message, "sender_id", "") or "") == bot_id
            or str(getattr(message, "sender_role", "") or "").lower() == "bot"
        ]
        bot_message_ids = {
            str(getattr(message, "message_id", "") or "")
            for message in bot_messages
            if getattr(message, "message_id", "")
        }
        hard_decision = hard_rule_decision(
            is_private=is_private,
            bot_id=bot_id,
            bot_nickname=bot_nicknames,
            unread_messages=unread_msgs,
            bot_message_ids=bot_message_ids,
        )
        if hard_decision is not None:
            state.decision = hard_decision
            self._log_reply_decision(state.decision)
            return

        decision_config = config.decision
        store = get_participation_store()
        participation = store.get(
            self.stream_id,
            ttl_seconds=max(60.0, float(decision_config.state_ttl_minutes) * 60.0),
            max_streams=max(1, int(decision_config.max_state_streams)),
        )

        fallback_window = max(
            0.0,
            float(
                getattr(
                    decision_config,
                    "contextual_fallback_recent_reply_window_seconds",
                    300.0,
                )
            ),
        )
        reply_age = time.time() - participation.last_reply_at
        suppress_contextual_fallback = bool(
            getattr(
                decision_config,
                "enable_contextual_fallback_recent_reply_suppression",
                True,
            )
            and participation.last_reply_at > 0
            and 0.0 <= reply_age < fallback_window
        )

        if should_get_distracted(
            enabled=bool(config.humanize.enable_proactive),
            probability=float(config.humanize.distraction_probability),
            is_direct=False,
        ):
            state.decision = ReplyDecision(
                DecisionAction.SILENT,
                DecisionSource.HARD_RULE,
                reasons=["distracted"],
            )
            self._log_reply_decision(state.decision)
            return

        history_limit = max(1, int(decision_config.history_message_limit))
        recent_history = history_messages[-history_limit:]
        history_text = "\n".join(
            self.format_message_line(message) for message in recent_history
        )
        unread_text = "\n".join(
            self.format_message_line(message) for message in unread_msgs
        )

        features = extract_features(
            unread_messages=unread_msgs,
            bot_id=bot_id,
            bot_nickname=bot_nicknames,
            bot_message_ids=bot_message_ids,
            participation=participation,
            semantic_continuity=None,
            bot_history_continuity=None,
            participation_window_seconds=float(
                decision_config.participation_window_seconds
            ),
            rhythm_cooldown_seconds=float(decision_config.rhythm_cooldown_seconds),
        )

        if decision_config.local_gate_enabled:
            topic_candidates = [
                value
                for value in (
                    state.perceived_topic,
                    participation.last_reply_topic,
                )
                if value
            ]
            bot_candidates = [
                str(
                    getattr(message, "processed_plain_text", None)
                    or getattr(message, "content", "")
                    or ""
                )
                for message in reversed(bot_messages)
            ]
            semantic, bot_semantic = await compute_semantic_relevance(
                current_text=state.unread_texts,
                topic_candidates=topic_candidates,
                bot_candidates=bot_candidates,
                model_task=str(decision_config.embedding_task or "embedding"),
                candidate_limit=max(1, int(decision_config.semantic_candidate_limit)),
                max_chars_per_candidate=max(
                    64, int(decision_config.semantic_candidate_max_chars)
                ),
            )
            features.semantic_continuity = semantic or 0.0
            features.bot_history_continuity = bot_semantic or 0.0
            if semantic is not None or bot_semantic is not None:
                features.uncertainty = max(0.0, features.uncertainty - 0.18)
                features.reasons = [
                    reason for reason in features.reasons if reason != "semantic_unknown"
                ]
            local_decision = score_features(features, decision_config)
            if local_decision is not None:
                state.decision = local_decision
                self._log_reply_decision(state.decision)
                return

        score, lower, upper = interval_summary(features, decision_config)
        logger.debug(
            f"[{self.stream_id[:8]}] 本地判断未达直通条件，交由子决策模型："
            f"评分={score:.3f}，区间=[{lower:.3f}, {upper:.3f}]，"
            f"回复下界={float(decision_config.local_reply_lower_bound):.3f}，"
            f"静默上界={float(decision_config.local_silent_upper_bound):.3f}，"
            f"当前话题相关度={features.semantic_continuity:.3f}，"
            f"我的历史相关度={features.bot_history_continuity:.3f}，"
            f"直接称呼={features.direct_address:.0f}，"
            f"本地原因={features.reasons}"
        )
        template = get_prompt_manager().get_template(
            "agentic_chatter_reply_decision"
        )
        system_prompt = (
            await template.build()
            if template is not None
            else "只输出 JSON，判断本轮应 respond 还是 silent。"
        )
        state.decision = await decide_with_sub_actor(
            self,
            config=decision_config,
            system_prompt=system_prompt,
            unread_text=unread_text,
            history_text=history_text,
            signal_summary={
                "question_or_request": features.question_or_request,
                "directed_elsewhere": features.directed_elsewhere,
                "interruption_cost": features.interruption_cost,
                "topic_closure": features.topic_closure,
                "semantic_continuity": features.semantic_continuity,
                "bot_history_continuity": features.bot_history_continuity,
                "consecutive_participation": participation.consecutive_participation,
                "consecutive_silence": participation.consecutive_silence,
            },
            local_score=score,
            lower_bound=lower,
            upper_bound=upper,
            reasons=features.reasons,
            suppress_contextual_fallback=suppress_contextual_fallback,
        )
        self._log_reply_decision(state.decision)

    def _log_reply_decision(self, decision: ReplyDecision | None) -> None:
        """以中文记录本轮回复判断与诊断信息。"""
        if decision is None:
            return
        action, source, reasons = describe_decision(decision)
        logger.info(
            f"[{self.stream_id[:8]}] 回复判断：{action}"
            f"（方式：{source}；原因：{reasons}）"
        )
        logger.debug(
            f"[{self.stream_id[:8]}] 决策诊断：评分={decision.score:.3f}，"
            f"区间=[{decision.lower_bound:.3f}, {decision.upper_bound:.3f}]，"
            f"置信度={decision.confidence:.3f}，内部原因={decision.reasons}"
        )

    async def _stage_plan(
        self,
        config: AgenticChatterConfig | None,
        state: TurnState,
    ) -> None:
        """意图规划阶段：产出行动意图。

        Args:
            config: 插件配置。
            state: 回合状态。
        """
        if config is None or not state.unread_texts.strip():
            return

        note = await self._summarize(
            state.unread_texts,
            task=str(config.pipeline.plan_model_task or "utils"),
            template_name="agentic_chatter_plan",
        )
        if note and note != "不介入":
            state.plan_note = note

    async def _stage_act(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
        unread_msgs: list["Message"],
        *,
        claim: TurnClaim | None = None,
    ) -> None:
        """行动阶段：agent 主循环。

        这是与 DFC 差异最大的部分。循环不会因为「模型说完话了」
        而提前终止，模型可以在说完话之后继续调用工具。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
            unread_msgs: 本轮未读消息。
            claim: mailbox 固定的本轮输入；兼容直接调用时可为空。
        """
        usables = await self.get_llm_usables()
        usables = await self.modify_llm_usables(usables)  # type: ignore[arg-type]

        layout = self._build_layout(config, usables)
        blacklist = [] if config is None else list(config.tools.blacklist)
        layout.exposed = [usable for usable in layout.exposed if usable is not ExploreToolsTool]
        explore_enabled = config is None or bool(config.tools.enable_explore_tools)
        explore_blacklisted = is_blacklisted_component(ExploreToolsTool, blacklist) or signature_matches(
            "agentic_chatter:tool:explore_tools", blacklist
        )
        if explore_enabled and not explore_blacklisted:
            layout.exposed.insert(0, ExploreToolsTool)
            set_stream_catalog(
                self.stream_id,
                layout.collapsed_categories,
                layout.collapsed_classes,
            )
        else:
            clear_stream_catalog(self.stream_id)

        registry = ToolRegistry()
        for usable_cls in layout.exposed:
            try:
                registry.register(usable_cls)  # type: ignore[arg-type]
            except Exception as exc:
                logger.debug(f"注册组件失败，已跳过：{exc}")

        system_text = await self._build_system_prompt(config, chat_stream, layout)
        user_text = await self._build_user_prompt(config, chat_stream, state, unread_msgs)

        request = self._create_act_request(config)
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_text)))
        request.add_payload(LLMPayload(ROLE.TOOL, registry.get_all()))  # type: ignore[arg-type]
        request.add_payload(LLMPayload(ROLE.USER, Text(user_text)))

        max_iterations = 6 if config is None else max(1, int(config.pipeline.max_iterations))
        response: Any = request
        while should_continue_loop(state, max_iterations):
            state.iterations += 1

            stream_session = self._create_qqbot_streaming_session(
                config,
                unread_msgs,
            ) if self._streaming_allowed(config, state) else None
            try:
                response = await response.send(stream=True)
                if stream_session is not None:
                    await response.stream_events_with_callback(stream_session.on_event)
                else:
                    await response
            except Exception as exc:
                if stream_session is not None and stream_session.started:
                    await stream_session.finalize(
                        str(getattr(response, "message", "") or "")
                    )
                state.failed = True
                state.error = str(exc)
                logger.error(f"[{self.stream_id[:8]}] 语言模型调用失败：{exc}")
                clear_stream_catalog(self.stream_id)
                return

            calls = list(getattr(response, "call_list", None) or [])
            if await self._has_new_unreads(config, unread_msgs, claim=claim):
                if stream_session is not None and stream_session.started:
                    await stream_session.finalize(
                        str(getattr(response, "message", "") or "")
                    )
                append_interrupted_tool_results(response, calls)
                state.interrupted_by_new_input = True
                state.failed = True
                state.error = "生成期间收到新消息，重新规划"
                clear_stream_catalog(self.stream_id)
                return

            had_spoken_before_iteration = state.spoke
            normal_calls, end_seconds, stop_minutes = classify_calls(calls)
            if stream_session is not None and stream_session.started:
                cleaned_result = await stream_session.finalize(
                    str(getattr(response, "message", "") or "")
                )
                spoke_now, duplicate_text = self._record_streamed_message(
                    config,
                    response,
                    state,
                    cleaned_result,
                )
            else:
                spoke_now, duplicate_text = await self._deliver_message(
                    config, response, state
                )

            tool_progress = 0
            if normal_calls:
                tool_progress = await self._execute_calls(
                    normal_calls,
                    response,
                    state,
                    registry,
                    unread_msgs,
                    log_tool_calls=bool(
                        getattr(getattr(config, "tools", None), "log_tool_calls", False)
                    ),
                    tool_call_mode=self._tool_call_mode(config),
                )
                expanded = consume_expansion(self.stream_id)
                allowed_expanded = [
                    usable_cls
                    for usable_cls in expanded
                    if not is_blacklisted_component(usable_cls, blacklist)
                ]
                if allowed_expanded:
                    for usable_cls in allowed_expanded:
                        try:
                            registry.register(usable_cls)  # type: ignore[arg-type]
                        except Exception as exc:
                            logger.debug(f"展开工具注册失败，已跳过：{exc}")
                    response.add_payload(
                        LLMPayload(ROLE.TOOL, allowed_expanded)  # type: ignore[arg-type]
                    )

            append_control_tool_results(response, calls)

            made_progress = spoke_now or tool_progress > 0
            if made_progress:
                state.no_progress_iterations = 0
            else:
                state.no_progress_iterations += 1
            if had_spoken_before_iteration:
                state.post_speech_iterations += 1

            pipeline_config = config.pipeline if config is not None else None
            max_no_progress = max(
                1,
                int(getattr(pipeline_config, "max_no_progress_iterations", 2)),
            )
            max_post_speech = max(
                1,
                int(getattr(pipeline_config, "max_post_speech_iterations", 3)),
            )
            max_duplicate_streak = max(
                1,
                int(
                    getattr(
                        getattr(config, "humanize", None),
                        "max_duplicate_streak",
                        2,
                    )
                ),
            )
            iteration_decision = decide_iteration(
                normal_call_count=len(normal_calls),
                end_seconds=end_seconds,
                stop_minutes=stop_minutes,
                spoke_now=spoke_now,
                duplicate_text=duplicate_text,
                tool_progress=tool_progress,
                no_progress_iterations=state.no_progress_iterations,
                post_speech_iterations=state.post_speech_iterations,
                duplicate_text_streak=state.duplicate_text_streak,
                max_no_progress=max_no_progress,
                max_post_speech=max_post_speech,
                max_duplicate_streak=max_duplicate_streak,
            )
            control_call_count = len(calls) - len(normal_calls)
            logger.info(
                f"[{self.stream_id[:8]}] 行动迭代 event=act_iteration "
                f"iteration={state.iterations} normal_calls={len(normal_calls)} "
                f"control_calls={control_call_count} tool_success={tool_progress} "
                f"spoke={spoke_now} duplicate={duplicate_text} "
                f"termination={iteration_decision.reason or 'continue'} "
                f"termination_label={_label(_TERMINATION_LABELS, iteration_decision.reason or 'continue', '继续')}"
            )
            if not iteration_decision.should_continue:
                state.termination = iteration_decision
                if stop_minutes is not None:
                    state.stop_requested = True
                    state.stop_minutes = stop_minutes
                elif end_seconds is not None:
                    state.end_turn_requested = True
                    state.end_turn_seconds = end_seconds
                logger.info(
                    f"[{self.stream_id[:8]}] 行动自动结束 event=act_auto_end "
                    f"reason={iteration_decision.reason} "
                    f"reason_label={_label(_TERMINATION_LABELS, iteration_decision.reason)}"
                )
                clear_stream_catalog(self.stream_id)
                return

            if not spoke_now and not normal_calls:
                append_no_op_nudge(response)

        if state.termination is not None:
            logger.info(
                f"[{self.stream_id[:8]}] 行动自动结束 event=act_auto_end "
                f"reason={state.termination.reason} "
                f"reason_label={_label(_TERMINATION_LABELS, state.termination.reason)}"
            )
        clear_stream_catalog(self.stream_id)

    async def _stage_reflect(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
    ) -> None:
        """反思阶段：更新情绪与全局心智。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
        """
        if config is None:
            return

        mind = get_global_mind()

        if config.humanize.enable_mood and state.unread_texts.strip():
            delta, reason = infer_mood_delta(state.unread_texts)
            if delta:
                if config.global_mind.share_mood:
                    mind.nudge_mood(
                        delta,
                        reason,
                        decay_per_minute=float(config.humanize.mood_decay_per_minute),
                    )
                else:
                    mind.nudge_stream_mood(
                        self.stream_id,
                        delta,
                        reason,
                        decay_per_minute=float(config.humanize.mood_decay_per_minute),
                    )

        if not config.global_mind.enabled:
            return

        digest_mode = str(config.global_mind.digest_mode or "reuse").strip().lower()
        if digest_mode == "off":
            return

        topic = state.perceived_topic
        if not topic and digest_mode == "llm":
            topic = await self._summarize(
                state.unread_texts,
                task=str(config.global_mind.digest_model_task or "utils_small"),
                template_name="agentic_chatter_perceive",
            )

        mind.update_digest(
            self.stream_id,
            stream_name=str(getattr(chat_stream, "stream_name", "") or ""),
            chat_type=str(chat_stream.chat_type or ""),
            topic=topic,
            unread_count=0,
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _create_act_request(
        self,
        config: AgenticChatterConfig | None,
    ) -> LLMRequest:
        """根据聊天类型创建主 Agent 请求。"""
        task = "actor" if config is None else str(config.plugin.model_task or "actor")
        private_model_name = (
            ""
            if config is None or self.chat_type != ChatType.PRIVATE
            else str(config.plugin.private_model_name or "").strip()
        )
        if not private_model_name:
            return self.create_request(task=task, request_name=self.name)

        task_model_set = llm_api.get_model_set_by_task(task)
        first_task_model = task_model_set[0] if task_model_set else {}
        model_set = llm_api.get_model_set_by_name(
            private_model_name,
            temperature=first_task_model.get("temperature"),
            max_tokens=first_task_model.get("max_tokens"),
        )
        logger.info(
            f"[{self.stream_id[:8]}] 私聊主 Agent 使用指定模型 "
            f"model_name={private_model_name}"
        )
        return LLMRequest(
            model_set=model_set,
            request_name=self.name,
            meta_data={"stream_id": self.stream_id},
            context_manager=LLMContextManager(
                context_compression_handler=default_chat_context_compression_handler,
            ),
        )

    def _build_layout(
        self,
        config: AgenticChatterConfig | None,
        usables: list[type],
    ) -> Any:
        """构建本轮的工具分层布局。

        Args:
            config: 插件配置。
            usables: 可用组件类列表。

        Returns:
            ToolLayout: 分层结果。
        """
        if config is None:
            return build_tool_layout(
                usables,
                always_visible=[],
                collapsed=[],
                blacklist=[],
                max_exposed=0,
            )

        layout = build_tool_layout(
            usables,
            always_visible=list(config.tools.always_visible),
            collapsed=list(config.tools.collapsed),
            blacklist=list(config.tools.blacklist),
            max_exposed=int(config.tools.max_exposed),
        )
        logger.info(
            f"[{self.stream_id[:8]}] 工具分层: 暴露 {len(layout.exposed)}，"
            f"折叠 {len(layout.collapsed_signatures)}，丢弃 {layout.dropped_count}"
        )
        return layout

    def _streaming_allowed(
        self,
        config: AgenticChatterConfig | None,
        state: TurnState,
    ) -> bool:
        """确认本轮仍允许启动一条新的可见流式消息。"""
        max_emissions = max(
            1,
            int(
                getattr(
                    getattr(config, "pipeline", None),
                    "max_visible_text_emissions",
                    3,
                )
            ),
        )
        return state.visible_text_emissions == 0 and max_emissions > 0

    def _create_qqbot_streaming_session(
        self,
        config: AgenticChatterConfig | None,
        unread_msgs: list["Message"],
    ) -> Any | None:
        """为当前 QQBot C2C 触发消息创建实时流式会话。"""
        humanize = getattr(config, "humanize", None)
        trigger_message = unread_msgs[-1] if unread_msgs else None
        return create_streaming_session(
            trigger_message,
            enabled=bool(getattr(humanize, "streaming_enabled", False)),
            service_signature=str(
                getattr(
                    humanize,
                    "streaming_service_signature",
                    "qqbot_adapter:service:qqbot",
                )
                or ""
            ),
            initial_chars=int(getattr(humanize, "streaming_initial_chars", 1)),
            update_min_chars=int(
                getattr(humanize, "streaming_update_min_chars", 1)
            ),
        )

    def _log_main_agent_thoughts(
        self,
        response: Any,
        cleaned_result: CleanReplyResult,
    ) -> None:
        """记录结构化推理和正文中被清洗的明确思考块。"""
        reasoning = str(getattr(response, "reasoning_content", "") or "").strip()
        if reasoning:
            logger.info(_thought_log_line(self.stream_id, "reasoning_content", reasoning))
        if cleaned_result.removed_thoughts:
            logger.info(
                _thought_log_line(
                    self.stream_id,
                    "removed_message_block",
                    " | ".join(cleaned_result.removed_thoughts),
                )
            )

    def _record_streamed_message(
        self,
        config: AgenticChatterConfig | None,
        response: Any,
        state: TurnState,
        cleaned_result: CleanReplyResult,
    ) -> tuple[bool, bool]:
        """把已由 QQBot controller 输出的最终正文计入回合状态。"""
        message = cleaned_result.text
        if not message:
            return False, False

        humanize = getattr(config, "humanize", None)
        dedup_enabled = bool(getattr(humanize, "enable_reply_dedup", True))
        if dedup_enabled and is_repeated_reply(
            message,
            state.sent_texts,
            similarity_threshold=float(
                getattr(humanize, "reply_similarity_threshold", 0.88)
            ),
            containment_threshold=float(
                getattr(humanize, "reply_containment_threshold", 0.90)
            ),
        ):
            state.duplicate_text_streak += 1
            logger.warning(
                f"[{self.stream_id[:8]}] event=stream_duplicate_already_emitted "
                f"chars={len(message)} duplicate_streak={state.duplicate_text_streak}"
            )
            return False, True

        self._log_main_agent_thoughts(response, cleaned_result)
        state.spoke = True
        state.sent_texts.append(message)
        state.visible_text_emissions += 1
        state.duplicate_text_streak = 0
        return True, False

    async def _deliver_message(
        self,
        config: AgenticChatterConfig | None,
        response: Any,
        state: TurnState,
    ) -> tuple[bool, bool]:
        """将模型的非重复文本作为回复发送出去。

        这是「文本即回复」的实现点：模型不需要调用任何发送工具，
        它输出的文本会被直接分段发送。

        Args:
            config: 插件配置。
            response: 当前 LLM 响应。
            state: 回合状态。

        Returns:
            tuple[bool, bool]: 是否发送了消息、文本是否因复读被抑制。
        """
        raw_message = str(getattr(response, "message", "") or "").strip()
        cleaned_result = clean_reply_text_with_metadata(raw_message)
        message = cleaned_result.text
        if not message:
            return False, False
        provider_error_rule = detect_provider_error_text(message)
        if provider_error_rule is not None:
            logger.warning(
                f"[{self.stream_id[:8]}] event=provider_error_text_intercepted "
                f"rule={provider_error_rule} chars={len(message)}"
            )
            if bool(
                getattr(
                    getattr(config, "tools", None),
                    "record_provider_error_request_body",
                    False,
                )
            ):
                try:
                    record = build_provider_error_request_record(
                        getattr(response, "payloads", None),
                        rule=provider_error_rule,
                        stream_id=self.stream_id,
                        provider_error_chars=len(message),
                        persona_values=self._get_persona_values(),
                    )
                    record_path = await append_provider_error_request_record(record)
                    logger.info(
                        f"[{self.stream_id[:8]}] 供应商异常请求体已记录 "
                        "event=provider_error_request_recorded "
                        f"rule={provider_error_rule} payloads={record['payload_count']} "
                        f"truncated={record['truncated']} path={record_path}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{self.stream_id[:8]}] 供应商异常请求体记录失败 "
                        f"event=provider_error_request_record_failed error={type(exc).__name__}"
                    )
            return False, False
        decision_json_action = detect_reply_decision_json_text(message)
        if decision_json_action is not None:
            logger.warning(
                f"[{self.stream_id[:8]}] 已拦截主 Agent 输出的子决策 JSON "
                f"event=reply_decision_json_intercepted "
                f"action={decision_json_action} chars={len(message)}"
            )
            return False, False
        if is_framework_message_line(message):
            logger.warning(
                f"[{self.stream_id[:8]}] "
                "event=framework_message_line_intercepted "
                "reason=complete_framework_message_line "
                f"chars={len(message)}"
            )
            return False, False

        humanize = config.humanize if config is not None else None
        dedup_enabled = bool(getattr(humanize, "enable_reply_dedup", True))
        duplicate = dedup_enabled and is_repeated_reply(
            message,
            state.sent_texts,
            similarity_threshold=float(
                getattr(humanize, "reply_similarity_threshold", 0.88)
            ),
            containment_threshold=float(
                getattr(humanize, "reply_containment_threshold", 0.90)
            ),
        )
        if duplicate:
            state.duplicate_text_streak += 1
            logger.info(f"[{self.stream_id[:8]}] 检测到本轮复读，已抑制文本发送")
            return False, True

        max_emissions = max(
            1,
            int(
                getattr(
                    config.pipeline if config is not None else None,
                    "max_visible_text_emissions",
                    3,
                )
            ),
        )
        if state.visible_text_emissions >= max_emissions:
            state.duplicate_text_streak += 1
            logger.info(f"[{self.stream_id[:8]}] 已达到单轮可见发言次数上限")
            return False, True

        segments = build_speak_segments(message, humanize)
        if not segments:
            return False, False

        self._log_main_agent_thoughts(response, cleaned_result)

        async def speak(text: str) -> bool:
            """发送单条消息。

            Args:
                text: 消息文本。

            Returns:
                bool: 是否发送成功。
            """
            try:
                return await send_api.send_text(content=text, stream_id=self.stream_id)
            except Exception as exc:
                logger.error(f"[{self.stream_id[:8]}] 发送失败: {exc}")
                return False

        sent = await deliver_segments(segments, speak)
        if sent > 0:
            state.spoke = True
            state.sent_texts.append(message)
            state.visible_text_emissions += 1
            state.duplicate_text_streak = 0
            return True, False
        return False, False

    async def _execute_calls(
        self,
        calls: list[Any],
        response: Any,
        state: TurnState,
        registry: ToolRegistry,
        unread_msgs: list["Message"],
        *,
        log_tool_calls: bool = False,
        tool_call_mode: str = "planning",
    ) -> int:
        """执行本轮的普通工具调用并回灌结果。

        Args:
            calls: 待执行的 tool call 列表。
            response: 当前 LLM 响应。
            state: 回合状态。
            registry: 当前轮次可调用组件注册表。
            unread_msgs: 本轮未读消息，用于恢复动作的发送上下文。

        Returns:
            int: 执行成功的工具调用数量。
        """
        trigger = unread_msgs[-1] if unread_msgs else None
        runnable: list[Any] = []
        tool_call_mode = tool_call_mode.strip().lower()
        if tool_call_mode not in ("planning", "batch"):
            tool_call_mode = "planning"

        for call in calls:
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", None)
            args = args if isinstance(args, dict) else {}
            state.tool_calls.append(name)

            decision = state.deduper.check(name, args)
            if not decision.allow:
                append_tool_result(response, call, decision.note)
                state.tool_ledger.append(
                    ToolExecutionRecord(
                        iteration=state.iterations,
                        call_id=(
                            str(getattr(call, "id", None))
                            if getattr(call, "id", None) is not None
                            else None
                        ),
                        name=name,
                        outcome="skipped",
                        result_capture="not_applicable",
                        result_preview=decision.note,
                        counts_as_progress=False,
                    )
                )
                continue
            runnable.append(call)

        if not runnable:
            return 0

        success_count = 0
        batches = [runnable] if tool_call_mode == "batch" else [[call] for call in runnable]
        for batch in batches:
            if log_tool_calls:
                for call in batch:
                    name = str(getattr(call, "name", "") or "")
                    args = getattr(call, "args", None)
                    logger.info(
                        f"[{self.stream_id[:8]}] 调用工具 event=tool_call "
                        f"iteration={state.iterations} name={name} "
                        f"args={build_log_args(args if isinstance(args, dict) else {})}"
                    )

            payload_start = len(list(getattr(response, "payloads", None) or []))
            results = await self.run_tool_call(batch, response, registry, trigger)
            captured = collect_tool_result_delta(response, payload_start)
            records = build_tool_execution_records(
                batch,
                results,
                captured,
                iteration=state.iterations,
            )
            state.tool_ledger.extend(records)

            for call, record in zip(batch, records, strict=False):
                args = getattr(call, "args", None)
                state.deduper.record_result(
                    record.name,
                    args if isinstance(args, dict) else {},
                    record.result_preview,
                )
                if record.counts_as_progress:
                    success_count += 1
                logger.info(
                    f"[{self.stream_id[:8]}] 工具结果 event=tool_result "
                    f"iteration={record.iteration} name={record.name} "
                    f"outcome={record.outcome} "
                    f"outcome_label={_label(_TOOL_OUTCOME_LABELS, record.outcome)} "
                    f"capture={record.result_capture} "
                    f"capture_label={_label(_RESULT_CAPTURE_LABELS, record.result_capture)} "
                    f"preview_chars={len(record.result_preview)}"
                )
        return success_count

    @staticmethod
    def _message_match_key(message: "Message") -> tuple[str, ...]:
        """返回可用于重建消息唯一匹配的复合键。"""
        message_id = str(getattr(message, "message_id", "") or "").strip()
        if message_id:
            return (
                "platform",
                str(getattr(message, "stream_id", "") or ""),
                str(getattr(message, "platform", "") or "").strip().lower(),
                str(getattr(message, "sender_id", "") or ""),
                message_id,
            )
        return ("fingerprint", message_key(message))

    @classmethod
    def _match_claim_message(
        cls,
        claim_message: "Message",
        unread_messages: list["Message"],
        history_messages: list["Message"],
        unread_consumed: set[int],
        history_consumed: set[int],
    ) -> tuple[str, int | None, bool]:
        """按对象、复合身份或指纹匹配一条 claim 消息。"""
        for index, candidate in enumerate(unread_messages):
            if index not in unread_consumed and candidate is claim_message:
                return "unread", index, False
        for index, candidate in enumerate(history_messages):
            if index not in history_consumed and candidate is claim_message:
                return "history", index, False

        match_key = cls._message_match_key(claim_message)
        unread_matches = [
            index
            for index, candidate in enumerate(unread_messages)
            if index not in unread_consumed
            and cls._message_match_key(candidate) == match_key
        ]
        history_matches = [
            index
            for index, candidate in enumerate(history_messages)
            if index not in history_consumed
            and cls._message_match_key(candidate) == match_key
        ]
        if len(unread_matches) + len(history_matches) != 1:
            return "missing", None, bool(unread_matches or history_matches)
        if unread_matches:
            return "unread", unread_matches[0], False
        return "history", history_matches[0], False

    async def _match_claim_unreads(
        self,
        unread_messages: list["Message"],
    ) -> _ClaimUnreadMatch | None:
        """无副作用地把 claim 匹配到当前未读和历史快照。"""
        chat_stream = await stream_api.get_stream(stream_id=self.stream_id)
        if not chat_stream:
            logger.error(
                f"[{self.stream_id[:8]}] 消息确认匹配失败 "
                "event=claim_match_failed reason=stream_missing "
                f"expected={len(unread_messages)} actual=0"
            )
            return None

        context = chat_stream.context
        current_unreads = list(context.unread_messages)
        history_attr = (
            "history_messages"
            if hasattr(context, "history_messages")
            else "history"
        )
        current_history = list(getattr(context, history_attr))
        unread_consumed: set[int] = set()
        history_consumed: set[int] = set()
        matched: list["Message"] = []
        history_matched: list["Message"] = []
        ambiguous = False

        for claim_message in unread_messages:
            location, index, is_ambiguous = self._match_claim_message(
                claim_message,
                current_unreads,
                current_history,
                unread_consumed,
                history_consumed,
            )
            ambiguous = ambiguous or is_ambiguous
            if location == "unread" and index is not None:
                unread_consumed.add(index)
                matched.append(current_unreads[index])
            elif location == "history" and index is not None:
                history_consumed.add(index)
                history_matched.append(current_history[index])

        missing_count = len(unread_messages) - len(matched) - len(history_matched)
        if missing_count:
            identified = sum(
                bool(str(getattr(message, "message_id", "") or "").strip())
                for message in unread_messages
            )
            missing_keys = sorted(
                message_key(message)
                for message in unread_messages
                if not any(
                    message is matched_message
                    for matched_message in (*matched, *history_matched)
                )
            )
            missing_digest = hashlib.sha256(
                "\n".join(missing_keys).encode("utf-8")
            ).hexdigest()[:12]
            reason = "ambiguous_match" if ambiguous else "claim_messages_missing"
            logger.error(
                f"[{self.stream_id[:8]}] 消息确认匹配失败 "
                f"event=claim_match_failed reason={reason} "
                f"expected={len(unread_messages)} "
                f"actual={len(matched) + len(history_matched)} "
                f"matched_unread={len(matched)} already_in_history={len(history_matched)} "
                f"missing={missing_count} current_unreads={len(current_unreads)} "
                f"current_history={len(current_history)} identified={identified} "
                f"unidentified={len(unread_messages) - identified} "
                f"missing_key_digest={missing_digest}"
            )
            return None

        return _ClaimUnreadMatch(
            context=context,
            matched=tuple(matched),
            history_matched=tuple(history_matched),
            unread_before=tuple(current_unreads),
            history_before=tuple(current_history),
            history_attr=history_attr,
        )

    @classmethod
    def _resolve_claim_at_apply(
        cls,
        claim_messages: tuple["Message", ...],
        unread_messages: list["Message"],
        history_messages: list["Message"],
    ) -> tuple[list["Message"], list["Message"]]:
        """在提交时重新按唯一证据定位 claim 消息。"""
        unread_consumed: set[int] = set()
        history_consumed: set[int] = set()
        unread_matched: list["Message"] = []
        history_matched: list["Message"] = []
        for claim_message in claim_messages:
            location, index, ambiguous = cls._match_claim_message(
                claim_message,
                unread_messages,
                history_messages,
                unread_consumed,
                history_consumed,
            )
            if ambiguous:
                raise RuntimeError("消息在提交前无法唯一确认")
            if index is None:
                raise RuntimeError("未读消息在提交前发生变化")
            if location == "unread":
                unread_consumed.add(index)
                unread_matched.append(unread_messages[index])
            else:
                history_consumed.add(index)
                history_matched.append(history_messages[index])
        return unread_matched, history_matched


    @classmethod
    def _apply_claim_unread_match(cls, match: _ClaimUnreadMatch) -> None:
        """一次性应用已完成全量预检的消息确认结果。"""
        claim_messages = (*match.matched, *match.history_matched)
        unread_before_apply = list(match.context.unread_messages)
        history_before_apply = list(getattr(match.context, match.history_attr))
        try:
            unread_matched, _ = cls._resolve_claim_at_apply(
                claim_messages,
                unread_before_apply,
                history_before_apply,
            )
            consumed_ids = {id(message) for message in unread_matched}
            remained: list["Message"] = []
            consumed_counts: dict[int, int] = {}
            for message in unread_matched:
                identity = id(message)
                consumed_counts[identity] = consumed_counts.get(identity, 0) + 1
            for message in unread_before_apply:
                identity = id(message)
                count = consumed_counts.get(identity, 0)
                if identity in consumed_ids and count > 0:
                    consumed_counts[identity] = count - 1
                    match.context.add_history_message(message)
                else:
                    remained.append(message)
            match.context.unread_messages = remained
        except Exception:
            match.context.unread_messages = unread_before_apply
            setattr(match.context, match.history_attr, history_before_apply)
            raise

    @staticmethod
    def _restore_claim_unread_match(match: _ClaimUnreadMatch) -> None:
        """恢复应用未读确认前的上下文快照。"""
        match.context.unread_messages = list(match.unread_before)
        setattr(match.context, match.history_attr, list(match.history_before))

    async def _flush_claim_unreads(self, unread_messages: list["Message"]) -> int:
        """兼容直接调用：全量匹配成功后一次性确认 claim。"""
        if not unread_messages:
            return 0
        match = await self._match_claim_unreads(unread_messages)
        if match is None:
            return 0
        self._apply_claim_unread_match(match)
        return len(match.matched) + len(match.history_matched)

    async def _has_new_unreads(
        self,
        config: AgenticChatterConfig | None,
        original_unreads: list["Message"],
        *,
        claim: TurnClaim | None = None,
    ) -> bool:
        """检测生成期间是否到达了新的未读消息。

        Args:
            config: 插件配置。
            original_unreads: 本轮开始时的未读消息快照。
            claim: 当前 mailbox claim；直接调用时可为空。

        Returns:
            bool: 是否应当中止当前生成并重新规划。
        """
        if config is None or not config.humanize.enable_interrupt:
            return False
        _, current_unreads = await self.fetch_unreads()
        if claim is not None:
            mailbox = get_stream_mailbox(self.stream_id)
            added = await mailbox.merge_snapshot(current_unreads)
            pending_count, merge_wait = await mailbox.pending_state(
                max(
                    0.0,
                    float(
                        getattr(
                            config.humanize,
                            "input_merge_window_seconds",
                            0.0,
                        )
                    ),
                )
            )
            if not added:
                return False
            if merge_wait > 0:
                logger.info(
                    f"[{self.stream_id[:8]}] event=input_interrupt_suppressed "
                    f"reason=merge_window pending={pending_count} "
                    f"wait_seconds={merge_wait:.3f}"
                )
                return False
            max_interruptions = max(
                0,
                int(
                    getattr(
                        config.humanize,
                        "max_consecutive_interruptions",
                        3,
                    )
                ),
            )
            streak = await mailbox.interruption_state()
            if max_interruptions and streak >= max_interruptions:
                logger.info(
                    f"[{self.stream_id[:8]}] event=input_interrupt_suppressed "
                    f"reason=consecutive_limit pending={pending_count} "
                    f"consecutive_interruptions={streak} "
                    f"max_consecutive_interruptions={max_interruptions}"
                )
                return False
            new_streak = await mailbox.record_interruption()
            logger.info(
                f"[{self.stream_id[:8]}] event=input_interrupt_triggered "
                f"pending={pending_count} consecutive_interruptions={new_streak} "
                f"max_consecutive_interruptions={max_interruptions}"
            )
        else:
            original_ids = {id(message) for message in original_unreads}
            new_count = sum(id(message) not in original_ids for message in current_unreads)
            return should_interrupt(enabled=True, new_unread_count=new_count)
        return True

    async def _summarize(
        self,
        content: str,
        *,
        task: str,
        template_name: str,
    ) -> str:
        """用小模型做一次短摘要。

        Args:
            content: 待摘要的内容。
            task: 模型任务名。
            template_name: 提示词模板名。

        Returns:
            str: 摘要文本；失败或超长时返回空字符串。
        """
        template = get_prompt_manager().get_template(template_name)
        if template is None:
            return ""

        try:
            prompt = await template.set("conversation", content).build()
            request = self.create_request(task=task, request_name=template_name)
            request.add_payload(LLMPayload(ROLE.USER, Text(prompt)))
            response = await request.send()
        except Exception as exc:
            logger.debug(f"[{self.stream_id[:8]}] 摘要调用失败: {exc}")
            return ""

        text = " ".join(str(getattr(response, "message", "") or "").split())
        if not text or len(text) > MAX_DIGEST_CHARS * 2:
            return text[:MAX_DIGEST_CHARS] if text else ""
        return text

    def _tool_call_mode(self, config: AgenticChatterConfig | None) -> str:
        """读取并规范化工具调用模式。"""
        if config is None:
            return "planning"
        mode = str(getattr(config.tools, "tool_call_mode", "planning") or "planning")
        mode = mode.strip().lower()
        return mode if mode in ("planning", "batch") else "planning"

    def _tool_call_mode_guidance(self, mode: str) -> str:
        """构建工具调用模式提示。"""
        if mode == "batch":
            return (
                "当前工具调用模式为批量调度模式：本轮普通 Tool Call 会整批交给 "
                "MoFox Core 调度。不要依赖同一批调用中的先后顺序；如果后续工具需要前置工具的真实结果，"
                "必须拆分到下一轮，等待 Tool Result 后再调用。只有互相独立的工具才适合同轮组合，"
                "批量调度不代表绝对并行或固定执行顺序。"
            )
        return (
            "当前工具调用模式为规划模式：Agent 会按规划顺序逐个提交普通 Tool Call。"
            "如果工具之间存在结果依赖，先调用前置工具，等待 Tool Result 后再在下一轮规划后续调用；"
            "互相独立的工具仍可以组合调用。"
        )

    @staticmethod
    def _prompt_section(
        value: Any,
        default: str,
        tag: str,
    ) -> str:
        """返回用户自定义或内置默认的系统提示词区块。"""
        if isinstance(value, str) and value.strip():
            return f"<{tag}>\n{value}\n</{tag}>"
        return default

    async def _build_system_prompt(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        layout: Any,
    ) -> str:
        """构建系统提示词。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            layout: 工具分层布局。

        Returns:
            str: 系统提示词文本。
        """
        template = get_prompt_manager().get_template("agentic_chatter_system")
        if template is None:
            return "你是一个聊天助手。"

        theme_guide = ""
        extra = ""
        how_you_speak = DEFAULT_HOW_YOU_SPEAK
        how_you_act = DEFAULT_HOW_YOU_ACT
        when_to_stop = DEFAULT_WHEN_TO_STOP
        encouragement = ""
        awareness = ""
        mood_text = ""
        tool_call_mode = self._tool_call_mode(config)
        tool_call_mode_guidance = self._tool_call_mode_guidance(tool_call_mode)

        if config is not None:
            chat_type = str(chat_stream.chat_type or "").lower()
            if chat_type == ChatType.PRIVATE.value:
                theme_guide = config.persona.private_guide
            elif chat_type == ChatType.GROUP.value:
                theme_guide = config.persona.group_guide

            extra = config.persona.system_prompt_extra
            how_you_speak = self._prompt_section(
                getattr(config.persona, "how_you_speak", ""),
                DEFAULT_HOW_YOU_SPEAK,
                "how_you_speak",
            )
            how_you_act = self._prompt_section(
                getattr(config.persona, "how_you_act", ""),
                DEFAULT_HOW_YOU_ACT,
                "how_you_act",
            )
            when_to_stop = self._prompt_section(
                getattr(config.persona, "when_to_stop", ""),
                DEFAULT_WHEN_TO_STOP,
                "when_to_stop",
            )

            if config.tools.encourage_prompt:
                encouragement = build_encouragement_prompt()

            mind = get_global_mind()

            if config.global_mind.enabled:
                awareness = render_global_awareness(
                    mind,
                    current_stream_id=self.stream_id,
                    max_chars=int(config.global_mind.max_chars),
                    max_streams=int(config.global_mind.max_streams),
                    stale_minutes=float(config.global_mind.stream_stale_minutes),
                    share_mood=bool(config.global_mind.share_mood),
                    mood_decay_per_minute=float(config.humanize.mood_decay_per_minute),
                    recent_notes_limit=int(config.global_mind.recent_notes_limit),
                )

            if config.humanize.enable_mood:
                mood_state = (
                    mind.get_mood()
                    if config.global_mind.share_mood
                    else mind.get_stream_mood(self.stream_id)
                )
                mood_value = mood_state.decayed(
                    float(config.humanize.mood_decay_per_minute)
                )
                mood_text = describe_mood_for_prompt(mood_value)

        return await (
            template
            .set("nickname", str(getattr(chat_stream, "bot_nickname", "") or ""))
            .set("how_you_speak", how_you_speak)
            .set("how_you_act", how_you_act)
            .set("when_to_stop", when_to_stop)
            .set("theme_guide", theme_guide)
            .set("tool_encouragement", encouragement)
            .set("tool_call_mode_guidance", tool_call_mode_guidance)
            .set("collapsed_tools", layout.describe_categories())
            .set("global_awareness", awareness)
            .set("mood_guidance", mood_text)
            .set("system_prompt_extra", extra)
            .build()
        )

    async def _build_user_prompt(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
        unread_msgs: list["Message"],
    ) -> str:
        """构建用户提示词。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
            unread_msgs: 本轮未读消息。

        Returns:
            str: 用户提示词文本。
        """
        import datetime

        template = get_prompt_manager().get_template("agentic_chatter_user")
        if template is None:
            return state.unread_texts

        history = "\n".join(
            self.format_message_line(msg) for msg in chat_stream.context.history_messages
        )
        unreads = "\n".join(self.format_message_line(msg) for msg in unread_msgs)

        extra_parts: list[str] = []
        if state.perceived_topic:
            extra_parts.append(f"当前话题：{state.perceived_topic}")
        if state.plan_note:
            extra_parts.append(f"你刚才的打算：{state.plan_note}")

        return await (
            template
            .set("stream_name", str(getattr(chat_stream, "stream_name", "") or "未知对话"))
            .set("current_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            .set("platform", str(getattr(chat_stream, "platform", "") or "未知"))
            .set("chat_type", str(chat_stream.chat_type or "未知"))
            .set("platform_name", str(getattr(chat_stream, "bot_nickname", "") or "未知"))
            .set("platform_id", str(getattr(chat_stream, "bot_id", "") or "未知"))
            .set("history", history)
            .set("unreads", unreads)
            .set("extra", "\n".join(extra_parts))
            .build()
        )


class AgenticChatter(_AgenticChatterBase):
    """群聊 Agent 式聊天器。"""

    name = "agentic"
    description = "Agent 式回复流程：文本即回复、工具分层暴露、跨流全局心智"
    chat_type = ChatType.GROUP


class AgenticPrivateChatter(_AgenticChatterBase):
    """私聊 Agent 式聊天器。"""

    name = "agentic_private"
    description = "私聊 Agent 式回复流程：文本即回复、工具分层暴露、跨流全局心智"
    chat_type = ChatType.PRIVATE


class AgenticDiscussChatter(_AgenticChatterBase):
    """讨论组 Agent 式聊天器。"""

    name = "agentic_discuss"
    description = "讨论组 Agent 式回复流程：文本即回复、工具分层暴露、跨流全局心智"
    chat_type = ChatType.DISCUSS

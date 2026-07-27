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
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api import send_api
from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.chatter import (
    BaseChatter,
    ChatterResult,
    Failure,
    Stop,
    Success,
    Wait,
    WaitResumeEvent,
)
from src.core.components.types import ChatType
from src.core.prompt import get_prompt_manager
from src.kernel.llm import LLMPayload, ROLE, Text, ToolRegistry

from .config import AgenticChatterConfig
from .global_mind import get_global_mind, render_global_awareness
from .humanize.attention import should_get_distracted, should_interrupt
from .humanize.mood import describe_mood_for_prompt, infer_mood_delta
from .pipeline.loop import (
    append_interrupted_tool_results,
    append_no_op_nudge,
    append_tool_result,
    build_speak_segments,
    classify_calls,
    deliver_segments,
    should_continue_loop,
)
from .pipeline.stages import (
    STAGE_ACT,
    STAGE_PERCEIVE,
    STAGE_PLAN,
    STAGE_REFLECT,
    resolve_stage_order,
)
from .pipeline.state import TurnState
from .tooling.dedupe import CallDeduper
from .tooling.explore import (
    ExploreToolsTool,
    clear_stream_catalog,
    consume_expansion,
    set_stream_catalog,
)
from .tooling.registry import build_encouragement_prompt, build_tool_layout

if TYPE_CHECKING:
    from src.core.models.message import Message
    from src.core.models.stream import ChatStream

logger = get_logger("agentic_chatter")

# 摘要类小模型调用的最大输出长度，防止小模型话痨
MAX_DIGEST_CHARS = 40


class AgenticChatter(BaseChatter):
    """Agent 式聊天器。"""

    chatter_name = "agentic"
    chatter_description = (
        "Agent 式回复流程：文本即回复、工具分层暴露、跨流全局心智"
    )
    chat_type = ChatType.ALL

    def __init__(self, stream_id: str, plugin: Any) -> None:
        """初始化聊天器。

        Args:
            stream_id: 聊天流 ID。
            plugin: 所属插件实例。
        """
        super().__init__(stream_id=stream_id, plugin=plugin)
        self._deduper = CallDeduper()

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
            logger.error(f"[{self.stream_id[:8]}] agent 循环异常: {exc}")
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
        from src.core.managers import get_stream_manager

        while True:
            chat_stream = await get_stream_manager().get_or_create_stream(
                stream_id=self.stream_id
            )
            unread_text, unread_msgs = await self.fetch_unreads()

            if not unread_msgs:
                yield Wait()
                continue

            if self._should_skip(config, chat_stream, unread_msgs):
                logger.info(f"[{self.stream_id[:8]}] 本轮走神，略过 {len(unread_msgs)} 条消息")
                await self.flush_unreads(unread_msgs)
                yield Wait()
                continue

            state = TurnState(
                stream_id=self.stream_id,
                unread_texts=unread_text,
                deduper=self._build_deduper(config),
            )

            outcome_result = await self._run_pipeline(
                config=config,
                chat_stream=chat_stream,
                state=state,
                unread_msgs=unread_msgs,
            )

            if state.failed:
                logger.warning(
                    f"[{self.stream_id[:8]}] 本轮未完成，保留未读消息等待重试: {state.error}"
                )
                yield Wait(time=5.0)
                continue

            await self.flush_unreads(unread_msgs)

            if outcome_result.should_stop:
                yield Stop(outcome_result.stop_seconds)
                continue

            yield Wait(time=outcome_result.wait_seconds)

    def _should_skip(
        self,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        unread_msgs: list["Message"],
    ) -> bool:
        """判断本轮是否因走神而略过。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            unread_msgs: 本轮未读消息。

        Returns:
            bool: 为 True 表示略过本轮。
        """
        if config is None:
            return False

        is_private = str(chat_stream.chat_type or "").lower() == ChatType.PRIVATE.value
        nickname = str(getattr(chat_stream, "bot_nickname", "") or "")
        mentioned = bool(nickname) and any(
            nickname in (msg.processed_plain_text or str(msg.content or ""))
            for msg in unread_msgs
        )

        return should_get_distracted(
            enabled=bool(config.humanize.enable_proactive),
            probability=float(config.humanize.distraction_probability),
            is_direct=is_private or mentioned,
        )

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

    async def _run_pipeline(
        self,
        *,
        config: AgenticChatterConfig | None,
        chat_stream: "ChatStream",
        state: TurnState,
        unread_msgs: list["Message"],
    ) -> Any:
        """按配置的阶段顺序执行管线。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
            unread_msgs: 本轮未读消息。

        Returns:
            TurnOutcome: 本轮结果。
        """
        if config is None:
            order = [STAGE_ACT]
        else:
            order = resolve_stage_order(
                list(config.pipeline.stage_order),
                enable_perceive=bool(config.pipeline.enable_perceive),
                enable_plan=bool(config.pipeline.enable_plan),
                enable_reflect=bool(config.pipeline.enable_reflect),
            )

        for stage_name in order:
            if stage_name == STAGE_PERCEIVE:
                await self._stage_perceive(config, state)
            elif stage_name == STAGE_PLAN:
                await self._stage_plan(config, state)
            elif stage_name == STAGE_ACT:
                await self._stage_act(config, chat_stream, state, unread_msgs)
            elif stage_name == STAGE_REFLECT:
                await self._stage_reflect(config, chat_stream, state)
            else:
                from .service import get_custom_stage

                custom_stage = get_custom_stage(stage_name)
                if custom_stage is None:
                    logger.warning(f"[{self.stream_id[:8]}] 未注册管线阶段: {stage_name}")
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
    ) -> None:
        """行动阶段：agent 主循环。

        这是与 DFC 差异最大的部分。循环不会因为「模型说完话了」
        而提前终止，模型可以在说完话之后继续调用工具。

        Args:
            config: 插件配置。
            chat_stream: 当前聊天流。
            state: 回合状态。
            unread_msgs: 本轮未读消息。
        """
        usables = await self.get_llm_usables()
        usables = await self.modify_llm_usables(usables)  # type: ignore[arg-type]

        layout = self._build_layout(config, usables)
        layout.exposed = [usable for usable in layout.exposed if usable is not ExploreToolsTool]
        explore_enabled = config is None or bool(config.tools.enable_explore_tools)
        if explore_enabled:
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
                logger.debug(f"注册组件失败，已跳过: {exc}")

        system_text = await self._build_system_prompt(config, chat_stream, layout)
        user_text = await self._build_user_prompt(config, chat_stream, state, unread_msgs)

        task = "actor" if config is None else str(config.plugin.model_task or "actor")
        request = self.create_request(task=task, request_name=self.chatter_name)
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_text)))
        request.add_payload(LLMPayload(ROLE.TOOL, registry.get_all()))  # type: ignore[arg-type]
        request.add_payload(LLMPayload(ROLE.USER, Text(user_text)))

        max_iterations = 6 if config is None else max(1, int(config.pipeline.max_iterations))
        response: Any = request

        while should_continue_loop(state, max_iterations):
            state.iterations += 1

            try:
                response = await response.send()
            except Exception as exc:
                state.failed = True
                state.error = str(exc)
                logger.error(f"[{self.stream_id[:8]}] LLM 调用失败: {exc}")
                clear_stream_catalog(self.stream_id)
                return

            calls = list(getattr(response, "call_list", None) or [])
            if await self._has_new_unreads(config, unread_msgs):
                append_interrupted_tool_results(response, calls)
                state.failed = True
                state.error = "生成期间收到新消息，重新规划"
                clear_stream_catalog(self.stream_id)
                return

            spoke_now = await self._deliver_message(config, response, state)
            normal_calls, end_seconds, stop_minutes = classify_calls(calls)

            if normal_calls:
                await self._execute_calls(
                    normal_calls, response, state, registry, unread_msgs
                )
                expanded = consume_expansion(self.stream_id)
                if expanded:
                    for usable_cls in expanded:
                        try:
                            registry.register(usable_cls)  # type: ignore[arg-type]
                        except Exception as exc:
                            logger.debug(f"展开工具注册失败，已跳过: {exc}")
                    response.add_payload(
                        LLMPayload(ROLE.TOOL, expanded)  # type: ignore[arg-type]
                    )

            if stop_minutes is not None:
                state.stop_requested = True
                state.stop_minutes = stop_minutes
                clear_stream_catalog(self.stream_id)
                return

            if end_seconds is not None:
                state.end_turn_requested = True
                state.end_turn_seconds = end_seconds
                clear_stream_catalog(self.stream_id)
                return

            if not spoke_now and not normal_calls:
                append_no_op_nudge(response)

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

    async def _deliver_message(
        self,
        config: AgenticChatterConfig | None,
        response: Any,
        state: TurnState,
    ) -> bool:
        """将模型的文本输出作为回复发送出去。

        这是「文本即回复」的实现点：模型不需要调用任何发送工具，
        它输出的文本会被直接分段发送。

        Args:
            config: 插件配置。
            response: 当前 LLM 响应。
            state: 回合状态。

        Returns:
            bool: 本次是否真的发送了消息。
        """
        message = getattr(response, "message", None)
        humanize = config.humanize if config is not None else None
        segments = build_speak_segments(message, humanize)
        if not segments:
            return False

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
            return True
        return False

    async def _execute_calls(
        self,
        calls: list[Any],
        response: Any,
        state: TurnState,
        registry: ToolRegistry,
        unread_msgs: list["Message"],
    ) -> None:
        """执行本轮的普通工具调用并回灌结果。

        Args:
            calls: 待执行的 tool call 列表。
            response: 当前 LLM 响应。
            state: 回合状态。
            registry: 当前轮次可调用组件注册表。
            unread_msgs: 本轮未读消息，用于恢复动作的发送上下文。
        """
        trigger = unread_msgs[-1] if unread_msgs else None
        runnable: list[Any] = []

        for call in calls:
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", None)
            args = args if isinstance(args, dict) else {}
            state.tool_calls.append(name)

            decision = state.deduper.check(name, args)
            if not decision.allow:
                append_tool_result(response, call, decision.note)
                continue
            runnable.append(call)

        if not runnable:
            return

        results = await self.run_tool_call(runnable, response, registry, trigger)
        for call, (_, success) in zip(runnable, results, strict=False):
            args = getattr(call, "args", None)
            state.deduper.record_result(
                str(getattr(call, "name", "") or ""),
                args if isinstance(args, dict) else {},
                "执行成功" if success else "执行失败",
            )

    async def _has_new_unreads(
        self,
        config: AgenticChatterConfig | None,
        original_unreads: list["Message"],
    ) -> bool:
        """检测生成期间是否到达了新的未读消息。

        Args:
            config: 插件配置。
            original_unreads: 本轮开始时的未读消息快照。

        Returns:
            bool: 是否应当中止当前生成并重新规划。
        """
        if config is None or not config.humanize.enable_interrupt:
            return False
        _, current_unreads = await self.fetch_unreads()
        original_ids = {id(message) for message in original_unreads}
        new_count = sum(id(message) not in original_ids for message in current_unreads)
        return should_interrupt(enabled=True, new_unread_count=new_count)

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
        encouragement = ""
        awareness = ""
        mood_text = ""

        if config is not None:
            chat_type = str(chat_stream.chat_type or "").lower()
            if chat_type == ChatType.PRIVATE.value:
                theme_guide = config.persona.private_guide
            elif chat_type == ChatType.GROUP.value:
                theme_guide = config.persona.group_guide

            extra = config.persona.system_prompt_extra

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
            .set("theme_guide", theme_guide)
            .set("tool_encouragement", encouragement)
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

"""AgenticChatter 插件配置定义。

本模块定义了 agentic_chatter 的全部可配置项，覆盖四个维度：

- ``pipeline``：回复管线的阶段编排（声明式，用户可增删改顺序）
- ``tools``：工具暴露策略（分层、上限、去重模式）
- ``humanize``：拟人化行为（分段发送、情绪、主动性、打断）
- ``global_mind``：跨流全局心智（情绪共享、跨流摘要）
"""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class AgenticChatterConfig(BaseConfig):
    """AgenticChatter 配置。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "Agentic Chatter 配置"

    @config_section("plugin", title="插件设置", tag="plugin")
    class PluginSection(SectionBase):
        """插件基础配置。"""

        enabled: bool = Field(
            default=True,
            description="是否启用 AgenticChatter",
            label="启用插件",
            tag="plugin",
        )
        model_task: str = Field(
            default="actor",
            description="主回复循环使用的模型任务名，对应 config/model.toml 中的 task key",
            label="主模型任务",
            tag="ai",
        )
        private_enabled: bool = Field(
            default=True,
            description="是否由 AgenticChatter 接管私聊；修改后需重载插件或重启",
            label="接管私聊",
            tag="plugin",
        )
        private_model_name: str = Field(
            default="",
            description=(
                "私聊主回复使用的模型名称，对应 config/model.toml 中 "
                "[[models]].name；留空时使用主模型任务"
            ),
            label="私聊模型名称",
            tag="ai",
        )

    @config_section("pipeline", title="回复管线", tag="ai")
    class PipelineSection(SectionBase):
        """回复管线编排。

        管线由若干阶段组成，按 ``stage_order`` 声明候选顺序执行。
        阶段是否实际执行还会受到对应开关过滤：perceive、plan、reflect
        分别由 ``enable_perceive``、``enable_plan``、``enable_reflect`` 控制，
        decide 由 ``decision.enabled`` 控制。关闭的阶段会从实际顺序中移除；
        act 是必需阶段，即使未写入 ``stage_order`` 也会自动补上。
        """

        max_iterations: int = Field(
            default=6,
            description=(
                "单轮对话中 act 阶段的最大迭代次数。"
                "每次迭代 = 一次 LLM 调用 + 一批工具执行。达到上限后强制结束本轮。"
            ),
            label="最大迭代次数",
            tag="ai",
            hint="过小会导致 agent 来不及完成多步任务；过大可能导致单轮耗时过长",
        )
        max_no_progress_iterations: int = Field(
            default=2,
            description="连续多少次既无新文本也无新工具进展后自动结束本轮",
            label="无进展迭代上限",
            tag="ai",
        )
        max_post_speech_iterations: int = Field(
            default=3,
            description="首次发言后最多继续多少次迭代，用于完成工具任务并避免反复补话",
            label="发言后迭代上限",
            tag="ai",
        )
        max_visible_text_emissions: int = Field(
            default=3,
            description="单轮最多允许几次非重复的可见文本输出",
            label="单轮发言次数上限",
            tag="ai",
        )
        stage_order: list[str] = Field(
            default_factory=lambda: ["perceive", "decide", "act", "reflect"],
            description=(
                "回复管线的候选阶段顺序，并不表示列表中的阶段一定执行。可用阶段："
                "perceive（情境感知）、decide（是否自然介入）、plan（意图规划）、"
                "act（行动与表达）、reflect（回合反思）。"
                "perceive、plan、reflect 分别受对应 enable_* 开关控制；"
                "decide 受 decision.enabled 控制。关闭的阶段会从实际顺序中移除，"
                "act 阶段必须存在且会自动补上。"
            ),
            label="阶段顺序",
            tag="ai",
        )
        enable_perceive: bool = Field(
            default=False,
            description=(
                "是否启用情境感知阶段。开启后会在正式回复前用小模型对当前对话情境做一次摘要，"
                "提升长对话中的话题把握能力，但每轮会额外增加一次 LLM 调用。"
            ),
            label="启用情境感知",
            tag="ai",
        )
        enable_plan: bool = Field(
            default=False,
            description=(
                "是否启用意图规划阶段。开启后会在行动前先产出本轮的行动意图，"
                "适合需要多步推理的复杂场景，但会让简单闲聊显得迟钝。"
            ),
            label="启用意图规划",
            tag="ai",
        )
        enable_reflect: bool = Field(
            default=True,
            description=(
                "是否启用回合反思阶段。开启后会在本轮结束时更新情绪状态与跨流摘要。"
                "关闭后全局心智将无法获得本流的最新动态。"
            ),
            label="启用回合反思",
            tag="ai",
        )
        perceive_model_task: str = Field(
            default="utils_small",
            description="情境感知阶段使用的模型任务名，建议使用小模型以降低开销",
            label="感知阶段模型",
            tag="ai",
        )
        plan_model_task: str = Field(
            default="utils",
            description="意图规划阶段使用的模型任务名",
            label="规划阶段模型",
            tag="ai",
        )

    @config_section("decision", title="回复决策", tag="ai")
    class DecisionSection(SectionBase):
        """群聊自然介入的分层决策配置。

        决策顺序：私聊、可靠回复和直接召唤先由硬规则处理；其余群聊先提取
        本地信号与 embedding 连续性，计算线性评分及其置信区间。区间明确越过
        回复或静默边界时直接决定；两条边界之间属于灰区，才交给 sub_actor。
        sub_actor 不可用时才按 fallback_mode 兜底。
        """

        enabled: bool = Field(
            default=True,
            description=(
                "回复决策阶段的总开关。开启后，stage_order 中的 decide 会执行硬规则、"
                "本地评分和置信区间初判，灰区才调用 sub_actor；若 stage_order 未包含 decide，"
                "会在 act 前自动补入。关闭后，无论 stage_order 是否包含 decide，都会将其从实际"
                "执行顺序中移除；local_gate_enabled、评分边界和 sub_actor 均不生效，"
                "其余阶段仍按各自开关过滤后执行。"
            ),
            label="启用回复决策",
            tag="ai",
        )
        local_gate_enabled: bool = Field(
            default=True,
            description=(
                "是否启用本地规则、embedding 和置信区间初判。关闭后，除私聊和"
                "直接召唤等硬规则外，普通群聊不做本地直通，全部交给 sub_actor；"
                "这不会关闭回复功能。"
            ),
            label="启用本地初判",
            tag="ai",
        )
        model_task: str = Field(
            default="sub_actor",
            description=(
                "灰区回复决策使用的模型任务名，对应 config/model.toml 中的 task key。"
                "仅本地无法明确回复或静默时调用。"
            ),
            label="决策模型任务",
            tag="ai",
        )
        embedding_task: str = Field(
            default="embedding",
            description=(
                "计算当前消息与近期话题、Bot 历史发言连续性时使用的 embedding 任务名。"
                "该相关度只是本地信号之一，不能单独保证回复。"
            ),
            label="Embedding 任务",
            tag="ai",
        )
        max_input_tokens: int = Field(
            default=1500,
            description=(
                "决策模型总输入 token 的软上限，包含未读消息、历史和本地信号。"
                "调大可保留更多上下文，但会增加模型成本和延迟；不改变评分公式。"
            ),
            label="决策输入上限",
            tag="performance",
        )
        max_unread_tokens: int = Field(
            default=700,
            description=(
                "决策输入中未读消息部分的 token 软上限。调大更利于理解长消息批次，"
                "调小可限制成本；其余上下文预算仍由总输入上限约束。"
            ),
            label="未读输入上限",
            tag="performance",
        )
        history_message_limit: int = Field(
            default=10,
            description=(
                "决策最多读取的近期历史消息条数，用于判断话题和 Bot 发言连续性。"
                "调大可参考更长上下文，但会增加输入量和延迟。"
            ),
            label="历史消息上限",
            tag="performance",
        )
        semantic_candidate_limit: int = Field(
            default=4,
            description=(
                "每类 embedding 语义候选的最大数量，分别从近期话题和 Bot 历史中选取。"
                "调大可扩大语义参考范围，但会增加 embedding 输入量。"
            ),
            label="语义候选上限",
            tag="performance",
        )
        semantic_candidate_max_chars: int = Field(
            default=360,
            description=(
                "单个 embedding 语义候选的字符上限。调大可保留更完整的长消息，"
                "调小可减少 embedding 成本；不会改变候选数量。"
            ),
            label="语义候选字数",
            tag="performance",
        )
        local_reply_lower_bound: float = Field(
            default=0.30,
            description=(
                "本地直接回复边界。实际比较的是 lower_bound = score - uncertainty，"
                "而非原始 score；当下界大于等于此值时直接回复，不调用 sub_actor。"
                "调高更谨慎、灰区更多；调低更积极、误插话风险更高。"
            ),
            label="回复直通边界",
            tag="ai",
        )
        local_silent_upper_bound: float = Field(
            default=-0.25,
            description=(
                "本地直接静默边界。实际比较的是 upper_bound = score + uncertainty，"
                "当上界小于等于此值时直接暂不回复。两条直通边界之间属于灰区，"
                "会交给 sub_actor；调高更容易本地静默，调低则更常进入灰区。"
            ),
            label="静默直通边界",
            tag="ai",
        )
        base_uncertainty: float = Field(
            default=0.12,
            description=(
                "置信区间的基础半径：lower_bound = score - uncertainty，"
                "upper_bound = score + uncertainty。数值越大区间越宽，本地越难确定，"
                "更多消息进入 sub_actor；语义不可用或信号冲突时还会额外增加不确定度。"
            ),
            label="基础不确定度",
            tag="ai",
        )
        gray_zone_randomness: float = Field(
            default=0.0,
            description=(
                "给本地评分加入的有限随机扰动，仅用于灰区附近的边界实验。默认 0 表示"
                "相同输入可复现；调大后边界消息可能得到不同结果，不建议稳定生产环境开启。"
            ),
            label="灰区随机扰动",
            tag="ai",
        )
        fallback_mode: str = Field(
            default="contextual",
            description=(
                "sub_actor 调用失败时的兜底策略，不参与正常主路径：contextual 按当前"
                "上下文倾向回复，fail_open 直接回复，fail_closed 直接静默。"
            ),
            label="失败回退",
            tag="ai",
        )
        enable_contextual_fallback_recent_reply_suppression: bool = Field(
            default=True,
            description=(
                "近期已成功回复时，是否抑制 sub_actor 失败后的 contextual 自动回复，"
                "避免故障回退时重复插话。不影响私聊、可靠回复和结构化 @ 等硬规则。"
            ),
            label="启用 fallback 近期回复抑制",
            tag="ai",
        )
        contextual_fallback_recent_reply_window_seconds: float = Field(
            default=300.0,
            description=(
                "contextual fallback 的近期成功回复抑制窗口（秒）。窗口内 sub_actor"
                "失败会优先静默以避免重复插话；设为 0 可关闭该窗口。"
            ),
            label="fallback 回复抑制窗口",
            tag="ai",
        )
        participation_window_seconds: float = Field(
            default=600.0,
            description=(
                "近期参与惯性的衰减窗口（秒）。窗口越长，Bot 过去参与当前流的记录"
                "保留越久；调大更易延续已参与话题，调小则更快回到中性状态。"
            ),
            label="参与惯性窗口",
            tag="ai",
        )
        rhythm_cooldown_seconds: float = Field(
            default=35.0,
            description=(
                "Bot 成功回复后的群聊节奏冷却时间（秒）。冷却内会提高重复插话的代价；"
                "调大更克制，调小更容易连续参与。"
            ),
            label="节奏冷却",
            tag="ai",
        )
        state_ttl_minutes: float = Field(
            default=180.0,
            description=(
                "单个聊天流参与状态在无活动后保留的最长时间（分钟）。过期后会忘记"
                "该流的近期回复与静默惯性；调大更连续，但会占用更多内存。"
            ),
            label="状态过期时间",
            tag="performance",
        )
        max_state_streams: int = Field(
            default=256,
            description=(
                "内存中最多保留参与状态的聊天流数量。超过上限时会淘汰较旧状态；"
                "调大可覆盖更多活跃群，但会增加内存占用。"
            ),
            label="状态流上限",
            tag="performance",
        )
        weight_direct_address: float = Field(
            default=0.30,
            description=(
                "弱受话人信号的正向加分权重，例如消息看起来在向 Bot 提问但未满足硬规则。"
                "权重越大，受话人信号越容易推动回复；不应单独作为插话依据。"
            ),
            label="受话人权重",
            tag="ai",
        )
        weight_semantic_continuity: float = Field(
            default=0.26,
            description=(
                "当前消息与近期话题 embedding 连续性的正向加分权重。权重越大，"
                "承接当前话题越容易回复；语义相关本身不能单独保证回复。"
            ),
            label="话题相关权重",
            tag="ai",
        )
        weight_bot_history_continuity: float = Field(
            default=0.18,
            description=(
                "当前消息与 Bot 近期发言 embedding 连续性的正向加分权重。调大可"
                "增强对 Bot 话题后续追问的响应；仍会受他人定向、冷却和打断成本保护。"
            ),
            label="历史相关权重",
            tag="ai",
        )
        weight_participation_momentum: float = Field(
            default=0.08,
            description=(
                "Bot 近期已参与当前流的正向加分权重。调大更倾向延续已参与对话，"
                "调小则降低刷存在感风险。"
            ),
            label="参与惯性权重",
            tag="ai",
        )
        weight_question_or_request: float = Field(
            default=0.08,
            description=(
                "消息呈现问题或请求形态时的正向加分权重。调大更愿意响应提问；"
                "泛问题、面向其他成员的问题仍可能进入灰区或静默。"
            ),
            label="问题请求权重",
            tag="ai",
        )
        weight_contribution_value: float = Field(
            default=0.10,
            description=(
                "消息信息量及 Bot 可提供有效贡献时的正向加分权重。调大更偏好有内容的"
                "后续讨论；它需要与连续性等信号结合，不能单独触发回复。"
            ),
            label="贡献价值权重",
            tag="ai",
        )
        weight_directed_elsewhere: float = Field(
            default=0.34,
            description=(
                "消息明确面向其他成员时的惩罚强度。配置填写正数，代码会自动取负加入评分；"
                "调大更不易打断他人对话。"
            ),
            label="他人定向权重",
            tag="ai",
        )
        weight_interruption_cost: float = Field(
            default=0.24,
            description=(
                "多人快速交谈时的打断代价惩罚强度。配置填写正数，代码会自动取负；"
                "调大后 Bot 会更少插入节奏紧密、没有明确入口的对话。"
            ),
            label="打断代价权重",
            tag="ai",
        )
        weight_topic_closure: float = Field(
            default=0.24,
            description=(
                "话题已收尾时的惩罚强度。配置填写正数，代码会自动取负；调大后"
                "更倾向让已结束的话题自然结束，减少无意义补话。"
            ),
            label="话题闭合权重",
            tag="ai",
        )
        weight_rhythm_cooldown: float = Field(
            default=0.20,
            description=(
                "刚回复后的节奏冷却惩罚强度。配置填写正数，代码会自动取负；调大后"
                "冷却期内更克制，调小则更容易连续回复。"
            ),
            label="节奏冷却权重",
            tag="ai",
        )
        weight_silence_momentum: float = Field(
            default=0.04,
            description=(
                "连续静默后的轻微惩罚强度。配置填写正数，代码会自动取负；调大后"
                "更保持沉默惯性，调小则更容易重新参与。"
            ),
            label="静默惯性权重",
            tag="ai",
        )
        weight_low_information: float = Field(
            default=0.10,
            description=(
                "低信息短消息的惩罚强度。配置填写正数，代码会自动取负；调大后"
                "对“嗯”“好的”等缺少新信息的消息更倾向静默。"
            ),
            label="低信息权重",
            tag="ai",
        )

    @config_section("tools", title="工具策略", tag="ai")
    class ToolsSection(SectionBase):
        """工具暴露与调用策略。

        这一节直接决定了 bot 的工具调用意愿。默认值针对
        "让 bot 更愿意调用工具" 做了专门调优。
        """

        max_exposed: int = Field(
            default=25,
            description=(
                "每轮注入给 LLM 的工具数量上限。"
                "工具过多会稀释模型注意力，导致反而不调用工具；"
                "超出上限的工具会被折叠，需要模型主动调用 explore_tools 才能展开。"
            ),
            label="工具暴露上限",
            tag="ai",
            hint="建议 15-40。设为 0 表示不限制（不推荐，可能导致工具调用率骤降）",
        )
        always_visible: list[str] = Field(
            default_factory=lambda: [
                "emoji_sender:action:*",
                "booku_memory:tool:*",
                "profile_memory:tool:*",
                "todo_plugin:tool:*",
            ],
            description=(
                "始终可见的工具签名列表，支持 * 通配符。"
                "这些工具会被置顶且永不折叠，适合放置对拟人化最关键的能力"
                "（表情包、记忆、日程等）。"
            ),
            label="常驻工具",
            tag="ai",
        )
        collapsed: list[str] = Field(
            default_factory=lambda: ["onebot_expand:tool:*"],
            description=(
                "默认折叠的工具签名列表，支持 * 通配符。"
                "这些工具不会直接出现在工具列表中，而是收进分类目录，"
                "由模型调用 explore_tools 按需展开。适合数量庞大的平台 API 工具集。"
            ),
            label="折叠工具",
            tag="ai",
        )
        blacklist: list[str] = Field(
            default_factory=list,
            description="完全禁用的工具签名列表，支持 * 通配符。这些工具对模型完全不可见。",
            label="禁用工具",
            tag="ai",
        )
        enable_explore_tools: bool = Field(
            default=True,
            description=(
                "是否启用 explore_tools 渐进披露工具。"
                "关闭后被折叠的工具将完全无法使用。"
            ),
            label="启用工具探索",
            tag="ai",
        )
        dedupe_mode: str = Field(
            default="soft",
            description=(
                "工具重复调用的处理模式。"
                "soft：允许重复但附带上次结果提醒（推荐，不打击调用意愿）；"
                "hard：直接拒绝重复调用；"
                "off：完全不做去重处理。"
            ),
            label="去重模式",
            tag="ai",
            hint="hard 模式会显著降低模型的工具调用意愿，除非确有必要否则不建议",
        )
        soft_dedupe_limit: int = Field(
            default=3,
            description="soft 去重模式下，同一工具同参数在单轮内允许的最大重复次数",
            label="软去重上限",
            tag="ai",
        )
        log_tool_calls: bool = Field(
            default=False,
            description="是否记录实际工具调用的名称和安全处理后的参数，不记录工具返回正文",
            label="记录工具调用日志",
            tag="debug",
        )
        record_provider_error_request_body: bool = Field(
            default=False,
            description=(
                "供应商异常正文被拦截时，是否将受控 LLM 请求体记录到插件 JSONL。"
                "人设、身份、背景和回复风格会替换为变量；用户输入和工具调用仍会保留。"
            ),
            label="记录供应商异常请求体",
            tag="debug",
            hint="默认关闭；开启前请确认本地数据目录访问权限",
        )
        tool_call_mode: str = Field(
            default="planning",
            description=(
                "工具调用模式。planning：按依赖顺序逐个提交工具调用；"
                "batch：将本轮普通工具调用整批交给 MoFox Core 调度，"
                "不代表绝对并行或固定执行顺序。"
            ),
            label="工具调用模式",
            tag="ai",
            hint="可选 planning 或 batch，默认 planning",
        )
        encourage_prompt: bool = Field(
            default=True,
            description=(
                "是否在系统提示词中加入鼓励工具使用的引导语。"
                "开启后会明确告诉模型：查证、记录、表达情绪都应该主动使用工具，"
                "而不是仅凭上下文臆测。"
            ),
            label="鼓励工具使用",
            tag="ai",
        )

    @config_section("humanize", title="拟人化", tag="ai")
    class HumanizeSection(SectionBase):
        """拟人化行为配置。"""

        enable_segmentation: bool = Field(
            default=True,
            description=(
                "是否启用消息分段发送。开启后长回复会按语义边界切成多条依次发送，"
                "更接近真人的聊天习惯。"
            ),
            label="启用分段发送",
            tag="ai",
        )
        max_segment_chars: int = Field(
            default=60,
            description="单条消息的目标最大字符数，超出后会尝试在标点处切分",
            label="单段最大字数",
            tag="ai",
        )
        max_segments: int = Field(
            default=4,
            description="单轮回复最多切分成几条消息，超出部分会合并到最后一条",
            label="最大分段数",
            tag="ai",
        )
        typing_cps: float = Field(
            default=8.0,
            description=(
                "模拟打字速度（字符/秒），用于计算消息之间的发送延迟。"
                "设为 0 表示不模拟延迟，立即连续发送。"
            ),
            label="打字速度",
            tag="ai",
            hint="真人中文打字速度约 5-12 字/秒",
        )
        max_typing_delay: float = Field(
            default=4.0,
            description="单条消息的最大模拟打字延迟（秒），防止超长消息导致过久等待",
            label="最大打字延迟",
            tag="performance",
        )
        streaming_enabled: bool = Field(
            default=False,
            description="是否对 QQBot C2C 私聊启用实时 token 流式输出；默认关闭",
            label="启用 QQBot 实时流式输出",
            tag="ai",
        )
        streaming_service_signature: str = Field(
            default="qqbot_adapter:service:qqbot",
            description="QQBot 流式 Service 签名；留空时仅在唯一能力 Service 存在时自动发现",
            label="流式 Service 签名",
            tag="ai",
        )
        streaming_initial_chars: int = Field(
            default=1,
            description="累计到指定可见字符数后才启动 QQBot 流式消息，避免发送空内容",
            label="流式首发字符数",
            ge=1,
            le=100,
            tag="ai",
        )
        streaming_update_min_chars: int = Field(
            default=1,
            description="可见正文至少新增指定字符后才请求一次流式更新；适配器仍负责时间节流",
            label="流式最小更新字符数",
            ge=1,
            le=100,
            tag="performance",
        )
        enable_mood: bool = Field(
            default=True,
            description=(
                "是否启用情绪状态。开启后 bot 会维护一个不外显的情绪值，"
                "情绪会影响用词倾向，并随时间自然衰减回基线。"
            ),
            label="启用情绪状态",
            tag="ai",
        )
        mood_decay_per_minute: float = Field(
            default=0.05,
            description="情绪每分钟向基线衰减的幅度，值越大情绪恢复越快",
            label="情绪衰减速率",
            tag="ai",
        )
        enable_proactive: bool = Field(
            default=False,
            description=(
                "是否启用主动性与走神。开启后 bot 可能在沉默一段时间后主动起话题，"
                "也可能偶尔漏看消息不予回复。这是最拟人的特性，"
                "但也最容易被误认为是故障，因此默认关闭。"
            ),
            label="启用主动性与走神",
            tag="ai",
            hint="开启前请确认你能接受 bot 偶尔不回消息",
        )
        distraction_probability: float = Field(
            default=0.05,
            description="启用主动性后，单轮消息被「走神」略过的概率，有效范围 0.0-1.0",
            label="走神概率",
            tag="ai",
        )
        enable_reply_dedup: bool = Field(
            default=True,
            description="是否抑制同一轮中的相同或近似重复文本，同时继续执行有效工具调用",
            label="启用文本复读抑制",
            tag="ai",
        )
        reply_similarity_threshold: float = Field(
            default=0.88,
            description="字符 n-gram Jaccard 达到该值时视为近似复读",
            label="复读相似度阈值",
            tag="ai",
        )
        reply_containment_threshold: float = Field(
            default=0.90,
            description="较短文本被已发文本覆盖的比例达到该值时视为复读",
            label="复读覆盖率阈值",
            tag="ai",
        )
        max_duplicate_streak: int = Field(
            default=2,
            description="连续复读达到该次数且没有新工具调用时自动结束本轮",
            label="连续复读上限",
            tag="ai",
        )
        enable_interrupt: bool = Field(
            default=True,
            description=(
                "是否启用打断重规划。开启后，若在生成回复期间收到新消息，"
                "会中止当前回复并基于新消息重新规划，避免答非所问。"
            ),
            label="启用打断重规划",
            tag="ai",
        )
        input_merge_window_seconds: float = Field(
            default=0.0,
            description="新回合领取输入前的固定合并窗口；0 表示立即处理，不额外增加延迟",
            label="输入合并窗口（秒）",
            ge=0.0,
            tag="performance",
        )
        max_consecutive_interruptions: int = Field(
            default=3,
            description="同一聊天流连续因新输入中断的上限；0 表示不限制",
            label="连续中断上限",
            ge=0,
            tag="performance",
        )

    @config_section("global_mind", title="全局心智", tag="ai")
    class GlobalMindSection(SectionBase):
        """跨流全局心智配置。

        解决的问题：默认情况下每个聊天流的上下文彼此隔离，
        bot 在不同群里表现得像不同的人。全局心智让多个流共享
        情绪状态与高度压缩的跨流摘要，使人格保持连贯。

        注意力约束：跨流信息必须高度压缩后注入，否则会稀释模型
        对当前对话的注意力。因此这里的所有内容都有严格的长度上限。
        """

        enabled: bool = Field(
            default=True,
            description=(
                "是否启用跨流全局心智。关闭后各聊天流完全独立，"
                "退化为传统的单流行为。"
            ),
            label="启用全局心智",
            tag="ai",
        )
        max_chars: int = Field(
            default=600,
            description=(
                "注入到提示词中的全局感知块的总字符数硬上限。"
                "这是防止跨流信息稀释模型注意力的关键参数，超出部分会按活跃度截断。"
            ),
            label="全局块字数上限",
            tag="ai",
            hint="不建议超过 1000，否则会明显影响模型对当前对话的专注度",
        )
        max_streams: int = Field(
            default=6,
            description="全局感知中最多展示几个其他聊天流的摘要，按最近活跃度排序",
            label="展示流数量上限",
            tag="ai",
        )
        share_mood: bool = Field(
            default=True,
            description=(
                "情绪是否跨流共享。开启后在某个群里产生的情绪会带到其他群，"
                "更接近真人；关闭则每个流维护独立情绪。"
            ),
            label="情绪跨流共享",
            tag="ai",
        )
        digest_mode: str = Field(
            default="reuse",
            description=(
                "跨流摘要的产出方式。"
                "reuse：复用主回复循环的产出，不额外调用 LLM（推荐，零开销）；"
                "llm：用小模型专门总结，质量更高但每轮增加一次调用；"
                "off：不产出摘要，仅共享情绪。"
            ),
            label="摘要产出方式",
            tag="ai",
        )
        digest_model_task: str = Field(
            default="utils_small",
            description="digest_mode 为 llm 时使用的模型任务名",
            label="摘要模型任务",
            tag="ai",
        )
        stream_stale_minutes: float = Field(
            default=120.0,
            description=(
                "聊天流摘要的过期时间（分钟）。"
                "超过该时长未活跃的流不再出现在全局感知中。"
            ),
            label="流摘要过期时间",
            tag="ai",
        )
        recent_notes_limit: int = Field(
            default=8,
            description=(
                "跨流可见的「近期要闻」条数上限。"
                "要闻是指 bot 在各个流中学到或承诺过的事，例如"
                "「答应了小明晚上帮他看代码」。"
            ),
            label="近期要闻上限",
            tag="ai",
        )

    @config_section("persona", title="人设补充", tag="text")
    class PersonaSection(SectionBase):
        """人设与场景引导。"""

        private_guide: str = Field(
            default=(
                "你当前处于私聊环境。私聊是一对一的、私密的，"
                "请结合你与对方的关系亲疏来把握分寸：对陌生人不必过分亲昵，"
                "对熟识的人也不该冷淡敷衍。保持独立判断，不要被对方的话术带着走。"
            ),
            description="私聊场景的额外提示词",
            label="私聊场景引导",
            input_type="textarea",
            rows=3,
            tag="text",
        )
        group_guide: str = Field(
            default=(
                "你当前处于群聊环境。群里同时有很多人，你只是其中一员，"
                "不必每条消息都接话。插话前先判断你的介入是否自然；"
                "决定参与时就认真参与，不要敷衍，也不要过度刷存在感。"
            ),
            description="群聊场景的额外提示词",
            label="群聊场景引导",
            input_type="textarea",
            rows=3,
            tag="text",
        )
        system_prompt_extra: str = Field(
            default="",
            description="追加到系统提示词末尾的自定义内容，留空则不追加",
            label="系统提示词追加",
            input_type="textarea",
            rows=3,
            tag="text",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    pipeline: PipelineSection = Field(default_factory=PipelineSection)
    decision: DecisionSection = Field(default_factory=DecisionSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    humanize: HumanizeSection = Field(default_factory=HumanizeSection)
    global_mind: GlobalMindSection = Field(default_factory=GlobalMindSection)
    persona: PersonaSection = Field(default_factory=PersonaSection)

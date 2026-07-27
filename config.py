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

    @config_section("pipeline", title="回复管线", tag="ai")
    class PipelineSection(SectionBase):
        """回复管线编排。

        管线由若干阶段组成，按 ``stage_order`` 声明的顺序依次执行。
        每个阶段都可以单独开关，未启用的阶段会被直接跳过。
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
        stage_order: list[str] = Field(
            default_factory=lambda: ["perceive", "act", "reflect"],
            description=(
                "回复管线的阶段执行顺序。可用阶段："
                "perceive（情境感知）、plan（意图规划）、act（行动与表达）、reflect（回合反思）。"
                "act 阶段必须存在，其余阶段可自由增删和调整顺序。"
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
        enable_interrupt: bool = Field(
            default=True,
            description=(
                "是否启用打断重规划。开启后，若在生成回复期间收到新消息，"
                "会中止当前回复并基于新消息重新规划，避免答非所问。"
            ),
            label="启用打断重规划",
            tag="ai",
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
    tools: ToolsSection = Field(default_factory=ToolsSection)
    humanize: HumanizeSection = Field(default_factory=HumanizeSection)
    global_mind: GlobalMindSection = Field(default_factory=GlobalMindSection)
    persona: PersonaSection = Field(default_factory=PersonaSection)

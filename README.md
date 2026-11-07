# Agentic Chatter

`Agentic Chatter` 是 Neo-MoFox 的 Agent 式聊天器插件（v0.8.1）。它将一轮回复拆分为可编排的处理阶段，并在群聊中先判断“是否应当参与”，再决定如何生成、执行和收尾回复，避免 Bot 对每条消息机械接话。插件支持按任务选择工具调用模式：`planning` 按顺序逐项提交，`batch` 将独立调用整批交给 Core 调度。插件还会拦截被上游错误包装成正常正文的明显供应商审核错误，避免将异常说明直接发送到聊天中；同时通过 mailbox 在回合结束时按对象、平台复合身份和完整指纹重新对账，歧义时拒绝确认，避免重复或错误确认未读消息。插件内置复杂任务运行时：主 Agent 可把复杂请求派发给带独立预算与工具权限的 sub-agent 后台执行（同流可并发多个），任务完成后自动回灌主 Agent 向用户汇报，并可通过 `manage_tasks` 工具全程查询与取消。

- **维护者**：qf
- **仓库**：`qingfeng66640/agentic_chatter`
- **最低核心版本**：`1.3.0-alpha.1`
- **许可证**：GPL-3.0，详见 [LICENSE](LICENSE)
- **升级文档**：从 0.7.x 升级请先读 [docs/upgrade-0.8.0.md](docs/upgrade-0.8.0.md)

## 安装与启用

将插件目录放入 Neo-MoFox 的插件目录，确保 `manifest.json` 与 `plugin.py` 位于插件根目录。启动 Neo-MoFox 后，插件管理器会按 manifest 加载 `agentic_chatter`；首次加载后会生成或更新插件配置文件。

插件不包含平台适配器，必须先配置并启用对应平台的 Adapter。插件也依赖 Neo-MoFox 核心提供的模型任务、上下文、工具和消息发送能力，不需要额外的 Python 第三方依赖。

## 功能概览

- **分阶段对话流程**：按 `perceive → decide → plan → act → reflect` 处理消息；阶段顺序可通过配置调整。
- **自然参与决策**：综合直接称呼、问题请求、话题与 Bot 历史连续性、多人对话、他人定向、话题收尾、消息信息量和近期参与状态，判断回复或暂不插话。
- **可靠唤醒**：私聊、回复 Bot 历史消息、平台确认的 `@Bot` 会直接进入回复流程。
- **人格称呼识别**：同时识别平台显示名、核心人格 `nickname` 和 `alias_names`。只有消息以独立称呼开头，且称呼后为结束、空白或标点时才视为召唤，避免将“讨论小蝶”之类的普通文本误判为呼叫。
- **分层决策**：明确场景由本地规则快速裁决；本地无法确定的灰区交给 `gray_zone_decider` 指定的裁决者（`sub_actor` 或 Jev）；裁决者不可用时按配置使用故障回退策略。
- **对话节奏控制**：Bot 刚回复、多人快速交流、消息面向其他成员或话题收尾时，会降低插话倾向。
- **工具渐进披露**：常用工具可常驻注入；大量外围工具可折叠，由模型通过 `explore_tools` 按需查询，避免工具列表稀释注意力。
- **工具调用保护**：需要查证或执行任务时优先调用工具；普通文本发送后自动结束，工具结果带来新信息时才补充，结合调用去重、无进展结束和复读抑制减少循环及重复发言。工具执行会记录本轮有界账本，并仅向后续去重提示回灌脱敏、限长的真实结果摘要。`[tools].tool_call_mode` 默认 `planning`，按顺序逐个提交调用；设为 `batch` 时会将本轮普通调用整批交给 Core 调度，但不保证绝对并行或执行顺序。可通过 `[tools].log_tool_calls` 开启工具名与脱敏参数日志。
- **连续输入 mailbox**：同一聊天流同一时刻只运行一个回合。生成期间到达的新消息会留在 mailbox，当前回合保留已领取输入的固定快照；成功后只确认该快照，失败或被打断时释放原输入并合并新输入，以避免漏读、误 flush 与并行回复。支持输入合并窗口和连续中断次数限制。
- **拟人化输出**：支持按语义分段发送、模拟打字延迟、情绪状态、主动性/走神和新消息到来时的打断重规划。分句以**不可拆单元**为最小粒度：表情序列（含肤色修饰、ZWJ 组合、变体选择符、keycap 与 tag 序列）与颜文字整体不参与切分，单元内部的标点也不会被当作语义边界，因此 `(´･ω･\`)`、`👨‍👩‍👧`、`(，´ω´，)` 不会被拆散。单个单元本身超过 `[humanize].max_segment_chars` 时按「宁可整条发出」处理，消息可能略超上限但内容不损坏；分段结果另有完整性自检兜底。
- **跨流全局心智**：在受控长度内共享情绪、各聊天流摘要和近期要闻，使 Bot 在不同会话中的人格与近期经历保持连贯。
- **可解释日志**：记录决策来源、评分、置信区间与原因，便于定位为何回复、静默、进入 `sub_actor` / Jev 或触发故障回退。
- **异常输出保护**：识别被上游错误包装为正常正文的供应商审核说明，以及被主 Agent 意外输出的完整 `sub_actor` 决策 JSON；在完整文本分段发送前阻止外发，并记录不包含原文的安全日志。
- **复杂任务 sub-agent**：以「任务：」前缀或任务关键词触发复杂请求时，进入独立任务运行时后台执行；sub-agent 拥有独立迭代/工具调用/超时预算与按任务类型固化的工具白名单，不影响主对话回合。任务完成或需要用户确认时自动唤醒主 Agent 汇报结果。
- **任务控制与续聊**：可用「暂停任务 / 继续任务 / 取消任务 / 查看任务状态」控制活动任务；用「补充任务：」或「补充资料：」前缀向任务追加输入。非直出任务的结果与确认提示由主 Agent 转述，直发任务的最终文本仍直接发送以避免重复打扰；推送行为由 `[tasks].report_task_events` 统一开关。
- **按需读取本地技能**：启动时扫描 `[skills].paths`，只向主 Agent 展示技能元数据；主 Agent 可用 `read_skill` 读取 `SKILL.md`，再用 `read_skill_reference` 读取已选技能的 Markdown 引用。技能正文仅为参考资料，不会执行脚本或授予额外权限；管理员可用 `agentic_skill refresh` 重新扫描目录。
- **可选原生图片输入**：启用 `[plugin].native_multimodal` 后，主 Agent 的 ACT 阶段可将消息里的图片与文本一并传给支持视觉的模型；默认关闭。单次请求的**图片总数**由 `[plugin].max_input_images` 限制（本轮新图与历史图共享同一预算，按「本轮优先、统一新→旧」选取，同一张图只内联一次），**图片总体积**由 `[plugin].max_input_image_bytes` 限制（默认 8MB，单张另有 12MB 硬上限）。体积按**原始图片字节**计量，不是 base64 文本长度——base64 约为原始体积的 4/3 倍，若按其计量，3.5MB 的手机随手拍会算成 4.7MB，8MB 预算只够 1 张，单张 12MB 上限也只等价于 9MB 原始图。预算不足的图片其占位符会改写为 `[图片已省略]`，同一张图重复出现时改写为 `[图片(同图,上文已提供)]`。其他阶段和后台任务仍仅使用文本上下文。

## 配置

插件配置由 `AgenticChatterConfig` 提供，主要分为以下部分：

| 配置节          | 用途                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| `[plugin]`      | 插件开关、主回复模型任务、私聊接管及私聊专用模型。                           |
| `[pipeline]`    | 阶段顺序、感知/规划/反思开关，以及单轮迭代和可见发言上限。                   |
| `[decision]`    | 本地评分权重、置信区间边界、语义任务、节奏冷却、灰区裁决者（sub_actor/jev）和故障回退策略。 |
| `[jev]`         | Jev（TypeSafe System One）灰区裁决：开关、API Key、失败回退、超时、重试与置信度门限。 |
| `[tools]`       | 工具可见性、折叠/黑名单、渐进探索、重复调用去重、工具调用模式及异常诊断记录。 |
| `[humanize]`    | 分段发送、打字延迟、QQBot C2C 实时流式输出、情绪、主动性、复读抑制和打断重规划。             |
| `[global_mind]` | 跨流情绪、摘要、近期要闻和注入长度限制。                                     |
| `[persona]`     | 私聊、群聊、系统提示词区块和额外系统提示词引导。                         |
| `[tasks]`       | 复杂任务运行时：开关、sub-agent 派发、任务预算、工具白名单/黑名单、检查点目录与完成回灌开关。 |
| `[skills]`      | 本地技能开关、技能根目录与技能描述长度限制。 |

### 灰区裁决者：sub_actor 或 Jev

`decide` 阶段的分层结构固定为：

```
硬规则 → 本地评分 + 置信区间 → 灰区裁决者 → 故障回退
```

**灰区**指的是本地评分区间既未越过 `local_reply_lower_bound`(0.35)、
也未越过 `local_silent_upper_bound`(0.2) 的那段区间，也就是「本地拿不准、
需要外部判断」的消息。灰区由谁裁决，由 `[decision].gray_zone_decider` 显式指定：

```toml
[decision]
gray_zone_decider = "sub_actor"   # sub_actor（默认）| jev

[jev]                            # 仅 decider = jev 时本节才生效
enabled = true
api_key = "ts_xxx"               # 请限制配置文件读取权限，不要提交到版本库
base_url = "https://api.typesafe.ai"   # 自建代理/镜像/本地网关改这里
fallback_decider = "sub_actor"    # sub_actor（默认）| silent
```

`base_url` 支持几种写法，路径 `/v1/systemone` 会自动补全，结尾有无斜杠均可：

| 填写 | 实际请求地址 |
| --- | --- |
| 留空 | `https://api.typesafe.ai/v1/systemone` |
| `https://api.typesafe.ai` | 同上 |
| `https://api.typesafe.ai/v1/systemone` | 同上（不会重复拼接） |
| `http://127.0.0.1:8080/jev-proxy` | `http://127.0.0.1:8080/jev-proxy/v1/systemone` |

| `gray_zone_decider` | 硬规则后 | 说明 |
| --- | --- | --- |
| `sub_actor`（默认） | → sub_actor | 与既有行为完全一致，不产生任何外部请求 |
| `jev` | → Jev | 灰区改由 Jev 裁决 |

硬规则在两种模式下都生效。是否还有本地评分直通，取决于 `decision.local_gate_enabled`：为 true 时先走评分直通、其余落入灰区交给裁决者；为 false 时不做评分直通也不计算 embedding，硬规则之后的每条消息都交给裁决者。

#### 两种 local_gate 模式

| `local_gate_enabled` | 硬规则之后 | embedding | 本地评分直通 |
| --- | --- | --- | --- |
| `true`（默认） | 评分直通 → 灰区 | 计算 | 有，越过边界即直接裁决 |
| `false` | 每条消息都交裁决者 | 不计算 | 无 |

`false` 时 `semantic_continuity` 与 `bot_history_continuity` 恒为 0
（图例已注明「本地规则未启用语义计算时恒为 0」），裁决者需主要依据对话原文判断。
`local_gate_enabled` 关闭时，硬规则与「走神」判定仍然生效。

#### Jev 灰区裁决

Jev 是 TypeSafe 的 System One 模型，返回 typed answer 与概率而非自由文本，
因此本节不涉及提示词模板与 JSON 解析。提问语料是「是否该回复」这一判断本身，
不包含相关性计算——embedding 连续性仍由本地评分算好后作为事实传入。

- **只有选 `jev` 才会发请求**。`decider = sub_actor` 时即使 `[jev]` 全部配置齐全，
  也不会产生任何外部调用。
- **提问语料**：`state` 只含 `bot` 身份、最近 `max_history_messages`(8) 条对话、
  本轮 `incoming` 新消息与 9 个本地信号值。单条消息按 `max_chars_per_message`(200)
  截断。Choice 的两个选项顺序固定（Jev 偏向排在前面的选项）。
- **bot 身份包含称呼**：`state.bot.names` 下发群里称呼 bot 的昵称与别名，
  并在 instructions 中写明「以这些名字开头或 @ 这些名字即召唤」。缺少这一步时，
  Jev 不认识自己的名字，冷却期内喊名字会被判成静默。
- **`direct_address` 信号**：本地规则已经确认消息在称呼 bot 时会下发该信号，
  避免 Jev 仅凭字面语义重新判断一次召唤。
- **失败处理**：超时、限流(429)、过载(529) 按 `max_retries`(2) 递增退避重试；
  鉴权(401)、参数错误(422)、网络错误、响应结构非法、置信度低于 `min_confidence`
  均视为不可用，按 `fallback_decider` 处理：
  - `sub_actor`（默认）：回退 sub_actor，沿用 `fallback_mode` 的既有故障回退策略。
  - `silent`：**直接静默本轮，绝不调用 sub_actor**。灰度实验需要纯净的 Jev
    行为、不希望混入 LLM 裁决时使用此项。
  - 无论哪种策略，**真实调用失败都会以 WARNING 记录细分原因码**，不会因为回退
    成功而被淹没。码表见 [docs/jev-troubleshooting.md](docs/jev-troubleshooting.md)。
- **总耗时受 `total_budget_seconds` 硬约束**（默认 4 秒）。单次尝试超时取 `timeout_seconds` 与剩余预算的较小值，预算耗尽后放弃剩余重试；群聊回合在此期间被阻塞，不应把该值设得过大。
- **置信度门限谨慎设置**。Jev 的 `confidence` 反映概率分布的集中程度，不等于判断
  正确率。建议先保持 `0`（全部接受）并开启 `log_answers` 观察真实分布后再调整。
- **日志可区分来源**：Jev 裁决的 `DecisionSource` 为 `jev`（日志措辞「Jev 决策」），
  与 `sub_actor` 分开，便于按来源筛选做灰度对比。

### 自定义系统提示词区块

可在 `[persona]` 中分别覆盖 Agent 的表达、行为和停止条件提示词：

```toml
[persona]
how_you_speak = """
请使用自然、简洁的表达方式，避免复述内部流程。
"""
how_you_act = """
需要查证时优先调用工具；工具存在依赖时等待前置结果。
"""
when_to_stop = """
完成当前任务后调用 end_turn，不要无休止地追加内容。
"""
```

- 三个字段只填写区块正文，不要重复写 `<how_you_speak>` 等标签；插件会自动保留结构化标签。
- 字段缺失、空字符串或仅包含空白时，使用插件内置原始模板，保持原有行为。
- 只修改其中一个字段不会影响另外两个区块。
- 修改后按现有插件配置重载或重启流程生效。


配置字段本身带有中文说明；生成或自动更新配置文件后可直接查看注释。人格称呼不在插件配置中重复维护，而是读取核心配置的：

```toml
[personality]
nickname = "Bot 本名"
alias_names = ["别名一", "别名二"]
```

## 使用与回复决策

插件的主聊天功能无需用户命令；启用后由聊天器根据收到的消息自动参与。另提供仅所有者可调用的 `agentic_skill refresh` 技能目录刷新命令。处理流程如下：

```text
新消息
  │
  ├─ perceive（可选）：概括当前情境与话题
  ├─ decide：判断是否自然参与
  ├─ plan（可选）：形成行动意图
  ├─ act：生成文本、调用工具或执行 Action
  └─ reflect（可选）：更新情绪与跨流摘要
```

`pipeline.stage_order` 声明的是候选阶段及其顺序，并不表示列表中的阶段一定执行：`perceive`、`plan`、`reflect` 分别受 `enable_perceive`、`enable_plan`、`enable_reflect` 控制，`decide` 受 `decision.enabled` 控制；关闭的阶段会从实际执行顺序中移除。`act` 是必需阶段，即使未写入 `stage_order` 也会自动补上。其他插件也可以通过 `PipelineService` 注册自定义阶段，并把阶段名加入 `pipeline.stage_order`。

`decision.enabled` 是 `decide` 阶段的总开关。关闭后，即使 `stage_order` 包含 `decide`，实际管线也会移除该阶段，`local_gate_enabled`、本地评分边界和灰区裁决者均不生效；其余阶段仍按各自开关过滤后执行。开启后，若 `stage_order` 未包含 `decide`，插件会在 `act` 前自动补入，并确保它位于 `plan`、`act` 之前。

回复决策启用后的优先级如下：

1. **硬规则**：私聊、回复 Bot 发出的消息、平台明确确认当前 Bot 被 `@`。
2. **本地决策**：提取本地信号并计算评分与置信区间；明确组合规则可直接回复或静默。
3. **灰区裁决**：评分区间仍处于灰区时，由 `decision.gray_zone_decider` 指定的裁决者判断——`sub_actor`（默认）交给专用模型，`jev` 交给 Jev。
4. **故障回退**：`sub_actor` 无法调用、无有效输出或返回格式非法时按 `fallback_mode` 处理：
   - `contextual`：保留明显值得响应的上下文机会；近期已经回复时可抑制重复插话；
   - `fail_open`：默认回复；
   - `fail_closed`：默认静默。

本地评分不是“回复概率”，而是衡量当前 Bot 是否适合介入；评分区间越宽，说明本地越不确定，越可能交给灰区裁决者。

### 私聊接管与专用模型

`[plugin]` 可单独控制私聊接管，并为私聊主 Agent 指定模型：

```toml
[plugin]
enabled = true
model_task = "actor"
private_enabled = true
private_model_name = "private-chat-model"
```

- `private_enabled` 默认开启；关闭后不注册私聊 Chatter，新私聊会由其他兼容 Chatter 接管；修改后需重载插件或重启；
- `private_model_name` 填写 `config/model.toml` 中 `[[models]].name`，不是上游 `model_identifier` 或 `[model_tasks.*]` 的 task key；
- 留空时私聊继续使用 `model_task`；指定后私聊 `act` 主循环固定使用该单模型，不使用主 task 的多模型回退列表；
- 该覆盖不影响 perceive、plan、`sub_actor`、embedding 和全局心智摘要等辅助任务。

### 工具调用超时管理

主对话与复杂任务执行器的普通工具提交使用同一套超时保护，仍走框架原有的 Tool / Action / Agent Manager，保留事件、历史与缓存路径。

```toml
[tools]
tool_call_mode = "planning"
tool_timeout_seconds = 30.0
tool_cancel_grace_seconds = 1.0
tool_timeout_overrides = { "my_plugin:tool:slow_query" = 90.0 }
```

- `tool_timeout_seconds` 默认 30 秒，设置为 `0` 关闭本插件的工具超时保护（覆盖表也不生效）；覆盖表按完整组件签名匹配，值必须是有限正数。
- `planning` 仍逐个提交，每个提交使用对应预算；`batch` 仍整批提交，整批预算是各调用预算之和，不承诺每个调用单独到点终止。当前核心 dispatcher 逐个等待 Manager；插件不会把 batch 拆开，也不擅自并行化。
- `agentic_chatter:action:dispatch_task` 未显式覆盖时，预算自动取普通预算与 `tasks.timeout_seconds + 30` 的较大值，避免默认 30 秒误杀长任务。
- 超时后保留已返回的结果正文，只为缺项补写 TOOL_RESULT；整批 dispatcher 尚未返回的成功状态不作猜测，因此超时提交中的调用均保守记为失败，不计成功进展。迟到结果不会再写入当前响应链。
- 超时或外部取消后，请求取消该次 dispatcher 及其框架执行任务，最多等待 `tool_cancel_grace_seconds`（默认 1 秒，可设 0–10 秒）。外部取消仍向上传播。不会使用全局任务差集取消其它聊天流。
- 适配通过临时、链式保留既有 event-loop task factory 与 ContextVar 识别框架推进任务；在原 factory 构造期间记录原协程归属，因此标准 eager 调度下的两层/三层嵌套也能追踪，不用 wrapper 改变交给旧 factory 的协程身份/code。已验证普通及标准 eager factory；自行包装协程、使 `Task.get_coro()` 不再返回传入协程的第三方 factory 不在此归属保证范围内。
- 现代 factory 接收并转发全部任务参数；旧签名仅适配 name/context，不支持的参数明确报错，不捕获 TypeError 重试。Python 3.11/3.12 已运行回归；3.13.3/3.14 的 name/context 参数契约已有模拟测试，但尚未在这些版本实机运行，不能视为完整跨版本认证。
- 取消时会清理未启动的业务 coroutine，并有界执行暂停迭代器的 `aclose()`；清理和执行取消共用宽限。0 秒不等待清理完成，但仍给予清理首次执行机会；拒绝取消的清理仍可能继续。同一 execution 限定在同一事件循环使用，跨 scope 只允许一次清理尝试。关闭前先同步终止执行，禁止再次 resume；终止状态不等于资源已释放。被其它 scope 成功接手的执行不再由旧 scope 主动清理，创建失败留下的旧任务不会被误判为接手。清理任务创建失败会记录“事件=工具清理创建失败”，不重试，也不阻断其余受管任务的取消；此时不保证资源释放成功。日志“已追踪任务取消未收敛”仅描述已登记任务集合，不证明资源全部释放或任意派生工作已停止。
- 适配依赖核心推进协程的 code、frame 字段及执行状态契约，核心升级后应重跑插件契约测试。宽限结束后恢复原 factory；如果其它代码已替换 factory，不覆盖它。
- 取消是协作式的：忽略取消的协程仍可能继续运行，工具自行派生的普通后台任务不在此追踪范围，线程与外部副作用无法强杀或回滚。事件循环被同步阻塞时，超时回调也无法准时触发。不要自动重试可能已有副作用的调用，应先查证状态或向用户说明。

### 普通对话失败后的处理

- 模型请求的重试由核心 LLM 层管理。错误传播到插件后，普通用户输入回合进入失败终态，不再把整批旧输入重新入队，也不自动重发已发送文本或重做工具调用。
- 失败与真实新输入中断分开处理：只有 `interrupted_by_new_input` 标记的输入打断仍按既有规则重新规划；失败后不继续执行反思等后续阶段。本阶段未改变无用户输入的后台任务汇报重试策略。
- 能唯一匹配的失败输入会归入聊天历史，仅表示输入已处理，不表示业务成功；日志使用 `事件=回合失败终止`，记录失败类别、异常类型、可取得的 HTTP 状态、输入键摘要与已记录动作计数，不输出供应商原始错误正文。
- 消息匹配或归档失败时，原消息留在进程内隔离区，不删除、不重新执行。失败键在当前 mailbox 生命周期内不驱逐；最近 32 条失败摘要可通过 `recent_failed_turns()` 检查，未归档原消息可通过 `failed_messages()` 检查。摘要淘汰不会解除隔离。
- 失败后零秒等待只用于再检查一次队列，处理失败期间已到达的新输入；没有可领取消息后返回普通的新消息等待，不启动定时业务重试。
- 本阶段没有自动恢复失败回合的命令，也没有跨重启幂等账本。修正模型或网关问题后，用户可发起新的明确请求；对可能已产生副作用的动作，应先核实结果再决定是否重试。已归档输入遵循核心历史保留策略，不构成永久失败原文存档。
- 失败隔离只在当前进程有效；未归档原消息和失败键不会因容量限制自动淘汰，长期持续故障时内存可能增长，应及时处理故障。重启或清空 mailbox 后不保证仍能阻止旧输入重放。

### 供应商异常请求体诊断

当模型把 Google/Gemini 审核错误包装成普通正文时，插件会在发送前拦截该正文。默认仅记录不含原文的安全日志；如需复盘触发拦截的插件可见请求上下文，可显式开启以下诊断开关：

```toml
[tools]
record_provider_error_request_body = true
```

- 默认关闭；仅在已命中供应商异常正文拦截时记录，不命中时不会序列化或创建文件。
- 记录追加到 `data/agentic_chatter/provider_error_requests.jsonl`，每行一条 UTF-8 JSON 对象。该快照是插件可见的 `response.payloads`，不保证等同供应商最终的原生 HTTP 请求体。
- system 提示词中的昵称、别名、人格、身份、背景和回复风格会替换为稳定变量；用户输入、历史上下文、工具定义、Tool Call 及 Tool Result 按受控长度保留。
- 不记录 reasoning；图片、音频等文件只保留类型和必要元数据，不写入二进制或 base64；工具参数与结果中的 `token`、`secret`、`password`、`authorization`、`cookie`、`api_key` 等常见敏感字段会脱敏。
- 记录仍可能包含用户输入和工具数据。开启前请确认本地数据目录的访问权限，并按部署环境的数据留存规则清理该文件。

### QQBot C2C 实时流式输出

在 `[humanize]` 中启用 `streaming_enabled` 后，QQBot C2C 私聊会按 LLM 可见文本 token 实时更新同一条流式消息。还需在 `qqbot_adapter` 中启用 `features.streaming`。

- 默认 Service 签名为 `qqbot_adapter:service:qqbot`，可通过 `streaming_service_signature` 调整；
- `streaming_initial_chars` 控制累计多少个安全可见字符后启动流；
- `streaming_update_min_chars` 控制正文至少增加多少字符后请求更新，时间节流由 QQBot Adapter 自身处理；
- 只处理 `text_delta`，结构化 reasoning 及 `<think>` / `<analysis>` / `<reasoning>` 块不会发给用户；
- 非 QQ、群聊、元数据不完整、流式 Service 不可用或启动失败时自动使用原有普通分段发送；
- controller 已启动后的更新或收尾失败不会再补发普通消息，以避免重复内容。

### 复杂任务 sub-agent

以「任务：」/「任务:」前缀的消息，或包含「查询资料」「分析代码」「修改代码」等关键词的复杂请求，会进入独立任务运行时后台执行：

- **触发与路由**：消息带任务前缀直接创建通用任务；任务运行期间的新消息按控制命令（暂停/继续/取消/查看状态）、补充输入（「补充任务：」「补充资料：」或任务 ID 前缀）与普通聊天自动分流，只有任务相关内容进入后台执行器。
- **多任务并发**：同一聊天流允许同时运行多个任务，上限由 `[tasks].max_concurrent_tasks`（默认 2，最大 8）控制；达到上限时新建请求会被明确拒绝并提示当前任务列表，而不是静默失败。
- **任务管理工具**：主 Agent 可调用 `manage_tasks` 工具（list/status/cancel）查询本流全部任务的状态与进度、取消指定任务；`list` 输出中的序号或任务 ID 前 6 位以上前缀可用于精确定位。
- **控制消息词表**：以下整串消息会被识别为控制命令（去首尾标点后精确匹配）：
  - 取消：取消任务/取消这个任务/取消掉任务/停止任务/停掉任务/终止任务/放弃任务/别做了/别弄了/不用做了/不要做了/不用继续了
  - 暂停：暂停任务/任务暂停/先暂停任务
  - 继续：继续任务/恢复任务/接着做任务/继续做任务
  - 状态：任务状态/任务进度/任务列表/任务怎么样了/查看任务状态/查看任务进度/查看任务/有什么任务
  - 带序号变体：取消任务2、暂停任务1、#2 取消 等可精确定位任务。控制命令在多个任务运行时会先回复任务列表要求消歧。
- **独立预算**：每个任务有独立的迭代次数、工具调用次数、失败次数、无进展步数与超时预算（`[tasks]` 配置），后台执行不影响主对话回合，主对话被打断或迭代耗尽也不会终止后台任务。
- **超时兜底**：任务协程悬挂（如网络请求无响应）超过预算时限 + 30 秒宽限后，会被自动暂停为 PAUSED 状态并发送提示；用户可发送「继续任务」恢复。注意：悬挂在不可恢复网络 I/O 上的任务恢复后可能再次超时。
- **工具权限**：按任务类型模板固化白名单（通用/查询类任务默认只读：`read*`、`search*`、`list*`、`query*`、`test*`、`explore*`；代码修改类任务额外放开 `edit*`），`[tasks].denied_tools` 全局黑名单优先级最高，需确认的工具（如 `delete*`）会转入等待用户输入状态。工具调用名经 schema 前缀还原后统一判定，注入与执行两侧一致。`manage_tasks` 不在任务白名单内，任务内的 sub-agent 看不到它。
- **结果汇报（回灌）**：非直出任务成功完成或任务等待用户输入时，插件会唤醒原聊天流的主 Agent，以独立汇报轮把结果自然转述给用户；等待确认时会说明需要用户做什么并提示「补充任务：」用法。直发任务的结果文本仍直接发送，不再重复推送；`[tasks].report_task_events`（默认开启）控制整体回灌行为。
- **持久化与恢复**：任务检查点写入 `[tasks].checkpoint_directory`（默认 `data/agentic_chatter/tasks`），插件重载或重启后恢复本流的活动任务并继续后台执行。
- **已知限制**：任务列表序号随任务增删漂移，长 ID 前缀是更稳定的引用方式；「别做了」等短语仅在流内存在活动任务时才会被识别为控制命令。

## 对外提供的组件

插件加载后会注册以下组件：

- `agentic_chatter:chatter:agentic`：`AgenticChatter`，群聊聊天器；
- `agentic_chatter:chatter:agentic_private`：`AgenticPrivateChatter`，可配置启用的私聊聊天器；
- `agentic_chatter:chatter:agentic_discuss`：`AgenticDiscussChatter`，讨论组聊天器；
- `agentic_chatter:service:pipeline`：`PipelineService`，供其他插件注册自定义管线阶段、读写全局心智；
- `agentic_chatter:tool:explore_tools`：`ExploreToolsTool`，供模型按需展开折叠工具；
- `agentic_chatter:tool:manage_tasks`：`ManageTasksTool`，供模型列出/查询/取消当前对话的后台任务；
- `agentic_chatter:action:say`：`SayAction`，通过必填 `target_type`（`reply`/`mention`）与 `target_id` 执行引用回复或群聊 @；
- `agentic_chatter:action:end_turn`：`EndTurnAction`，结束当前 Agent 回合；
- `agentic_chatter:action:stop_conversation`：`StopConversationAction`，停止当前会话处理；
- `agentic_chatter:action:dispatch_task`：`DispatchTaskAction`，供主 Agent 以同步方式把复杂任务派发给 sub-agent 执行（受 `[tasks].action_enabled` 控制）；
- `agentic_chatter:service:tasks`：`TaskRuntimeService`，供其他插件查询与操控任务运行时。

## 数据边界与限制

- 插件会读取核心人格、当前聊天上下文、历史消息、工具信息和模型响应，用于生成回复和参与决策。
- 开启 `[tools].record_provider_error_request_body` 后，命中供应商异常正文拦截时会额外持久化受控请求上下文；其内容范围、脱敏和路径见“供应商异常请求体诊断”。
- 跨流全局心智会持久化插件配置允许范围内的情绪、摘要和近期要闻；具体保留数量与长度由 `[global_mind]` 配置节限制。
- 插件可能调用已注册的其他插件工具或 Action；这些副作用由被调用组件自身的权限和实现决定。
- 插件不负责平台连接、账号认证或密钥管理；平台凭据应由核心配置和对应 Adapter 管理。
- 这是 Agent 式生成插件，模型输出质量、可用性和平台能力会影响最终回复；建议先在测试环境验证配置。

## 开发与验证

在 Neo-MoFox 项目根目录执行：

```bash
ruff check plugins/agentic_chatter/
mpdt plugin check plugins/agentic_chatter --level warning
python -X utf8 -m pytest plugins/agentic_chatter/tests -q --no-cov -p no:randomly
python -X utf8 "$SKILL_DIR/scripts/verify_plugin.py" plugins/agentic_chatter
```

`verify_plugin.py` 用于验证框架真实加载、组件注册和卸载；`$SKILL_DIR` 为本机 `mofox-plugin-workflow` Skill 的安装目录。

## 日志排查

每轮决策会输出类似日志：

```text
回复判断：回复（方式：本地判断；原因：明确称呼了我，且当前没有明显冲突信号）
```

常见“方式”含义：

| 方式         | 含义                                              |
| ------------ | ------------------------------------------------- |
| `直接唤醒`   | 命中私聊、回复 Bot 或可靠 `@Bot` 等硬规则。       |
| `本地判断`   | 本地组合规则或评分区间已能明确决策。              |
| `子决策模型` | 本地处于灰区，由 `sub_actor` 完成最终裁决。       |
| `Jev 决策`   | 本地处于灰区，由 Jev 完成最终裁决。              |
| `故障回退`   | `sub_actor` 异常后按 `fallback_mode` 得出的结果。 |

DEBUG 日志还会输出评分、置信区间、当前话题相关度、Bot 历史相关度、是否命中直接称呼
以及内部原因的中文说明。

INFO 日志显示“其他判断信号”，说明该原因码尚未被中文映射表收录，多半是 `sub_actor`
返回了自定义码。Jev 的死因对照表见 [docs/jev-troubleshooting.md](docs/jev-troubleshooting.md)。

Jev 不可用时（无论 `fallback_decider` 是 `sub_actor` 还是 `silent`）都会以 WARNING
输出一行，带细分死因与诊断字段：

```text
[1a2b3c4d] Jev 不可用：Jev 鉴权失败（HTTP 401，尝试 1 次，端点 https://api.typesafe.ai/v1/systemone），灰区回退子决策模型
```

失败原因共 15 类，覆盖限流、过载、鉴权、额度、超时、网络、预算耗尽、响应结构异常等，
日志中一律以中文呈现。完整对照表与按现象反查见
[docs/jev-troubleshooting.md](docs/jev-troubleshooting.md)。

## 许可证与维护者

本插件由 qf 维护，使用 GPL-3.0 许可证发布。仓库地址由市场发布参数指定为 `qingfeng66640/agentic_chatter`。

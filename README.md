# Agentic Chatter

`Agentic Chatter` 是 Neo-MoFox 的 Agent 式聊天器插件（v0.2.2）。它将一轮回复拆分为可编排的处理阶段，并在群聊中先判断“是否应当参与”，再决定如何生成、执行和收尾回复，避免 Bot 对每条消息机械接话。

- **维护者**：qf
- **仓库**：`qingfeng66640/agentic_chatter`
- **最低核心版本**：`1.2.0-alpha`
- **许可证**：GPL-3.0，详见 [LICENSE](LICENSE)

## 安装与启用

将插件目录放入 Neo-MoFox 的插件目录，确保 `manifest.json` 与 `plugin.py` 位于插件根目录。启动 Neo-MoFox 后，插件管理器会按 manifest 加载 `agentic_chatter`；首次加载后会生成或更新插件配置文件。

插件不包含平台适配器，必须先配置并启用对应平台的 Adapter。插件也依赖 Neo-MoFox 核心提供的模型任务、上下文、工具和消息发送能力，不需要额外的 Python 第三方依赖。

## 功能概览

- **分阶段对话流程**：按 `perceive → decide → plan → act → reflect` 处理消息；阶段顺序可通过配置调整。
- **自然参与决策**：综合直接称呼、问题请求、话题与 Bot 历史连续性、多人对话、他人定向、话题收尾、消息信息量和近期参与状态，判断回复或暂不插话。
- **可靠唤醒**：私聊、回复 Bot 历史消息、平台确认的 `@Bot` 会直接进入回复流程。
- **人格称呼识别**：同时识别平台显示名、核心人格 `nickname` 和 `alias_names`。只有消息以独立称呼开头，且称呼后为结束、空白或标点时才视为召唤，避免将“讨论小蝶”之类的普通文本误判为呼叫。
- **分层决策**：明确场景由本地规则快速裁决；本地无法确定的灰区才调用 `sub_actor`；模型不可用时按配置使用故障回退策略。
- **对话节奏控制**：Bot 刚回复、多人快速交流、消息面向其他成员或话题收尾时，会降低插话倾向。
- **工具渐进披露**：常用工具可常驻注入；大量外围工具可折叠，由模型通过 `explore_tools` 按需查询，避免工具列表稀释注意力。
- **工具调用保护**：需要查证或执行任务时优先调用工具；普通文本发送后自动结束，工具结果带来新信息时才补充，结合调用去重、无进展结束和复读抑制减少循环及重复发言。
- **拟人化输出**：支持按语义分段发送、模拟打字延迟、情绪状态、主动性/走神和新消息到来时的打断重规划。
- **跨流全局心智**：在受控长度内共享情绪、各聊天流摘要和近期要闻，使 Bot 在不同会话中的人格与近期经历保持连贯。
- **可解释日志**：记录决策来源、评分、置信区间与原因，便于定位为何回复、静默、进入 `sub_actor` 或触发故障回退。

## 配置

插件配置由 `AgenticChatterConfig` 提供，主要分为以下部分：

| 配置节          | 用途                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| `[plugin]`      | 插件开关、主回复模型任务、私聊接管及私聊专用模型。                           |
| `[pipeline]`    | 阶段顺序、感知/规划/反思开关，以及单轮迭代和可见发言上限。                   |
| `[decision]`    | 本地评分权重、置信区间边界、语义任务、节奏冷却、`sub_actor` 和故障回退策略。 |
| `[tools]`       | 工具可见性、折叠/黑名单、渐进探索及重复调用去重。                            |
| `[humanize]`    | 分段发送、打字延迟、QQBot C2C 实时流式输出、情绪、主动性、复读抑制和打断重规划。             |
| `[global_mind]` | 跨流情绪、摘要、近期要闻和注入长度限制。                                     |
| `[persona]`     | 私聊、群聊及额外系统提示词引导。                                             |

配置字段本身带有中文说明；生成或自动更新配置文件后可直接查看注释。人格称呼不在插件配置中重复维护，而是读取核心配置的：

```toml
[personality]
nickname = "Bot 本名"
alias_names = ["别名一", "别名二"]
```

## 使用与回复决策

插件不提供用户命令；启用后由聊天器根据收到的消息自动参与。处理流程如下：

```text
新消息
  │
  ├─ perceive（可选）：概括当前情境与话题
  ├─ decide：判断是否自然参与
  ├─ plan（可选）：形成行动意图
  ├─ act：生成文本、调用工具或执行 Action
  └─ reflect（可选）：更新情绪与跨流摘要
```

其中 `act` 是必需阶段；其余内置阶段可单独启用或关闭。其他插件也可以通过 `PipelineService` 注册自定义阶段，并把阶段名加入 `pipeline.stage_order`。

回复决策只在启用 `decision.enabled` 时生效，优先级如下：

1. **硬规则**：私聊、回复 Bot 发出的消息、平台明确确认当前 Bot 被 `@`。
2. **本地决策**：提取本地信号并计算评分与置信区间；明确组合规则可直接回复或静默。
3. **`sub_actor` 决策**：评分区间仍处于灰区时，将近期消息、历史和本地信号交给专用模型进一步判断。
4. **故障回退**：`sub_actor` 无法调用、无有效输出或返回格式非法时按 `fallback_mode` 处理：
   - `contextual`：保留明显值得响应的上下文机会；近期已经回复时可抑制重复插话；
   - `fail_open`：默认回复；
   - `fail_closed`：默认静默。

本地评分不是“回复概率”，而是衡量当前 Bot 是否适合介入；评分区间越宽，说明本地越不确定，越可能进入 `sub_actor`。

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

### QQBot C2C 实时流式输出

在 `[humanize]` 中启用 `streaming_enabled` 后，QQBot C2C 私聊会按 LLM 可见文本 token 实时更新同一条流式消息。还需在 `qqbot_adapter` 中启用 `features.streaming`。

- 默认 Service 签名为 `qqbot_adapter:service:qqbot`，可通过 `streaming_service_signature` 调整；
- `streaming_initial_chars` 控制累计多少个安全可见字符后启动流；
- `streaming_update_min_chars` 控制正文至少增加多少字符后请求更新，时间节流由 QQBot Adapter 自身处理；
- 只处理 `text_delta`，结构化 reasoning 及 `<think>` / `<analysis>` / `<reasoning>` 块不会发给用户；
- 非 QQ、群聊、元数据不完整、流式 Service 不可用或启动失败时自动使用原有普通分段发送；
- controller 已启动后的更新或收尾失败不会再补发普通消息，以避免重复内容。

## 对外提供的组件

插件加载后会注册以下组件：

- `agentic_chatter:chatter:agentic`：`AgenticChatter`，群聊聊天器；
- `agentic_chatter:chatter:agentic_private`：`AgenticPrivateChatter`，可配置启用的私聊聊天器；
- `agentic_chatter:chatter:agentic_discuss`：`AgenticDiscussChatter`，讨论组聊天器；
- `agentic_chatter:service:pipeline`：`PipelineService`，供其他插件注册自定义管线阶段、读写全局心智；
- `agentic_chatter:tool:explore_tools`：`ExploreToolsTool`，供模型按需展开折叠工具；
- `agentic_chatter:action:say`：`SayAction`，发送文本消息；
- `agentic_chatter:action:end_turn`：`EndTurnAction`，结束当前 Agent 回合；
- `agentic_chatter:action:stop_conversation`：`StopConversationAction`，停止当前会话处理。

## 数据边界与限制

- 插件会读取核心人格、当前聊天上下文、历史消息、工具信息和模型响应，用于生成回复和参与决策。
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
| `故障回退`   | `sub_actor` 异常后按 `fallback_mode` 得出的结果。 |

DEBUG 日志还会输出评分、置信区间、当前话题相关度、Bot 历史相关度、是否命中直接称呼以及内部原因码。若 INFO 日志显示“其他判断信号”，通常表示 `sub_actor` 返回了尚未被中文映射表收录的原因码；可在同轮 DEBUG 的“内部原因”中查看原始码。

## 许可证与维护者

本插件由 qf 维护，使用 GPL-3.0 许可证发布。仓库地址由市场发布参数指定为 `qingfeng66640/agentic_chatter`。

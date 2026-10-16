# Jev 失败排查

本文档说明 Jev 灰区裁决**失败时如何看出死因**。

## 一、为什么要看这份文档

Jev 不可用时，插件不会中断回合，而是按 `[jev].fallback_decider` 回退。回退本身
是兜底、不是崩溃，很容易被忽略——bot 看起来"在正常聊天"，实际上灰区裁决已经
全线停摆，而日志里查不出死因。

现在每一类失败都会被分成独立类别并给出中文说明。插件内部仍保留一个稳定的
机器可读原因码（用于聚合统计与自动化检索），但**日志输出的一律是中文**，
排查时看到什么就照着下面第三节处理。

## 二、日志在哪

真实调用失败（HTTP 层、协议层、传输层）一律以 **WARNING** 留痕，覆盖两种
回退策略——否则"回退成功"会掩盖"Jev 已经死了"这件事：

```text
[1a2b3c4d] Jev 不可用：Jev 鉴权失败（HTTP 401，尝试 1 次，端点 https://api.typesafe.ai/v1/systemone），灰区回退子决策模型
[1a2b3c4d] Jev 不可用：Jev 被限流（HTTP 429，尝试 3 次，耗时 4.12 秒，端点 https://api.typesafe.ai/v1/systemone），回退策略为静默，本轮灰区静默，不调用子决策模型
```

成功状态码（2xx）不会显示：请求既已到达上游，写一个「HTTP 200」只会让人误判为上游报错。

括号内的诊断信息：

| 内容 | 含义 |
| --- | --- |
| `HTTP 401` | 上游返回的原始状态码；无响应或为 2xx 时不出现 |
| `尝试 1 次` | 已消耗的尝试次数（含首次） |
| `耗时 3.21 秒` | 本次裁决的累计耗时 |
| `端点 …` | 实际请求的完整端点，**已脱敏** |

### 端点脱敏

`[jev].base_url` 由使用者填写，自建网关或代理常写成
`https://svc:token@gw.internal/v1/systemone` 这种带凭据的形式。端点进日志前
会脱敏，只抹凭据、保留协议/主机/端口/路径：

| 写法 | 日志中的样子 |
| --- | --- |
| `https://api.typesafe.ai/v1/systemone` | 原样输出 |
| `https://svc:token@gw.internal/v1/systemone` | `https://[REDACTED]@gw.internal/v1/systemone` |
| `svc:token@gw.internal:8080/v1/systemone`（省略协议） | `[REDACTED]@gw.internal:8080/v1/systemone` |
| `.../v1/systemone?api_key=sk-live&model=jev-latest` | `.../v1/systemone?api_key=[REDACTED]&model=jev-latest` |

脱敏还会把换行等控制字符折成空格并限长 200 字符：`base_url` 里的换行会被
日志后端渲染成伪造的第二行，破坏"一行即一件事"的可信度。

**已知不覆盖的情形**：形如 `https://host/v1/sk-live-abc` 的**路径内**凭据不会
被脱敏。路径是排查连接问题最关键的信息，而其中任意一段都无法与普通路径参数
区分，盲目脱敏会让正常端点也失去可读性。若你的网关把 Key 放在路径里，请勿依赖
日志脱敏，应自行确认该配置不会外泄。

占位符沿用插件既有的 `[REDACTED]`（与工具参数脱敏一致），运维用一个串就能
扫全插件日志。因此**userinfo 与查询串里的凭据不会出现在日志里**；排查连接问题
时请以主机与路径为准，不要期待日志里能看到用户名或 Key。

配置前置检查（未启用、未配置 API Key）**不是**请求失败，仍留在 DEBUG：
灰区每条消息都会走到这里，按灰区频率打 WARNING 会变成常态噪音。

`silent` 回退模式下，决策日志的「原因」会直接显示中文死因：

```text
回复判断：暂不回复（方式：故障回退；原因：包含问题或请求、Jev 被限流）
```

## 三、死因对照表

按日志里出现的措辞查。每行第一列就是日志中会看到的文案。

### 状态码类

| 日志中的措辞 | 含义 | 该做什么 |
| --- | --- | --- |
| Jev 被限流 | HTTP 429，重试用尽 | 降低灰区频率或调大 `retry_backoff_seconds`；配额本身不足则等配额刷新 |
| Jev 服务过载 | HTTP 529，重试用尽 | 属上游容量问题，通常自愈；频繁出现考虑调大 `total_budget_seconds` 或减少调用 |
| Jev 鉴权失败 | HTTP 401 / 403 | 401 核对 `[jev].api_key` 与 System One 权限；403 在自建网关/WAF 下也常表示来源 IP 或路径不被允许，需一并核对网关策略 |
| Jev 额度不足 | HTTP 402 | 充值；这是账户问题，调参无效 |
| Jev 端点或模型不存在 | HTTP 404 | 核对 `[jev].base_url` 与 `[jev].model`；`base_url` 可写到域名或含 `/v1/systemone`，路径会自动补全 |
| Jev 拒绝了请求 | HTTP 400 / 422 | 多为 state 体积或 schema 问题；先调小 `max_history_messages` / `max_chars_per_message` 验证 |
| Jev 返回异常状态码 | 未细分的状态码 | 按括号里的原始码自行判断 |

### 传输与预算类

> **预算耗尽不会顶替死因。** 若因 `total_budget_seconds` 提前收手，日志仍报上游
> 报出来的那一类失败（如「Jev 请求超时」），预算信息只附加在消息末尾。默认配置下
> 上游宕机必然先耗尽预算，若把它误报成预算问题，排查方向会被直接带偏。

| 日志中的措辞 | 含义 | 该做什么 |
| --- | --- | --- |
| Jev 请求超时 | 单次尝试超时，重试用尽 | 调大 `timeout_seconds`，或接受当前超时上限 |
| Jev 网络不可达 | 连接被拒 / DNS / TLS 等传输层失败 | 确认 `base_url` 可达、出网未被拦；代理场景检查 `base_url` 是否写全协议 |
| Jev 裁决超出总预算 | 超出 `total_budget_seconds`，剩余重试被放弃 | 该值是灰区裁决的**总**硬上限。调大它会同步拉长回合阻塞时间，需权衡 |

### 协议与答案类

| 日志中的措辞 | 含义 | 该做什么 |
| --- | --- | --- |
| Jev 响应不是合法 JSON | 200 但响应不是 JSON | 通常是网关返回了 HTML 错误页；核对端点是否指向正确路径 |
| Jev 响应结构非法 | 200 + JSON 但结构非法，或缺 `answers.action` | 上游协议变更或被中间层改写；记录原始响应后向 TypeSafe 反馈 |
| Jev 答案结构非法 | 答案的类型、取值或置信度非法 | 属上游答案异常，通常与「响应结构非法」同时排查 |
| Jev 置信度低于门限 | 置信度低于 `min_confidence` 门限 | **不是调用故障**，是一次有效裁决被门限挡回。该做的是调小 `min_confidence`，不是排查网络 |
| Jev 调用失败（原因未分类） | 原因未分类 | 属兜底，出现即说明有未覆盖的失败出口，值得反馈 |

无法判定故障类别的异常不会被伪装成已知类别，日志中写作「Jev 调用失败（<类名>）」——
类名往往就是定位根因的唯一线索。看到这种措辞意味着还有失败出口没覆盖。

## 四、按现象反查

| 现象 | 先看 |
| --- | --- |
| 灰区完全不表态，bot 变得很安静 | 「Jev 未启用」/「Jev 未配置 API Key」（DEBUG），或静默回退下的任一失败行 |
| 偶发丢消息，过一会儿又好了 | 「Jev 被限流」/「Jev 服务过载」/「Jev 请求超时」 |
| 完全不发言，日志刷同一条 | 「Jev 鉴权失败」/「Jev 额度不足」/「Jev 拒绝了请求」 |
| 开了 `log_answers` 却看不到答案 | 「Jev 响应不是合法 JSON」/「Jev 响应结构非法」 |
| 裁决成功但仍走回退 | 「Jev 置信度低于门限」，调小 `min_confidence` |

## 五、事件键对照表

运行日志的事件键已由英文改为中文。若你有基于旧键名的告警规则、看板或日志检索脚本，
按下表替换即可。中文同样可检索（`grep 多模态图片超限`），只是字面量变了。

| 旧键 | 新键 |
| --- | --- |
| `event=claim_commit_failed` | `事件=消息认领提交失败` |
| `event=claim_release_failed` | `事件=认领释放失败` |
| `event=owner_release_failed` | `事件=持有者释放失败` |
| `event=claim_match_failed` | `事件=消息确认匹配失败` |
| `event=turn_committed` | `事件=回合已确认` |
| `event=turn_released` | `事件=回合已释放` |
| `event=turn_delayed_for_input_merge` | `事件=等待合并输入` |
| `event=input_merged` | `事件=输入已合并` |
| `event=input_interrupt_suppressed` | `事件=输入打断已抑制` |
| `event=input_interrupt_triggered` | `事件=输入打断已触发` |
| `event=task_created` | `事件=已创建复杂任务` |
| `event=task_claim_commit_failed` | `事件=任务认领提交失败` |
| `event=task_background_started` | `事件=任务已转入后台` |
| `event=task_turn_committed` | `事件=任务消息已确认` |
| `event=task_completion_reported` | `事件=任务完成已回灌主Agent` |
| `event=jev_decided` | `事件=Jev裁决灰区` |
| `event=jev_answer` | `事件=Jev应答` |
| `event=act_iteration` | `事件=行动迭代` |
| `event=act_auto_end` | `事件=行动自动结束` |
| `event=tool_call` | `事件=调用工具` |
| `event=tool_result` | `事件=工具结果` |
| `event=stream_duplicate_already_emitted` | `事件=流式输出重复已跳过` |
| `event=provider_error_text_intercepted` | `事件=已拦截供应商异常文本` |
| `event=provider_error_request_recorded` | `事件=供应商异常请求体已记录` |
| `event=provider_error_request_record_failed` | `事件=供应商异常请求体记录失败` |
| `event=reply_decision_json_intercepted` | `事件=已拦截子决策JSON` |
| `event=framework_message_line_intercepted` | `事件=已拦截框架消息行` |
| `event=multimodal_image_limit` | `事件=多模态图片超限` |
| `event=multimodal_image_invalid` | `事件=多模态图片载荷非法` |

此外，随事件一起输出的诊断字段名也改为中文：`image_id`→`图片标识`、
`model_name`→`模型`、`generation`→`认领代数`、`messages`→`消息数`、
`reason`→`原因`、`来源=reasoning_content`→`来源=推理内容`。原先与中文说明并列
输出的冗余原始码（如 `终止原因=max_iterations` 与 `终止说明=达到最大迭代次数` 并存）
只保留中文一侧。

## 六、相关配置

| 配置项 | 默认值 | 影响 |
| --- | --- | --- |
| `[jev].enabled` | — | 为假时不发任何请求 |
| `[jev].api_key` | — | 为空时不发任何请求 |
| `[jev].base_url` | `https://api.typesafe.ai` | 可填到域名或含 `/v1/systemone` |
| `[jev].timeout_seconds` | 8.0 | 单次尝试上限 |
| `[jev].total_budget_seconds` | 4.0 | **总**硬上限，含全部重试与退避 |
| `[jev].max_retries` | 2 | 429/529/超时/传输错误的重试次数 |
| `[jev].min_confidence` | 0.0 | 置 0 表示全接受 |
| `[jev].fallback_decider` | `sub_actor` | `sub_actor` 回退或 `silent` 直接静默 |

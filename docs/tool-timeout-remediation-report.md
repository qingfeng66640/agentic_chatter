# agentic_chatter 工具超时问题与修复思路报告

## MMP 复审后的修复架构与交付（2026-11-03）

本节取代下方首次实施的结论。MMP 会话 `x-workspace-py-neo-mofox-neo-mofox-plugi-local-adb7480b5d3f` 对 12 个未提交文件复审，确认 **3 项重大生命周期缺陷与 2 项测试缺口**，本地门禁为 blocked；未向 GitHub 发布评论。首次实施的“未发现新增阻断”和原有绿色测试不足以覆盖这些反例，不再作为当前通过结论。

### 范围与验收清单

- 仅修改 `plugins/agentic_chatter`，不修改核心、其他插件或生产仓；不提交、推送、升版本或发布。
- 修复清理启动失败中断取消、跨 scope 创建失败误判归属、异步关闭与 resume 竞态三项问题。
- 补强“禁止第二步”和“清理实际收到取消”的断言，并用进程内变异检查测试能否检出退化。
- 保留 Manager 事件/历史/缓存路径、planning/batch 提交边界、预算规则、迟到 TOOL_RESULT 隔离、共享 Future 保护。
- 运行 Python 3.11 插件回归、Python 3.12 普通/标准 eager 定向测试、Ruff、框架加载/卸载，并完成独立复核；其他版本仅如实描述已有参数契约测试。

### 审核后的最小架构

1. **归属记录**：`ExecutionCleanup` 保存注册时的 `execution.task` 快照。尚未 bind 时以快照为所有者，成功 bind 后以最新绑定 Task 为所有者；使用身份比较，真实换 Task 才放弃清理。新 scope 创建失败时，核心留下的旧已完成 Task 不再被误认为新所有者。
2. **一次性认领与同步终止**：确认归属且推进已停止后，通过模块级弱引用集合对 execution 同步认领一次清理尝试，防止旧 scope 与创建失败的新 scope 仍共用旧 Task 时重复关闭。此契约限定同一 execution 在同一事件循环使用；认领不跨 await，也不调用业务代码。随后在任何业务 close/aclose 之前同步设置终止状态，关闭核心 `resume()` 的 READY 门禁。`_close_iterator()` 只负责释放，不再在 finally 回写 execution。认领失败时其他记录只释放自身引用，不改执行状态。清理创建失败、运行失败或取消均不撤销认领、不重试；弱引用不延长 execution 生命周期。`_DONE` 表示不可继续推进，不证明资源释放成功。
3. **异常安全取消**：调度清理失败的普通异常在 `ExecutionCleanup` 内记录一次，覆盖直接 abort 和 Task done 回调；`_schedule_cleanup` 只关闭未接收协程并重抛，不重复记录，也不绕开 TaskManager 重试。`cancel_pending` 逐资源隔离其他普通异常，内层 finally 向所有受管推进任务发取消，外层 finally 执行清理截止处理和 scope 关闭。`BaseException` 保持传播，但不能跳过任务取消。
4. **预算与边界不变**：已开始的异步清理仍使用统一宽限、幂等取消，零宽限仍给予首次执行机会。不关闭仍在执行的业务对象，不主动取消共享 Future；成功转移到另一 scope 的执行继续由新所有者管理。
5. **测试证据**：真实框架对象覆盖 cleanup factory 拒绝、跨 scope 恢复失败、原生生成器/手写迭代器慢关闭、零/正宽限、普通/eager，以及异常传播。先断言 `_DONE` 再等待 `wait_done`，避免旧错误状态空转使事件循环内超时失效。

架构审核：`mmp-review:review-architecture` 首次静态审核为**有条件通过**；同步终止、单次日志、两层 finally 和防空转验收为必须落实条件。用户确认后实施。实现后的独立正确性复核发现双 scope 重复关闭，Python 3.11 普通、3.12 普通/标准 eager 的有界探针均记录两次并发 aclose；随后增加上述弱引用一次性认领，补充架构审核为**通过**。该新增发现也纳入修复和回归，不作为可接受遗留。最终独立正确性复核已对修正后的实现重跑三种模式的有界探针，四条生命周期问题的目标路径均收敛，未发现本次范围内新增阻断；具体测试证据与限制见下方。原始 MMP 报告保持不变，记录的是修复前门禁，本次未重新发布或修改其 GitHub 状态。

### 修复映射

| 编号 | MMP 发现 | 目标修改 |
|---|---|---|
| R1 | 清理创建失败跳过任务取消 | 局部调度异常隔离、全量取消 finally、创建失败日志 |
| R2 | 跨 scope resume 创建失败误判归属 | 注册任务快照与 bind 后身份判定 |
| R3 | aclose 期间允许 resume、迟到清理覆盖状态 | 同步终止门禁、清理不再写执行状态 |
| R4 | 修复中发现两个 scope 对同一 execution 重复关闭 | execution 级弱引用一次性认领、拒绝失败后重复尝试 |
| T2 | gather 吞禁止第二步 AssertionError | 独立副作用断言及 Task.cancelled 断言 |
| T3 | 宽限测试未证明取消送达 | CancelledError 事件、释放门闩前有界等待 |

### 最终检查结果

| 检查 | 当前结果 | 范围与限制 |
|---|---|---|
| Python 3.11.14 全插件 pytest | **863 passed、17 skipped、32 warnings**，17.41 秒 | 16 个标准 eager 能力条件跳过、1 个既有跳过；保留既有弃用及未 await 警告，不冒称零告警 |
| Python 3.12.6 超时专项 pytest | **77 passed、12 warnings**，3.11 秒 | `test_tool_timeout.py`、`test_tool_timeout_contracts.py`、`test_tool_timeout_cleanup.py`；含真实标准 eager；仅进程内补充现有依赖路径 |
| Ruff 全插件 | **All checks passed** | 无自动改写范围外代码 |
| 框架本体加载/卸载 | **22 PASS、1 WARN、0 FAIL** | 可选 `croniter>=6.0.0` 未安装；未运行自定义 lifecycle，也未调用生产服务 |
| Git diff 空白检查 | 通过 | Git 提示既有 `test_dispatch_action.py` 下一次处理可能转 CRLF；本轮未修改该文件 |
| 独立正确性复核 | 本次范围无新增阻断 | 3.11 普通、3.12 普通/标准 eager；真实模块/核心推进/TaskManager，有界内存探针；日志、包初始化及无关依赖隔离，不等同业务集成认证 |
| 进程内变异检查 | **7 个变体均检出退化** | T2 延迟取消；T3 禁用清理取消（零/正宽限）；R2 移除任务快照；R3 禁用同步终止；R4 禁用共享认领（手写/原生代理）；均恢复进程内替换，未修改磁盘源码 |

新建的 `tests/test_tool_timeout_cleanup.py` 共 **18 项参数化用例**，覆盖 READY 与 running 同时存在时 cleanup factory 拒绝、done 回调异常、跨 scope 恢复失败、共享认领防重复关闭、慢清理期间禁止 resume、普通异常与 BaseException/CancelledError 的取消收尾、创建拒绝后不重试、弱引用回收。既有两个测试补强副作用、Task.cancelled 和取消事件断言。

测试子任务初稿曾存在错误的 resume 异常预期和无界收尾，未作为通过证据；主会话接管后修正夹具、重写新测试文件并执行上述最终矩阵。临时运行脚本已可恢复移除，未发现遗留的对应测试进程。源码级问题的独立复核与最终正式 pytest 是两类证据，不混同。

主要复跑命令（在 Neo-MoFox 根目录，PowerShell）：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.venv/Scripts/python.exe' -B -X utf8 -m pytest plugins/agentic_chatter/tests -q --no-cov -p no:cacheprovider -p no:randomly --tb=short -o addopts=''
uv tool run ruff check plugins/agentic_chatter/
git -C plugins/agentic_chatter diff --check
```

### 交付及剩余边界

修复实施阶段修改六个文件：`tooling/execution_cleanup.py`、`tooling/timeout.py`、`tests/test_tool_timeout_contracts.py`、新增 `tests/test_tool_timeout_cleanup.py`、`README.md` 与本报告。该阶段基于插件 HEAD `36229c413234489e16df1e51994b807d3519fc75`、版本 `0.8.0` 进行，当时未提交、推送、打包或同步生产；这是修复阶段的历史快照，后续版本以 `manifest.json` 和 `plugin.py` 为准。

后续风格整理对齐核心的中文文档字符串、`Args / Returns / Raises` 说明与多行排版，补齐构造函数文档，并保持六个超时实现/测试文件的 LF 行尾。去除文档字符串后的 AST 与整理前一致；整理后重新运行 Python 3.11 全插件测试（863 passed、17 skipped、32 warnings）、Python 3.12 超时专项（77 passed、12 warnings）及 Ruff/格式/空白检查。

- Manager、planning/batch、预算及迟到结果隔离代码未在本轮改动，现有行为回归继续运行。
- 清理认领只代表一次尝试；factory 拒绝创建、清理失败或拒绝取消时，不保证资源释放成功，不自动重试。
- 同一 execution 限定在同一事件循环使用；跨线程/跨事件循环迁移不在此适配契约内。
- Python 3.13.3/3.13.4/3.14 未实机运行，已有参数契约模拟不能替代完整版本认证。
- 第三方 factory 自行包装输入协程，或先启动 Task 再抛出构造错误的异常行为不在保证范围内。
- 协作取消不能强杀线程、回滚外部副作用或停止任意派生后台任务；未访问生产数据库/API，未证实生产曾触发这些条件缺陷。

## 首次实施记录（历史，结论已被上方 MMP 复审修正）

下方 2026-10-30 正文保留为问题基线及原方案；旧行号和状态不代表当前实现。以下首次实施的测试数字仅作历史记录。

| 项目 | 当前状态 | 证据 / 限制 |
|---|---|---|
| P1 嵌套 eager 漏追踪 | 已实现修复 | 两层/三层、coroutine/iterator、普通/标准 eager 组合回归通过；移除构造中归属的进程内 mutation 被检出 |
| P2 factory 参数不兼容 | 已实现参数契约修复，目标版本实机待验 | 接受 **kwargs；现代 factory 全部转发，legacy 仅适配 name/context；name=None/具名及自定义参数测试通过；未在 3.13.3/3.14 实机执行 |
| T1 异常消费覆盖 | 已补有效断言 | 先断言失败 Task 异常未消费，再 track 后检查已消费且不入 pending；禁用消费的 mutation 被检出 |
| 业务对象清理 | 已实现并新增契约测试 | 首次执行前取消关闭业务 coroutine；yield 后 resume 前取消及 resume factory 抛错会 aclose 生成器；不主动清理尚未被 await 的共享 Future |

### 实际实现与原 wrapper 草案的差异

- 不将推进协程包装后交给旧 factory。`timeout.py` 在构造调用之前记录原协程到本 scope 的 `_constructing`；标准 eager 执行中的父 Task 可通过 `get_coro()` 与该集合建立归属，解决创建返回前的递归窗口，同时保留旧 factory 看到的原协程身份/code。
- 不放宽为“继承 ContextVar 的全部任务都归属”，普通后台任务仍排除。旧 factory 如果自行包装输入协程，导致返回 Task 的 get_coro 不再是该输入，不能宣称此类 factory 也已覆盖。
- `execution_cleanup.py` 明确解析已知核心推进协程 frame 中的执行实例与业务对象，只在推进任务空闲后清理；跨 scope resume 后旧记录放弃清理权。
- 清理任务通过 TaskManager 创建，并清除执行归属。已开始/迟到的清理使用同一宽限截止时间；截止与退出取消请求幂等，避免重复 cancel 打断异常收尾。0 宽限不等待，但不在首次运行前就取消 aclose。
- factory 接收 kwargs，并且不通过捕获任意 TypeError 重试。旧 factory 的未知参数明确拒绝，不静默丢弃。
- 日志已改为“已追踪任务取消未收敛”，不将集合计数包装成所有工作停止的证明。

### 本轮检查

- Python 3.12.6：原超时测试与新契约测试合计 **59 passed**；包含标准 eager 的真实运行，使用进程级补充现有依赖路径，不代表独立生产环境认证。
- Python 3.11.14：全插件 **849 passed、13 skipped、32 warnings**；12 项跳过为标准 eager 能力缺失，另有既有跳过项。
- Ruff 通过；框架本体真实加载/卸载 **22 PASS、1 WARN、0 FAIL**（可选 croniter 未安装；未运行自定义 lifecycle）。
- 新增 `tests/test_tool_timeout_contracts.py` 共23项参数化用例，覆盖构造归属、原factory code/参数保真、业务对象清理、零宽限、拒绝一次取消、跨scope接手和T1。
- 独立复核已返回：在声明支持范围内未发现新增阻断。复核者进行了源码核对和 Python 3.11 的9项隔离探针（TaskManager/日志使用替身），未独立运行 eager 或3.13/3.14；标准 eager 的真实模块测试及全插件回归由主会话执行，不能混同两类验证范围。
- 本轮仅修改插件文件；未提交、推送、升版本、打包或同步生产。

### 尚未覆盖与不作承诺

- 未安装或运行 Python 3.13.3/3.13.4/3.14，不把参数模拟测试当作实机验收；跨版本发布仍须相应解释器验证。
- 自行包装输入 coroutine 的第三方 task factory、异常地在已启动Task后抛出构造错误的factory不属于现有契约保证。
- 取消仍为协作式，无法强杀线程、回滚副作用、停止任意派生后台任务，也不能保证拒绝取消的清理最终完成。
- 对核心 code/frame/private状态的依赖仍存在，升级需重跑契约测试。

---

## 历史问题基线与修复方案（2026-10-30）

- 报告日期：2026-10-30
- 对象：开发仓 `plugins/agentic_chatter` 的工具调用超时管理实现
- 审查基线：插件独立仓 HEAD `36229c413234489e16df1e51994b807d3519fc75` 加当前未提交实现
- 性质：现状评估与待实施方案；不是修复完成证明，也不是整个插件的全面审计
- 修改边界：后续实施只允许修改插件目录，禁止修改核心 `src/`、其他插件和生产仓

## 一、执行摘要

当前工具超时实现已能在主聊天和后台任务执行器中限制一次工具提交的等待时间，保留原有 Manager 路径，并通过临时响应缓冲隔离迟到结果。之前发现的“dispatcher 第一层 eager 漏登记”和“factory 共存测试未证明原 factory 被调用”已修复并通过定向测试。

但仍有两项重大未关闭问题：

1. **嵌套 eager 执行漏取消**：工具或 Agent 执行体直接等待另一个框架 usable 时，内层执行任务可能在外层登记前创建而漏追踪；超时后内层可继续运行，且日志可能错误显示 `取消未收敛=0`。已由独立审查、反证复核和主会话直接调用当前模块复现。
2. **Python 3.13.3 / 3.14 任务工厂参数不兼容**：当前 factory 仅接收 `context`，无法接收这些版本传来的 `name` 等参数，具名任务创建可能直接抛出 TypeError。源码和标准库契约已核实；本机未进行对应解释器的完整运行验证，不能泛化为所有 Python 3.13 版本。

另有一项较小的测试证据缺口：尚未对“已完成且未被 await 的失败 Task，其异常被 `track()` 消费”作有效断言。它不是已经证实的运行错误。

**发布判断：两条原始发现可以关闭，但不能据此宣称整个超时管理通过审查；建议先关闭两项重大问题，再确认支持矩阵和上线条件。未证实现有生产插件已触发嵌套问题，也没有证据表明生产已经受害。**

## 二、真实机制及必须保持的行为

调用路径概括如下：

`AgenticChatter / TaskExecutor → run_with_timeout → BaseChatter.run_tool_call → dispatch_llm_tool_calls → Tool / Action / Agent Manager → execute_usable_instance → LLMUsableExecution`

框架与插件的关键事实：

- planning：插件逐个提交工具调用；batch：整批提交。
- 当前核心 dispatcher 对批内调用逐个等待 Manager；batch 不等于已经并行执行。底层统一执行器支持 READY/yield 调度，并不意味着上层 dispatcher 同时提交了多个调用。
- `LLMUsableExecution` 为协程或异步迭代推进另建 Task；每次 resume 可能换一个 Task。
- 调度器通过 `asyncio.wait` 等待这些 Task；仅取消外层 dispatcher 不会自动取消所有独立推进 Task。
- 基类和底层部分接口虽有 observer，但当前 Manager 分发链没有完整贯通；不能假定插件只传 observer 就能覆盖所有执行。
- 插件当前用 ContextVar、临时链式 task factory 和任务创建者归属条件识别本次执行任务，避免用全局任务差集误杀其他聊天流。
- 默认每调用预算 30 秒；batch 的提交预算为各调用预算之和，不是逐项硬截止时间。全局设为 0 关闭保护。
- `dispatch_task` 未覆盖时预算动态兼容 `tasks.timeout_seconds + 30`，且不低于普通预算。
- 超时先封口缓冲响应，保留已返回正文，只补缺项。未取得整批成功状态时保守记失败，不根据结果文本猜成功。

以上提交边界、Manager 路径、缓冲回灌和失败策略，不应在本轮修复中顺便改变。

## 三、问题清单

| 编号 | 状态 | 级别 | 触发条件与影响 | 主要位置 |
|---|---|---|---|---|
| P1 | 已确认、未修复 | 重大 | eager factory 下执行体直接嵌套框架 usable；内层漏取消，仍可产生副作用，取消统计不完整 | `tooling/timeout.py:73-94` |
| P2 | 已确认契约不兼容、未修复 | 重大 | Python 3.13.3 / 3.14 将 name 等参数传给 factory；当前签名报 TypeError | `tooling/timeout.py:68-72`、`35-49` |
| T1 | 已确认覆盖缺口 | 较小 | 已完成失败任务的异常消费没有非冗余断言；删除消费逻辑仍可能通过现有测试 | `tooling/timeout.py:163-164`、`tests/test_tool_timeout.py:623` |
| F1 | 已修复 | 原重大 | dispatcher eager 启动时尚未登记，第一层执行任务漏追踪 | `tooling/timed_calls.py:151-164` |
| F2 | 已修复 | 原较小 | 原 factory 被绕过时共存测试仍可能通过 | `tests/test_tool_timeout.py:336-340` |

### P1：嵌套 eager 漏追踪

发生顺序：

1. dispatcher 已通过 `tracked_dispatch()` 在入口登记。
2. dispatcher 请求创建外层 `LLMUsableExecution` 推进 Task。
3. tracking factory 调用原 eager factory；外层任务立即运行，但此时尚未执行创建返回后的 `scope.track(outer_task)`。
4. 外层执行体直接等待内层框架 usable；内层推进任务创建时，当前创建者 outer 尚未在 scope 集合中。
5. 内层因父任务归属条件不满足而漏追踪；随后外层登记并不能补回已经创建的内层。
6. 超时取消 dispatcher 和 outer；内层调度通过 asyncio.wait 等待，取消不会自动沿该等待关系传播给 inner。

主会话对当前真实模块的 Python 3.12.6 对照结果（30ms 超时）：

| 调度方式 | outer | inner | 释放 Event 后的模拟副作用 |
|---|---|---|---|
| 普通 factory | 已取消 | 已取消 | 无 |
| 标准 eager factory | 已取消 | 未结束、未取消 | 发生 |

两组均记录 `取消未收敛=0`；探针 finally 均清理全部任务并恢复原 factory。这里只用假的业务执行体及测试响应对象；真实 `run_with_timeout`、`execute_usable_instance`、`LLMUsableExecution` 和标准 factory 未被替换。

框架公开的 `BaseAgent.execute_local_usable()` 在 `src/core/components/base/agent.py:295` 委托统一执行器，说明嵌套属于存在的框架能力。但限定搜索未找到具体已部署插件实际触发实例，不能将条件复现写成生产事故。

P1 是原有追踪设计的残留，不是入口自登记修复新引入的回归；“第一层已修复”和“递归嵌套仍有窗口”同时成立。

### P2：task factory 参数契约

当前 `tracking_factory(loop, coroutine, context=None)` 不接受 `name` 和其他任务构造参数，`_make_task` 也仅处理 context。TaskManager 会创建具名 dispatcher。

Python 3.13.3 和 3.14 的参数转发行为与该签名不兼容；3.13.4 曾回退 name/context 的部分转发行为，因此不能把整个 3.13 系列一概标为受影响。

影响范围不限于当前工具：包装器安装在事件循环上，保护期间其他具名任务也可能被同一签名错误影响。对应版本尚未完成端到端运行测试，应在修复后补真实版本测试，而不是仅在 3.11 上手工调用 factory 便宣布跨版本兼容。

参考标准库版本记录：
https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.set_task_factory

### T1：已完成失败任务的异常消费测试

现有幂等测试等待的是成功完成任务，无法证明 `track(done_failed_task)` 确实读取了无人等待的失败异常。此处实现已有消费逻辑，当前只是测试不足，不应升级为已证实的异常泄漏。

建议新增未 await 的失败 Task，在确认任务 done 后交给 track，通过异常检索标记或受控 event-loop 异常处理器验证消费效果；不得通过先 await/gather 或先调用 exception() 使测试提前消费异常。

## 四、推荐修复设计

### 4.1 P1：在每个受保护执行任务入口自登记，而不只登记 dispatcher

**设计不变量：受保护的框架执行任务在执行任何工具业务代码之前，就应当被归入本次 scope。**

推荐在插件已有 task factory 适配层中完成，而不是再增加全局任务扫描：

1. 在调用原 factory **之前**，读取目标 Context、原协程 code 和当前创建者，决定该次创建是否属于当前 scope 的已知框架推进任务。
2. 对符合条件的 `_await_result` / `_advance_iterator` 协程，交给一个插件局部包装协程运行。
3. 包装协程入口取得自己的 current_task，先幂等登记，再 await 原始推进协程。原 factory 仍负责实际创建 Task，保留其 eager/自定义 Task 行为。
4. 返回后再做幂等登记作为兜底，但不能再依赖此时机建立唯一归属。
5. 外层推进任务在进入工具业务代码前已经登记；其直接创建的内层执行任务也走同一包装，自然覆盖更深层嵌套及异步生成器 resume。
6. 不符合框架 code 或创建者不归属的普通 TaskManager 后台任务不包装；仅继承 ContextVar 不足以成为归属依据。保留现有跨流和后台隔离。
7. 取消期间新建的受保护执行任务不能因“对自己发出 cancel”就继续立即进入原业务体：入口应明确阻止原执行体启动，不能只依赖取消请求在下一个 await 才生效。

该方案不需要改核心类、Manager 或 dispatcher，不拆 batch，也不把默认 factory 强行换成普通调度。核心 observer 将来若完整贯通，可再考虑替换适配；本轮不假定该能力已可用。

**实现风险必须专门验证：**包装前已创建的原协程，在包装 Task 尚未首次运行就取消、factory 构造抛错、scope 已关闭等路径上可能无人 await。应建立明确的协程所有权和 close 清理，防止 `coroutine was never awaited`；包装层不得吞 CancelledError、重复等待原协程或覆盖原异常。不能只加一个 wrapper 后便视为完成。

不推荐：删除父任务过滤、仅按 ContextVar 抓全部后代、all_tasks 差集取消、临时禁用 eager、复制/替换 Manager 调度逻辑。

### 4.2 P2：完整保留任务创建参数，明确旧 factory 边界

- factory 入口接受适用版本传入的 `**kwargs`，原样保留 name/context/eager_start 等任务构造参数（以实际 Python 版本契约为准）。
- 没有旧 factory 时，交给该版本支持的 Task 构造器；有旧 factory 时，向其按契约转发。
- 显式空 Context 必须与未提供 Context 区分，不能使用 truthiness 判断。
- 不通过捕获任意 TypeError 后无参数重试来兼容：旧 factory 自身也可能抛 TypeError，重试可能重复构造或掩盖真实错误。
- 如果承诺兼容旧签名 factory，应先检查签名并定义可模拟语义：name 可在返回任务上设置，旧式 context 路径维持 context.run；无法保真的参数应明确拒绝或声明不支持，而不是静默丢弃。
- 不为规避错误悄悄降低项目支持范围；若最终无法支持某版本，应明确写入支持矩阵并经用户确认。

### 4.3 T1 与诊断语义

- 补齐已完成失败任务异常消费断言，并保留重复 track 不叠加回调、done 不进 pending 集合的测试。
- `取消未收敛` 只统计已登记集合，不具备“所有相关任务已经停止”的证明能力。日志可改成“已追踪任务取消未收敛数”等准确表述。
- 日志至少可关联提交/流及预算；若增加作用域标识、追踪任务数量，不输出工具敏感参数或完整结果正文。
- 修复 P1 后仍不能承诺阻塞线程已终止、工具自建任意后台任务已停止或外部副作用已回滚。

## 五、分阶段实施与验收

### 阶段 A：先固定反例，再修复嵌套归属

建议文件：`tests/test_tool_timeout.py`、`tooling/timeout.py`；现有 dispatcher 自登记保留，避免无关重写。

必须先新增在当前实现上失败的双层/三层嵌套 eager 回归，并确保测试失败也能 finally 清理全部任务。随后实施入口包装及协程所有权清理。

验收：

- 普通和 eager 调度下双层、三层框架执行超时后所有协作式执行任务结束，不产生释放门闩后的副作用。
- 外层/内层分别覆盖 coroutine、async generator，多次 yield/resume 后仍追踪正确。
- 在进入受保护的业务体前已有任务归属，取消阶段新建执行不再启动业务代码。
- 独立 TaskManager 后台任务和其他流不被误杀；显式独立 Context 不被强行继承 scope。
- timeout、外部取消、重复取消、立即成功、立即异常、拒绝取消的有限宽限均保持原行为。
- wrapper 首次运行前取消、factory 抛错等路径没有未 await 协程和未消费 Task 异常。
- 进程内撤销入口包装后，新增嵌套测试必须失败；不能仅断言外层返回了 timeout。

### 阶段 B：修任务工厂跨版本契约

建议文件：`tooling/timeout.py` 与同一测试文件。

验收矩阵至少包括 Python 3.11、3.12（含真实 eager）、3.13.3、3.13.4、3.14 的相关参数行为；无法取得的版本明确标为未验证，不用模拟调用冒充真实版本测试。

覆盖有/无既有 factory、现代/legacy factory、name、显式 Context、该版本支持的其他参数、原 factory 调用与返回 Task 身份、异常不重试、退出恢复和第三方替换不被覆盖。

P1/P2 均涉及同一工厂适配层，应在各自回归通过后合并运行，避免分别正确、组合后失效。

### 阶段 C：闭环与交付

- 运行完整插件测试、ruff、diff 空白检查、真实框架加载/卸载检查。
- 由独立审查者检查工厂组合契约、协程所有权和新回归的检错能力。
- README 只同步必要的支持范围和取消限制，不创建重复开发过程文档。
- 不自动提交、推送、升版本、打包、发布或同步生产；这些动作不包含在本报告授权内。

## 六、已修复能力与当前检查基线

已修复并保留：

- dispatcher 在协程入口自登记，覆盖第一层 eager 漏追踪。
- track 幂等，已完成任务不进入待取消集合。
- factory 共存测试按 probe 协程身份断言实际调用一次、返回同一 Task；现代 factory 核对 Context 身份，现代/legacy 均核对运行 Context。
- TOOL_RESULT 按调用匹配，超时补缺不重复，封口后丢弃迟到回写。
- 主聊天、后台任务两处接线，保留 Manager 与 planning/batch 提交边界。

截至本报告前最近一次检查（2026-10-30，报告编写本身未再运行整套）：

| 检查 | 实际结果 | 说明 |
|---|---|---|
| Python 3.12.6 超时测试 | 36 passed，12 warnings | 8 项真实 eager 场景；使用进程级补充现有依赖路径，并非独立部署环境认证 |
| Python 3.11.14 全插件测试 | 830 passed，9 skipped，32 warnings | 8 项跳过为不具备标准 eager 能力；另有既有跳过项 |
| 相关文件 ruff / diff 空白 | 通过 | 不等于运行正确性保证 |
| 框架本体加载/卸载 | 2026-10-26 曾为22 PASS、1 WARN、0 FAIL | 警告为可选croniter未安装；未执行真实自定义lifecycle，非生产验证 |
| 第一层入口登记/原factory调用的反向检查 | 两项均检出回归 | 进程内移除入口登记/绕过原factory，无源码落盘修改 |
| 嵌套 eager 反例 | 仍能复现 | 说明现有36项通过不能证明递归嵌套正确 |

## 七、不能作为缺陷或修复承诺的事项

- 超时批次未知状态保守计失败：已公开策略，会消耗失败预算并可能暂停任务；不能从正文推断成功。如要改变应另行设计 unknown 状态及预算政策，不作为本轮顺手修复。
- 线程、同步阻塞和外部副作用：协作式取消的边界，外层超时不能修复 native/thread 死锁或保证撤销外部操作。
- 私有核心执行入口依赖：属于版本耦合，需要契约测试；使用私有入口本身不等于已经发现一个新的运行缺陷。
- 向量库、emoji_sender、调度器等此前话题不在本报告范围，不归因为 agentic_chatter 超时实现问题。
- 未证实当前宿主启用 eager factory、生产插件发生嵌套触发或生产事故，不能据条件探针推断线上已受害。

## 八、结论与行动顺序

建议按 **先补嵌套反例 → 修每层执行的入口归属及清理 → 补factory参数契约 → 完整回归与独立复核** 推进。

两项重大问题关闭前，不宜把“830项通过”或“超时函数能返回”作为全面正确或可上线的证明。当前可确认的是：第一层修复有效、共存测试证据有效，递归嵌套和跨版本契约仍须补齐。报告提供的是修复方向和验收门槛，不代表方案已实施或验证通过。

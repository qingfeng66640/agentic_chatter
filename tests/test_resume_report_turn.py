"""resume 事件透传与任务汇报轮的测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from ..task_runtime import coordinator
from ..task_runtime.models import TaskStatus
from ..task_runtime.templates import build_task_report_prompt
from .test_task_completion_report import _make_runtime  # 复用 runtime 构造

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.core.components.types import ChatterResult

    from ..pipeline.state import TurnState
    from ..task_runtime.executor import TaskRequest

# ---------------------------------------------------------------------------
# build_task_report_prompt
# ---------------------------------------------------------------------------


def test_report_prompt_contains_status_and_summary() -> None:
    """汇报提示应包含任务类型、状态、结果摘要和等待用户时的补充提示。"""
    events = (
        {
            "task_id": "t1",
            "task_type": "research",
            "status": "succeeded",
            "summary": "已查到三篇资料",
            "error": "",
        },
        {
            "task_id": "t2",
            "task_type": "general",
            "status": "waiting_user",
            "summary": "",
            "error": "",
        },
    )
    prompt = build_task_report_prompt(events)
    assert "research" in prompt
    assert "succeeded" in prompt
    assert "已查到三篇资料" in prompt
    assert "waiting_user" in prompt
    assert "补充任务" in prompt


def test_report_prompt_empty_events_renders_header_only() -> None:
    """没有完成事件时，汇报提示仍应保留子代理汇报标识。"""
    prompt = build_task_report_prompt(())
    assert "sub-agent" in prompt


# ---------------------------------------------------------------------------
# _is_task_resume_event
# ---------------------------------------------------------------------------


class _Event:
    """仅携带来源字段，用于区分消息、定时器和子代理唤醒。"""

    def __init__(self, source: str) -> None:
        """记录事件来源，供汇报轮识别和事件透传断言使用。"""
        self.source = source


def test_is_task_resume_event_matches_sub_agent() -> None:
    """仅子代理来源应被识别为任务唤醒，普通事件和空值均应排除。"""
    from ..chatter import _is_task_resume_event

    assert _is_task_resume_event(_Event("sub_agent")) is True
    assert _is_task_resume_event(_Event("message")) is False
    assert _is_task_resume_event(_Event("timer")) is False
    assert _is_task_resume_event(None) is False


# ---------------------------------------------------------------------------
# execute() 透传 asend 值
# ---------------------------------------------------------------------------


class _ProbeChatter:
    """最小 chatter 替身：记录 _run_turns 收到的 asend 值。"""

    def __init__(self) -> None:
        """准备固定流标识、事件接收列表和默认关闭的插件状态。"""
        self.stream_id = "probe-stream"
        self.received: list[_Event | None] = []
        self.enabled = False

    @property
    def cfg(self) -> SimpleNamespace:
        """根据当前启用标志提供最小插件配置。"""
        return SimpleNamespace(plugin=SimpleNamespace(enabled=self.enabled))

    async def _run_turns(
        self, config: SimpleNamespace
    ) -> AsyncGenerator[str, _Event | None]:
        # 首次 yield 后接收 asend 值，共收 3 个后结束
        """依次挂起三个结果，并记录每次恢复时传入的事件。

        Args:
            config: 外层执行入口传入的最小插件配置，本替身不读取。

        Yields:
            str: 当前挂起点的结果标识；恢复时可接收事件或空值。
        """
        received = yield "result-0"
        self.received.append(received)
        received = yield "result-1"
        self.received.append(received)
        received = yield "result-2"
        self.received.append(received)


@pytest.mark.asyncio
async def test_execute_forwards_asend_values() -> None:
    """外层执行生成器应保持结果顺序，并将恢复事件逐次透传给内部生成器。"""
    from ..chatter import _AgenticChatterBase

    probe = _ProbeChatter()
    probe.enabled = True
    event_a = _Event("message")
    event_b = _Event("sub_agent")

    gen = _AgenticChatterBase.execute(probe)
    first = await gen.asend(None)
    assert first == "result-0"
    second = await gen.asend(event_a)
    assert second == "result-1"
    third = await gen.asend(event_b)
    assert third == "result-2"
    await gen.aclose()

    # 框架 asend 进来的事件必须逐次透传到 _run_turns
    # （第三次 asend 的接收发生在 yield "result-2" 挂起期间，
    # aclose 之前替身已记录 event_b；第三个值随 GeneratorExit 丢失）
    assert probe.received == [event_a, event_b]


# ---------------------------------------------------------------------------
# 汇报轮消费（源码级验证 _run_turns 的分支与 _run_report_turn 行为）
# ---------------------------------------------------------------------------


def test_run_turns_has_report_branch() -> None:
    """主回合源码应保留任务唤醒识别、完成事件确认和汇报轮分支。"""
    import inspect

    from ..chatter import _AgenticChatterBase

    source = inspect.getsource(_AgenticChatterBase._run_turns)
    assert "_is_task_resume_event(resume_event)" in source
    assert "pending_completed_events" in source
    assert "acknowledge_completed_events" in source
    assert "_run_report_turn" in source
    assert "resume_event = yield turn_result" in source


def test_run_report_turn_no_claim_no_commit() -> None:
    """无消息汇报轮不得领取或提交消息，并应通过事件构建汇报提示。"""
    import inspect

    from ..chatter import _AgenticChatterBase

    source = inspect.getsource(_AgenticChatterBase._run_report_turn)
    # 不取 claim、不 commit：不应出现 claim_pending/commit_claim 调用
    assert "claim_pending" not in source
    assert "commit_claim" not in source
    assert "task_report=build_task_report_prompt(events)" in source


def test_input_confirmed_moved_after_commit() -> None:
    """P4：任务路由分支不再提前置位 input_confirmed。"""
    import inspect

    from ..chatter import _AgenticChatterBase

    run_turns = inspect.getsource(_AgenticChatterBase._run_turns)
    # 路由成功后到 commit 之前不得有 input_confirmed 置位
    route_index = run_turns.index("_route_task_message")
    commit_index = run_turns.index("_commit_task_turn")
    segment = run_turns[route_index:commit_index]
    assert "state.input_confirmed = True" not in segment


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "failed_state", "no_speech"])
async def test_report_turn_failure_keeps_events_for_retry(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """无 claim 的汇报异常、模型失败或无发言后仍可重试。"""
    from .. import chatter as chatter_module
    from ..chatter import _AgenticChatterBase

    stream_id = f"report-retry-{failure}"
    event = {"task_id": "retry-task", "status": "succeeded", "summary": "任务结果"}
    coordinator.push_completed_event(stream_id, event)

    class _Mailbox:
        """始终提供所有权但不提供消息，用于进入无领取记录的汇报轮。"""

        async def try_acquire(self, owner: object) -> int:
            """固定授予第一代所有权，避免其他所有者影响汇报重试。"""
            return 1

        async def merge_snapshot(
            self,
            messages: list[SimpleNamespace],
            *,
            owner: object,
            generation: int,
        ) -> int:
            """保持空邮箱，不将快照转为待处理消息。

            Args:
                messages: 待合并的消息替身，本场景为空。
                owner: 调用方的所有者令牌，本替身不校验。
                generation: 所有权代次，本替身不校验。

            Returns:
                int: 固定为零，表示没有新增待处理消息。
            """
            return 0

        async def pending_state(self, window: float) -> tuple[int, float]:
            """返回零条待处理消息和零等待时长，使流程只处理汇报事件。"""
            return 0, 0.0

        async def claim_pending(self, owner: object, generation: int) -> None:
            """对任意所有者和代次都返回空值，保持无消息领取记录。"""
            return None

        async def release_owner(self, owner: object, generation: int) -> bool:
            """忽略所有者校验并固定报告释放成功。"""
            return True

    class _Chatter:
        """记录汇报尝试，并仅在首轮注入指定失败以观察事件重试。"""

        def __init__(self) -> None:
            """绑定当前测试流，并初始化汇报尝试记录。"""
            self.stream_id = stream_id
            self.attempts: list[tuple[dict[str, Any], ...]] = []

        async def fetch_unreads(self) -> tuple[None, list[SimpleNamespace]]:
            """返回空消息集合，避免普通对话回合干扰汇报重试。"""
            return None, []

        def _build_deduper(self, config: object) -> None:
            """不建立去重器，让替身仅关注汇报事件是否被保留。"""
            return None

        async def _stage_act(
            self,
            config: object,
            chat_stream: object,
            state: TurnState,
            unread_msgs: list[SimpleNamespace],
            *,
            claim: object | None = None,
        ) -> None:
            """首轮按参数注入异常、失败状态或无发言，后续轮次标记已发言。

            Args:
                config: 行动阶段配置，本替身不读取。
                chat_stream: 聊天流占位对象，本替身不读取。
                state: 需要更新失败或发言标志的真实回合状态。
                unread_msgs: 未读消息替身列表，本场景为空。
                claim: 可选领取记录，本场景不使用。

            Raises:
                RuntimeError: 首次汇报且失败参数指定为异常时抛出。
            """
            self.attempts.append((event,))
            if len(self.attempts) == 1:
                if failure == "exception":
                    raise RuntimeError("模拟汇报失败")
                if failure == "failed_state":
                    state.failed = True
            else:
                state.spoke = True

        async def _run_report_turn(
            self, *args: Any, **kwargs: Any
        ) -> tuple[ChatterResult, bool]:
            """将动态调用参数转交真实汇报实现，以验证事件确认行为。

            Args:
                *args: 真实汇报实现接收的位置参数。
                **kwargs: 真实汇报实现接收的关键字参数，包括完成事件。

            Returns:
                tuple[ChatterResult, bool]: 汇报回合结果和是否成功完成汇报的标志。
            """
            return await _AgenticChatterBase._run_report_turn(self, *args, **kwargs)

        async def _release_owner_safely(self, *args: object) -> None:
            """接受清理接口参数但不执行释放，让测试无需真实邮箱所有权管理。"""
            return None

    monkeypatch.setattr(chatter_module, "get_stream_mailbox", lambda _: _Mailbox())
    import src.core.managers as managers

    class _StreamManager:
        """为汇报重试提供无需外部存储的聊天流获取接口。"""

        async def get_or_create_stream(self, *, stream_id: str) -> object:
            """忽略流标识并返回新的占位对象，不访问真实聊天流。"""
            return object()

    monkeypatch.setattr(managers, "get_stream_manager", lambda: _StreamManager())
    chatter = _Chatter()
    generator = _AgenticChatterBase._run_turns(chatter, None)
    try:
        await generator.asend(None)
        assert coordinator.has_completed_events(stream_id)
        await generator.asend(None)
        assert chatter.attempts == [(event,), (event,)]
        assert not coordinator.has_completed_events(stream_id)
        await generator.asend(None)
        assert chatter.attempts == [(event,), (event,)]
    finally:
        await generator.aclose()
        coordinator.drain_completed_events(stream_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [True, False])
async def test_normal_turn_acknowledges_report_only_after_success(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    """有真实消息时，普通回合失败保留事件，成功汇报才确认。"""
    from .. import chatter as chatter_module
    from ..chatter import _AgenticChatterBase

    stream_id = f"normal-report-{failed}"
    event = {"task_id": "normal-task", "status": "succeeded", "summary": "任务结果"}
    coordinator.push_completed_event(stream_id, event)
    message = SimpleNamespace(content="你好")

    class _Mailbox:
        """固定提供一条待处理消息，使完成事件随普通对话回合汇报。"""

        async def try_acquire(self, owner: object) -> int:
            """固定授予第一代所有权，使普通对话回合可继续执行。"""
            return 1

        async def merge_snapshot(
            self,
            messages: list[SimpleNamespace],
            *,
            owner: object,
            generation: int,
        ) -> int:
            """固定报告一条待处理消息，而不维护真实快照。

            Args:
                messages: 待合并的消息替身列表。
                owner: 调用方的所有者令牌，本替身不校验。
                generation: 所有权代次，本替身不校验。

            Returns:
                int: 固定为一，使流程保留普通消息处理分支。
            """
            return 1

        async def pending_state(self, window: float) -> tuple[int, float]:
            """返回一条待处理消息和零等待时长，让流程立即领取消息。"""
            return 1, 0.0

        async def claim_pending(
            self, owner: object, generation: int
        ) -> SimpleNamespace:
            """为任意所有者和代次返回包含当前消息的领取记录替身。"""
            return SimpleNamespace(messages=[message])

        async def release_owner(self, owner: object, generation: int) -> bool:
            """固定报告释放成功，不检查调用方所有权。"""
            return True

    class _Manager:
        """以空活跃任务列表关闭本场景的任务路由干扰。"""

        def get_active_tasks(self, stream_id: str) -> list[object]:
            """对任意流返回空任务列表，使普通回合不转入任务路由。"""
            return []

    class _Chatter:
        """收集普通回合内的汇报正文，并按参数设置失败与发言状态。"""

        def __init__(self) -> None:
            """绑定当前测试流，并准备汇报正文收集列表。"""
            self.stream_id = stream_id
            self.reports: list[str] = []

        async def fetch_unreads(self) -> tuple[None, list[SimpleNamespace]]:
            """返回当前测试消息，使主循环进入普通对话回合。"""
            return None, [message]

        def format_message_line(self, message: SimpleNamespace) -> str:
            """直接取消息正文作为格式化结果，避免引入框架格式化依赖。"""
            return message.content

        def _build_deduper(self, config: object) -> None:
            """省略去重器构造，保留普通回合汇报状态的验证重点。"""
            return None

        async def _run_normal_turn(
            self,
            config: object,
            mailbox: _Mailbox,
            claim: SimpleNamespace,
            chat_stream: object,
            state: TurnState,
            messages: list[SimpleNamespace],
            generation: int,
        ) -> tuple[object, None]:
            """记录注入的汇报正文，并设置本轮成功或失败状态。

            Args:
                config: 回合配置，本替身不读取。
                mailbox: 固定提供一条消息的邮箱替身。
                claim: 包含测试消息的领取记录替身。
                chat_stream: 聊天流占位对象。
                state: 含待汇报正文、并接收失败及发言标志的回合状态。
                messages: 本轮的消息替身列表。
                generation: 所有权代次，本替身不校验。

            Returns:
                tuple[object, None]: 回合结果占位对象和空的剩余领取记录。
            """
            self.reports.append(state.task_report)
            state.failed = failed
            state.spoke = not failed
            return object(), None

        async def _release_owner_safely(self, *args: object) -> None:
            """接受收尾参数但不操作真实邮箱，避免所有权清理影响断言。"""
            return None

    class _StreamManager:
        """提供独立的聊天流占位对象，隔离普通回合测试的外部依赖。"""

        async def get_or_create_stream(self, *, stream_id: str) -> object:
            """忽略流标识并返回占位对象，不建立真实聊天流。"""
            return object()

    monkeypatch.setattr(chatter_module, "get_stream_mailbox", lambda _: _Mailbox())
    monkeypatch.setattr(
        chatter_module, "_restore_persisted_tasks", lambda **kwargs: _Manager()
    )
    import src.core.managers as managers

    monkeypatch.setattr(managers, "get_stream_manager", lambda: _StreamManager())
    chatter = _Chatter()
    generator = _AgenticChatterBase._run_turns(chatter, None)
    try:
        await generator.asend(None)
        assert len(chatter.reports) == 1
        assert "任务结果" in chatter.reports[0]
        assert coordinator.has_completed_events(stream_id) is failed
    finally:
        await generator.aclose()
        coordinator.drain_completed_events(stream_id)


# ---------------------------------------------------------------------------
# task_report 注入 user prompt extra_parts
# ---------------------------------------------------------------------------


def test_turn_state_has_task_report_field() -> None:
    """新建回合状态的任务汇报字段应默认为空字符串。"""
    from ..pipeline.state import TurnState

    state = TurnState(stream_id="s", unread_texts="", deduper=None)
    assert state.task_report == ""


def test_user_prompt_appends_task_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """用户提示构建时应把回合中的任务汇报内容加入附加部分。"""
    from .. import chatter as chatter_module
    from ..chatter import _AgenticChatterBase
    from ..pipeline.state import TurnState

    captured: dict[str, str] = {}

    class _Template:
        """记录模板变量，并仅渲染附加部分以观察汇报内容注入。"""

        def set(self, key: str, value: str) -> "_Template":
            """保存模板变量并返回自身，保留真实模板的链式设置接口。"""
            captured[key] = value
            return self

        async def build(self) -> str:
            """返回已捕获的附加文本，未设置时返回空字符串。"""
            return captured.get("extra", "")

    class _PM:
        """仅为用户提示模板提供替身，其他模板名称返回空值。"""

        def get_template(self, name: str) -> _Template | None:
            """按模板名称返回用户提示模板替身或空值。"""
            return _Template() if name == "agentic_chatter_user" else None

    chatter = object.__new__(_AgenticChatterBase)
    chatter.stream_id = "extra-stream"
    state = TurnState(
        stream_id="extra-stream",
        unread_texts="用户消息",
        deduper=None,
        task_report="汇报内容",
    )

    class SimpleStream:
        """提供用户提示所需的聊天属性和空历史消息。"""

        stream_name = "s"
        platform = "p"
        bot_nickname = "n"
        bot_id = "b"
        chat_type = "private"
        context = type("C", (), {"history_messages": []})()

    monkeypatch.setattr(
        chatter_module,
        "get_prompt_manager",
        lambda: _PM(),
    )
    result = asyncio.run(chatter._build_user_prompt(None, SimpleStream(), state, []))
    assert "汇报内容" in result


# ---------------------------------------------------------------------------
# 防多重回复：push 幂等 + drain 单次消费（组合验证）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_push_and_single_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    # 手动打桩 resume_chatter（fixture 不跨测试文件共享）
    """重复推送同一完成事件后应只消费一次，并仅唤醒聊天器一次。"""
    import src.core.managers.chatter_manager as cm

    calls: list[tuple[str, str]] = []

    class _Manager:
        """记录聊天器唤醒请求，便于核对来源和调用次数。"""

        async def resume_chatter(self, stream_id: str, *, source: str = "") -> None:
            """收集流标识和唤醒来源，不启动真实聊天器。"""
            calls.append((stream_id, source))

    monkeypatch.setattr(cm, "get_chatter_manager", lambda: _Manager())
    runtime = _make_runtime("stream-m", TaskStatus.WAITING_USER)
    request = _make_request()
    await coordinator._notify_completion(
        object(), runtime, request, runtime.state.result
    )
    # 模拟重复 push（幂等）
    from ..task_runtime.coordinator import push_completed_event

    push_completed_event(
        "stream-m",
        {
            "task_id": runtime.state.task_id,
            "task_type": "general",
            "status": "waiting_user",
            "summary": "",
            "error": "",
        },
    )
    events = coordinator.drain_completed_events("stream-m")
    assert len(events) == 1
    assert calls == [("stream-m", "sub_agent")]


def _make_request() -> TaskRequest:
    """创建允许最终文本交付、但关闭过程事件汇报的任务请求。"""
    from ..task_runtime.executor import TaskRequest

    return TaskRequest(objective="目标", deliver_final_text=True, report_events=False)

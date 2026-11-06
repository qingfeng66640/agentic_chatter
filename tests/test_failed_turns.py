"""失败回合终态隔离测试，所有业务、模型及聊天流访问均使用内存替身。"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from src.app.plugin_system.base import Wait
from src.kernel.llm import LLMPayload, ROLE, ToolCall, ToolResult

from .. import chatter as chatter_module
from ..chatter import AgenticChatter
from ..pipeline.mailbox import StreamMailbox, TurnClaim
from ..pipeline.state import ToolExecutionRecord, TurnState


SECRET = "原始异常含敏感输入，不得写入日志"


class _ModelError(RuntimeError):
    """模拟携带 HTTP 状态但不可安全输出正文的模型错误。"""

    status_code = 429


def _message(message_id: str) -> SimpleNamespace:
    """构造拥有稳定键和敏感正文的内存消息。"""
    return SimpleNamespace(message_id=message_id, content=SECRET)


async def _claim(mailbox: StreamMailbox, messages: list[Any]) -> TurnClaim:
    """取得真实 mailbox 所有权并领取指定消息。"""
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot(messages)
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None
    return claim


def _logs(logger: Mock) -> str:
    """汇总捕获的全部日志参数，防止异常通过附加字段泄露。"""
    return repr(logger.mock_calls)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """装配真实聊天器和 mailbox，仅替换外部依赖与模型请求。"""
    old = _message("old")
    snapshot = [old]
    context = SimpleNamespace(unread_messages=[old], history_messages=[])
    context.add_history_message = context.history_messages.append
    stream = SimpleNamespace(context=context)
    mailbox = StreamMailbox("failed-turn-test")
    chatter = AgenticChatter(stream_id=mailbox.stream_id, plugin=object())
    logger = Mock()
    request = SimpleNamespace(
        add_payload=Mock(), send=AsyncMock(side_effect=_ModelError(SECRET))
    )
    monkeypatch.setattr(chatter_module, "logger", logger)
    monkeypatch.setattr(
        chatter_module, "get_stream_mailbox", Mock(return_value=mailbox)
    )
    monkeypatch.setattr(
        chatter_module,
        "_restore_persisted_tasks",
        Mock(return_value=SimpleNamespace(get_active_tasks=Mock(return_value=[]))),
    )
    monkeypatch.setattr(
        chatter_module, "has_completed_events", Mock(return_value=False)
    )
    monkeypatch.setattr(
        "src.core.managers.get_stream_manager",
        Mock(
            return_value=SimpleNamespace(
                get_or_create_stream=AsyncMock(return_value=stream)
            )
        ),
    )
    monkeypatch.setattr(
        chatter_module.stream_api, "get_stream", AsyncMock(return_value=stream)
    )
    monkeypatch.setattr(
        chatter_module.send_api,
        "send_text",
        AsyncMock(side_effect=AssertionError("测试不得发送真实消息")),
    )
    monkeypatch.setattr(
        chatter, "fetch_unreads", AsyncMock(return_value=("", snapshot))
    )
    monkeypatch.setattr(chatter, "format_message_line", Mock(return_value="测试输入"))
    monkeypatch.setattr(chatter, "get_llm_usables", AsyncMock(return_value=[]))
    monkeypatch.setattr(chatter, "modify_llm_usables", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        chatter,
        "_build_layout",
        Mock(
            return_value=SimpleNamespace(
                exposed=[], collapsed_categories={}, collapsed_classes={}
            )
        ),
    )
    monkeypatch.setattr(
        chatter,
        "_register_usables_into_registry",
        Mock(return_value=SimpleNamespace(get_all=Mock(return_value=[]))),
    )
    monkeypatch.setattr(chatter, "_build_system_prompt", AsyncMock(return_value="系统"))
    monkeypatch.setattr(chatter, "_build_user_prompt", AsyncMock(return_value="输入"))
    monkeypatch.setattr(chatter, "_create_act_request", Mock(return_value=request))
    monkeypatch.setattr(chatter, "_streaming_allowed", Mock(return_value=False))
    monkeypatch.setattr(chatter_module, "skill_catalog", Mock(return_value=[]))
    for name in ("clear_scope", "clear_stream_catalog", "set_stream_catalog"):
        monkeypatch.setattr(chatter_module, name, Mock())
    return SimpleNamespace(
        old=old,
        snapshot=snapshot,
        context=context,
        stream=stream,
        mailbox=mailbox,
        chatter=chatter,
        logger=logger,
        request=request,
    )


@pytest.mark.parametrize("side_effects", [False, True])
async def test_failed_model_turn_is_not_replayed(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    side_effects: bool,
) -> None:
    """反复拉起和旧快照不重放模型或副作用，新输入仍可独立执行。"""
    states: list[TurnState] = []
    effects: list[tuple[str, ...]] = []
    original_act = harness.chatter._stage_act

    async def act(
        config: Any,
        chat_stream: Any,
        state: TurnState,
        unread_msgs: list[Any],
        *,
        claim: TurnClaim | None = None,
    ) -> None:
        """模拟已发言及已执行工具，然后进入真实模型失败路径。"""
        states.append(state)
        if side_effects:
            effects.append(tuple(message.message_id for message in unread_msgs))
            state.spoke = True
            state.tool_calls.append("fake-tool")
            state.tool_ledger.append(
                ToolExecutionRecord(
                    iteration=1,
                    call_id="fake-call",
                    name="fake-tool",
                    outcome="success",
                    result_capture="captured",
                    result_preview="ok",
                    counts_as_progress=True,
                )
            )
        await original_act(config, chat_stream, state, unread_msgs, claim=claim)

    monkeypatch.setattr(harness.chatter, "_stage_act", act)
    generator = harness.chatter._run_turns(None)
    try:
        results = [await generator.asend(None) for _ in range(3)]
        assert all(isinstance(result, Wait) for result in results)
        assert [result.time for result in results] == [0, None, None]
        assert harness.request.send.await_count == 1
        assert len(states) == 1 and states[0].failed
        assert not states[0].interrupted_by_new_input
        (record,) = await harness.mailbox.recent_failed_turns()
        assert record.failure_kind == "model_error"
        assert record.spoke is side_effects
        assert record.tool_count == int(side_effects)
        assert record.input_acknowledged
        assert harness.context.history_messages == [harness.old]
        assert await harness.mailbox.failed_messages() == ()
        assert effects == ([("old",)] if side_effects else [])
        assert "事件=回合已确认" not in _logs(harness.logger)
        assert "原因=输入中断" not in _logs(harness.logger)
        assert SECRET not in _logs(harness.logger)

        new = _message("new")
        harness.context.unread_messages.append(new)
        harness.snapshot.append(new)
        assert (await generator.asend(None)).time == 0
        assert (await generator.asend(None)).time is None
        assert harness.request.send.await_count == 2
        assert len(states) == 2 and all(state.failed for state in states)
        assert harness.context.history_messages == [harness.old, new]
        assert effects == ([("old",), ("new",)] if side_effects else [])
    finally:
        await generator.aclose()
    owner = object()
    generation = await harness.mailbox.try_acquire(owner)
    assert generation is not None
    assert await harness.mailbox.release_owner(owner, generation)


@pytest.mark.parametrize("ack", ["success", "error", "none"])
async def test_fail_claim_ack_and_isolation(ack: str) -> None:
    """ACK 成功仅移除原文，失败或缺省仍保留隔离且摘要不可变。"""
    mailbox = StreamMailbox("ack")
    old = _message("old")
    claim = await _claim(mailbox, [old])
    apply = Mock(side_effect=RuntimeError(SECRET) if ack == "error" else None)
    record = await mailbox.fail_claim(
        claim,
        failure_kind="model_error",
        error_type="RuntimeError",
        status_code=429,
        spoke=True,
        tool_count=2,
        apply=apply if ack != "none" else None,
    )
    assert record is not None
    assert record.generation == claim.generation and record.message_count == 1
    assert record.failure_kind == "model_error" and record.error_type == "RuntimeError"
    assert record.status_code == 429 and record.spoke and record.tool_count == 2
    assert record.message_digest and SECRET not in repr(asdict(record))
    assert set(asdict(record)) == {
        "generation",
        "message_count",
        "message_digest",
        "failure_kind",
        "error_type",
        "status_code",
        "spoke",
        "tool_count",
        "input_acknowledged",
        "acknowledgement_error_type",
    }
    assert record.input_acknowledged is (ack == "success")
    assert record.acknowledgement_error_type == (
        "RuntimeError" if ack == "error" else ""
    )
    isolated = await mailbox.failed_messages()
    assert isolated == (() if ack == "success" else (old,))
    if ack != "success":
        assert isolated[0] is old
    assert await mailbox.merge_snapshot([_message("old")]) == 0
    assert await mailbox.pending_count() == 0
    assert not await mailbox.release_claim(claim)
    assert await mailbox.release_owner(claim.owner, claim.generation)
    with pytest.raises(FrozenInstanceError):
        setattr(record, "spoke", False)
    assert apply.call_count == int(ack != "none")


async def test_stale_claim_owner_and_other_stream_are_isolated() -> None:
    """失效身份不得应用 ACK 或合并快照，失败键不影响其他流。"""
    mailbox = StreamMailbox("first")
    old = _message("old")
    claim = await _claim(mailbox, [old])
    apply = Mock()
    kwargs = dict(
        failure_kind="model_error", error_type="RuntimeError", spoke=False, tool_count=0
    )
    assert await mailbox.fail_claim(replace(claim), apply=apply, **kwargs) is None
    assert await mailbox.fail_claim(claim, **kwargs) is not None
    assert await mailbox.release_owner(claim.owner, claim.generation)
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    assert await mailbox.fail_claim(claim, apply=apply, **kwargs) is None
    apply.assert_not_called()
    assert (
        await mailbox.merge_snapshot(
            [_message("late")], owner=claim.owner, generation=claim.generation
        )
        == 0
    )
    assert (
        await mailbox.merge_snapshot(
            [_message("late")], owner=owner, generation=claim.generation
        )
        == 0
    )
    assert (
        await mailbox.merge_snapshot(
            [_message("late")], owner=claim.owner, generation=generation
        )
        == 0
    )
    assert (
        await mailbox.merge_snapshot(
            [_message("fresh")], owner=owner, generation=generation
        )
        == 1
    )
    assert await mailbox.pending_keys() == ("id:fresh",)
    assert await mailbox.release_owner(owner, generation)
    other = StreamMailbox("other")
    other_claim = await _claim(other, [old])
    assert other_claim.messages == (old,)
    assert await other.recent_failed_turns() == ()
    assert await other.release_claim(other_claim)
    assert await other.release_owner(other_claim.owner, other_claim.generation)


@pytest.mark.parametrize("acknowledged", [False, True])
async def test_failed_keys_survive_summary_eviction(acknowledged: bool) -> None:
    """超过三十二条摘要后，已 ACK 和未 ACK 的失败键都不能复活。"""
    mailbox = StreamMailbox("eviction")
    messages = [_message(str(index)) for index in range(35)]
    for message in messages:
        claim = await _claim(mailbox, [message])
        await mailbox.fail_claim(
            claim,
            failure_kind="model_error",
            error_type="RuntimeError",
            spoke=False,
            tool_count=0,
            apply=Mock() if acknowledged else None,
        )
        assert await mailbox.release_owner(claim.owner, claim.generation)
    assert len(await mailbox.recent_failed_turns()) == 32
    assert (
        await mailbox.merge_snapshot([_message(str(index)) for index in range(35)]) == 0
    )
    isolated = await mailbox.failed_messages()
    assert isolated == (() if acknowledged else tuple(messages))
    if not acknowledged:
        assert isolated[0] is messages[0]
    assert await mailbox.pending_count() == 0


async def test_input_interruption_releases_old_and_new_messages(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实新输入中断仍释放旧 claim，日志采用释放后的未读计数。"""
    claim = await _claim(harness.mailbox, [harness.old])
    new = _message("new")
    await harness.mailbox.merge_snapshot([new])
    await harness.mailbox.record_interruption()
    state = TurnState(
        stream_id=harness.mailbox.stream_id,
        failed=True,
        interrupted_by_new_input=True,
        error="生成期间收到新消息，重新规划",
    )
    monkeypatch.setattr(
        harness.chatter, "_run_pipeline", AsyncMock(return_value=state.to_outcome())
    )
    result, remaining = await harness.chatter._run_normal_turn(
        None,
        harness.mailbox,
        claim,
        harness.stream,
        state,
        [harness.old],
        claim.generation,
    )
    assert isinstance(result, Wait) and remaining is None
    assert await harness.mailbox.pending_keys() == ("id:old", "id:new")
    assert await harness.mailbox.recent_failed_turns() == ()
    assert "原因=输入中断" in _logs(harness.logger)
    assert "剩余未读=2" in _logs(harness.logger)
    assert "model_error" not in _logs(harness.logger)
    assert "语言模型调用失败" not in _logs(harness.logger)
    assert state.failed and state.interrupted_by_new_input
    assert await harness.mailbox.release_owner(claim.owner, claim.generation)


@pytest.mark.parametrize(
    "failure", ["outer", "match_error", "match_missing", "apply_error"]
)
async def test_exception_and_ack_failure_do_not_requeue_or_consume_new_input(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """外层及 ACK 异常均终态收尾，不能误消费生成期间新增的未读。"""
    new = _message("new")
    harness.context.unread_messages.append(new)
    if failure == "outer":
        monkeypatch.setattr(
            harness.chatter,
            "_run_pipeline",
            AsyncMock(side_effect=RuntimeError(SECRET)),
        )
    elif failure == "match_error":
        monkeypatch.setattr(
            harness.chatter,
            "_match_claim_unreads",
            AsyncMock(side_effect=RuntimeError(SECRET)),
        )
    elif failure == "match_missing":
        monkeypatch.setattr(
            harness.chatter, "_match_claim_unreads", AsyncMock(return_value=None)
        )
    else:
        harness.context.add_history_message = Mock(side_effect=RuntimeError(SECRET))
    generator = harness.chatter._run_turns(None)
    try:
        assert (await generator.asend(None)).time == 0
        assert (await generator.asend(None)).time is None
        assert (await generator.asend(None)).time is None
        (record,) = await harness.mailbox.recent_failed_turns()
        assert record.failure_kind == (
            "execution_error" if failure == "outer" else "model_error"
        )
        assert new in harness.context.unread_messages
        assert new not in harness.context.history_messages
        assert await harness.mailbox.pending_count() == 0
        if failure != "outer":
            assert not record.input_acknowledged
            assert await harness.mailbox.failed_messages() == (harness.old,)
            assert harness.context.unread_messages == [harness.old, new]
        assert SECRET not in _logs(harness.logger)
        assert "事件=回合已确认" not in _logs(harness.logger)
    finally:
        await generator.aclose()


@pytest.mark.parametrize("failed", [False, True])
async def test_cancellation_only_releases_nonfailed_claim(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    """取消必须传播；已标记失败的 claim 不得被 finally 重新排队。"""

    async def cancel_pipeline(*, state: TurnState, **kwargs: Any) -> None:
        """在失败标记之前或之后同步注入取消，不创建后台任务。"""
        if failed:
            state.failed = True
            state.failure_kind = "model_error"
            state.error_type = "RuntimeError"
            state.error = SECRET
        raise asyncio.CancelledError

    monkeypatch.setattr(harness.chatter, "_run_pipeline", cancel_pipeline)
    generator = harness.chatter._run_turns(None)
    with pytest.raises(asyncio.CancelledError):
        await generator.asend(None)
    await generator.aclose()
    if failed:
        assert await harness.mailbox.pending_count() == 0
        assert await harness.mailbox.merge_snapshot([harness.old]) == 0
        assert len(await harness.mailbox.recent_failed_turns()) == 1
    else:
        assert await harness.mailbox.pending_keys() == ("id:old",)
        assert await harness.mailbox.recent_failed_turns() == ()
    owner = object()
    generation = await harness.mailbox.try_acquire(owner)
    assert generation is not None
    assert await harness.mailbox.release_owner(owner, generation)


async def test_stage_act_classifies_model_failure_without_sensitive_logs(
    harness: SimpleNamespace,
) -> None:
    """真实 act 捕获模型错误，只记录类型状态，不伪装成新输入中断。"""
    state = TurnState(stream_id=harness.mailbox.stream_id)
    await harness.chatter._stage_act(None, harness.stream, state, [harness.old])
    assert state.failed and state.failure_kind == "model_error"
    assert state.error_type == "_ModelError" and state.error_status_code == 429
    assert state.error == SECRET and not state.interrupted_by_new_input
    harness.request.send.assert_awaited_once_with(stream=True)
    logs = _logs(harness.logger)
    assert SECRET not in logs
    assert "_ModelError" in logs and "429" in logs


async def test_pipeline_does_not_reflect_after_failure(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实管线发生失败后停止后续反思，不将失败状态改成成功。"""
    state = TurnState(stream_id=harness.mailbox.stream_id)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            stage_order=["act", "reflect"],
            enable_perceive=False,
            enable_plan=False,
            enable_reflect=True,
        ),
        decision=SimpleNamespace(enabled=False),
    )

    async def fail_act(*args: Any, **kwargs: Any) -> None:
        """模拟行动阶段失败而不触发外部依赖。"""
        state.failed = True
        state.failure_kind = "model_error"

    reflect = AsyncMock()
    monkeypatch.setattr(harness.chatter, "_stage_act", fail_act)
    monkeypatch.setattr(harness.chatter, "_stage_reflect", reflect)
    await harness.chatter._run_pipeline(
        config=config,
        chat_stream=harness.stream,
        state=state,
        unread_msgs=[harness.old],
    )
    reflect.assert_not_awaited()
    assert state.failed


async def test_tool_results_then_model_failure_do_not_replay(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 ACT 发送文字并回灌工具后，第二次模型失败不得从头重放。"""

    class ToolResponse:
        """首轮模型响应包含文本和工具，后续请求失败。"""

        def __init__(self) -> None:
            """初始化响应正文、工具调用和载荷列表。"""
            self.message = "这次开始处理。"
            self.call_list = [ToolCall("call-one", "tool-probe", {})]
            self.payloads: list[LLMPayload] = []
            self.send = AsyncMock(side_effect=_ModelError(SECRET))

        async def ready(self) -> ToolResponse:
            """提供真实 ACT 所需的 awaitable 响应接口。"""
            return self

        def __await__(self) -> Any:
            """让响应等待至本地 ready，不创建后台任务。"""
            return self.ready().__await__()

        def add_payload(
            self, payload: LLMPayload, position: Any = None
        ) -> ToolResponse:
            """保存真实工具提交缓冲回灌的载荷。"""
            self.payloads.append(payload)
            return self

    response = ToolResponse()
    harness.request.send.side_effect = None
    harness.request.send.return_value = response
    sends = AsyncMock(return_value=True)
    monkeypatch.setattr(chatter_module.send_api, "send_text", sends)
    monkeypatch.setattr(chatter_module, "consume_expansion", Mock(return_value=[]))

    async def dispatch(
        calls: list[Any], receiver: Any, *args: Any
    ) -> list[tuple[bool, bool]]:
        """仅替换工具实际业务，保留真实提交、回灌和执行账本路径。"""
        receiver.add_payload(
            LLMPayload(
                ROLE.TOOL_RESULT,
                ToolResult("done", calls[0].id, calls[0].name),
            )
        )
        return [(True, True)]

    tool = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr(harness.chatter, "run_tool_call", tool)
    generator = harness.chatter._run_turns(None)
    try:
        assert (await generator.asend(None)).time == 0
        for _ in range(3):
            assert (await generator.asend(None)).time is None
        harness.request.send.assert_awaited_once()
        response.send.assert_awaited_once()
        sends.assert_awaited_once()
        tool.assert_awaited_once()
        (record,) = await harness.mailbox.recent_failed_turns()
        assert record.failure_kind == "model_error"
        assert record.spoke and record.tool_count == 1 and record.input_acknowledged
        assert SECRET not in _logs(harness.logger)
    finally:
        await generator.aclose()


async def test_new_input_arriving_during_failure_is_drained_once(
    harness: SimpleNamespace,
) -> None:
    """失败期间已经到达的新消息不能被 Wait(None) 永久挂起。"""
    new = _message("arrived-during-model")
    attempts = 0

    async def fail_request(**kwargs: Any) -> None:
        """首个请求期间注入新消息，然后让两个请求均失败。"""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            harness.context.unread_messages.append(new)
            harness.snapshot.append(new)
        raise _ModelError(SECRET)

    harness.request.send.side_effect = fail_request
    generator = harness.chatter._run_turns(None)
    try:
        assert (await generator.asend(None)).time == 0
        assert harness.context.history_messages == [harness.old]
        assert harness.context.unread_messages == [new]
        assert (await generator.asend(None)).time == 0
        assert (await generator.asend(None)).time is None
        assert attempts == 2
        assert harness.context.history_messages == [harness.old, new]
    finally:
        await generator.aclose()


async def test_outer_route_exception_is_terminal(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """领取消息后尚未进入管线的路由异常也不得无限重放。"""
    route = Mock(side_effect=RuntimeError(SECRET))
    monkeypatch.setattr(chatter_module, "_restore_persisted_tasks", route)
    generator = harness.chatter._run_turns(None)
    try:
        assert (await generator.asend(None)).time == 0
        assert (await generator.asend(None)).time is None
        route.assert_called_once()
        harness.request.send.assert_not_awaited()
        (record,) = await harness.mailbox.recent_failed_turns()
        assert record.failure_kind == "execution_error"
        assert record.input_acknowledged
        assert SECRET not in _logs(harness.logger)
    finally:
        await generator.aclose()


async def test_successful_pipeline_confirmation_error_is_terminal(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务已成功但确认时抛错，同样不能通过 finally 重放原输入。"""
    state_seen: list[TurnState] = []

    async def pipeline(*, state: TurnState, **kwargs: Any) -> Any:
        """模拟已完成的对外发言。"""
        state.spoke = True
        state_seen.append(state)
        return state.to_outcome()

    monkeypatch.setattr(harness.chatter, "_run_pipeline", pipeline)
    monkeypatch.setattr(
        harness.mailbox, "commit_claim", AsyncMock(side_effect=RuntimeError(SECRET))
    )
    generator = harness.chatter._run_turns(None)
    try:
        assert (await generator.asend(None)).time == 0
        assert (await generator.asend(None)).time is None
        assert len(state_seen) == 1 and state_seen[0].failed
        (record,) = await harness.mailbox.recent_failed_turns()
        assert record.spoke and record.failure_kind == "execution_error"
        assert await harness.mailbox.pending_count() == 0
    finally:
        await generator.aclose()


async def test_cancel_during_failed_input_confirmation_preserves_isolation(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型失败后，归档匹配期间取消也必须隔离原消息并释放所有权。"""
    monkeypatch.setattr(
        harness.chatter,
        "_match_claim_unreads",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    generator = harness.chatter._run_turns(None)
    with pytest.raises(asyncio.CancelledError):
        await generator.asend(None)
    await generator.aclose()
    assert await harness.mailbox.failed_messages() == (harness.old,)
    assert await harness.mailbox.merge_snapshot([harness.old]) == 0
    assert await harness.mailbox.pending_count() == 0
    assert harness.context.unread_messages == [harness.old]
    owner = object()
    generation = await harness.mailbox.try_acquire(owner)
    assert generation is not None
    assert await harness.mailbox.release_owner(owner, generation)

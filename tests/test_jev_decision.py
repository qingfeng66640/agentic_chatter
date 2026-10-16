"""Jev 灰区裁决测试。

覆盖四类路径：

- state 构建与截断；
- answer 解析（正常、非法结构、非法选项、置信度门限）；
- HTTP 层的正常应答与失败处理（401/422/429/529/超时/传输错误）；
- ``_decide_gray_zone_with_jev`` 的开关、失败回退与接线行为。

所有 HTTP 交互均通过 httpx MockTransport 拦截，不产生真实网络请求。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from ..decision import action_text
from ..decision.jev import (
    JevError,
    JevSettings,
    _MessageView,
    build_decision,
    build_fallback_decision,
    build_state,
    classify_failure,
    format_failure_fields,
    parse_choice_answer,
    redact_url,
    request_decision,
    status_failure_code,
)
from ..decision.models import DecisionAction, DecisionSource, describe_decision

# 占位符直接写死而不从实现里导入：这样"占位符被改名"会立刻让测试失败，
# 而从实现导入只会跟着一起变、永远测不出问题。取值须与
# tooling.dedupe 的日志脱敏保持一致，运维才能用一个串扫全插件日志。
REDACTED = "[REDACTED]"


def _settings(**overrides) -> JevSettings:
    """构造带默认值的 JevSettings。"""
    base = {
        "api_key": "ts_test_key",
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
    }
    base.update(overrides)
    return JevSettings(**base)

def _choice_payload(choice: str, confidence: float) -> dict:
    """构造一个合法的 System One 响应体。"""
    return {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"respond": 0.9, "silent": 0.1}
                if choice == "respond"
                else {"respond": 0.1, "silent": 0.9},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def _client(handler) -> httpx.AsyncClient:
    """用 MockTransport 构造客户端。"""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# state 构建
# ---------------------------------------------------------------------------


def test_build_state_keeps_only_configured_history_window() -> None:
    history = [_MessageView(sender=f"m{index}", text=f"历史{index}") for index in range(5)]
    incoming = [_MessageView(sender="u", text="在吗")]

    state = build_state(
        incoming=incoming,
        history=history,
        signals={"question_or_request": 1.0},
        max_history_messages=2,
    )

    # 只保留最近 2 条历史
    assert [entry["text"] for entry in state["messages"][:2]] == ["历史3", "历史4"]
    assert state["incoming"] == [
        {"sender": "u", "role": "member", "text": "在吗"}
    ]


def test_build_state_marks_bot_messages() -> None:
    history = [_MessageView(sender="bot", text="我是bot", is_bot=True)]

    state = build_state(
        incoming=[],
        history=history,
        signals={},
        max_history_messages=5,
    )

    assert state["messages"][0]["role"] == "bot"
    assert state["bot"]["is_chatbot"] is True


def test_build_state_drops_empty_messages() -> None:
    state = build_state(
        incoming=[_MessageView(sender="a", text="   ")],
        history=[_MessageView(sender="b", text="")],
        signals={},
        max_history_messages=5,
    )

    assert state["messages"] == []
    assert state["incoming"] == []


def test_build_state_with_zero_history_window() -> None:
    state = build_state(
        incoming=[_MessageView(sender="a", text="你好")],
        history=[_MessageView(sender="b", text="不该出现")],
        signals={},
        max_history_messages=0,
    )

    # 历史窗口为 0 时不送历史，但待判断消息仍保留
    assert [entry["text"] for entry in state["messages"]] == ["你好"]


def test_message_view_to_state_marks_bot_role() -> None:
    view = _MessageView(sender="a", text="你好", is_bot=True)

    assert view.to_state() == {"sender": "a", "role": "bot", "text": "你好"}


def test_build_state_uses_pre_trimmed_text() -> None:
    """长文本的截断由调用方在构造 _MessageView 前完成。"""
    state = build_state(
        incoming=[_MessageView(sender="a", text="很长" * 100)],
        history=[],
        signals={},
        max_history_messages=1,
    )

    # build_state 不再二次截断，直接透传调用方已裁剪的文本
    assert len(state["incoming"][0]["text"]) == 200


# ---------------------------------------------------------------------------
# answer 解析
# ---------------------------------------------------------------------------


def test_parse_choice_answer_respond() -> None:
    answer = _choice_payload("respond", 0.91)["answers"]["action"]

    result = parse_choice_answer(answer, min_confidence=0.0)

    assert result.action is DecisionAction.RESPOND
    assert result.confidence == pytest.approx(0.91)
    assert result.respond_probability == pytest.approx(0.9)
    assert result.silent_probability == pytest.approx(0.1)


def test_parse_choice_answer_silent() -> None:
    answer = _choice_payload("silent", 0.75)["answers"]["action"]

    result = parse_choice_answer(answer, min_confidence=0.0)

    assert result.action is DecisionAction.SILENT
    assert result.silent_probability == pytest.approx(0.9)
    assert result.respond_probability == pytest.approx(0.1)


def test_parse_choice_answer_rejects_non_dict() -> None:
    with pytest.raises(JevError):
        parse_choice_answer("not-a-dict", min_confidence=0.0)


def test_parse_choice_answer_rejects_unknown_option() -> None:
    with pytest.raises(JevError):
        parse_choice_answer(
            {"choice": "maybe", "confidence": 0.9},
            min_confidence=0.0,
        )


def test_parse_choice_answer_rejects_out_of_range_confidence() -> None:
    with pytest.raises(JevError):
        parse_choice_answer({"choice": "respond", "confidence": 1.5}, min_confidence=0.0)
    with pytest.raises(JevError):
        parse_choice_answer({"choice": "respond", "confidence": -0.2}, min_confidence=0.0)


def test_parse_choice_answer_enforces_min_confidence() -> None:
    answer = _choice_payload("respond", 0.4)["answers"]["action"]

    with pytest.raises(JevError, match="置信度低于门限"):
        parse_choice_answer(answer, min_confidence=0.8)


def test_parse_choice_answer_allows_boundary_confidence() -> None:
    answer = _choice_payload("respond", 0.8)["answers"]["action"]

    result = parse_choice_answer(answer, min_confidence=0.8)

    assert result.action is DecisionAction.RESPOND


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------


async def test_request_decision_success_sends_expected_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_choice_payload("respond", 0.88))

    async with _client(handler) as client:
        result = await request_decision(
            client,
            settings=_settings(),
            state={"messages": []},
        )

    assert result.action is DecisionAction.RESPOND
    assert result.confidence == pytest.approx(0.88)
    assert result.input_tokens == 100
    assert result.output_tokens == 10
    assert captured["url"] == _settings().endpoint
    assert captured["auth"] == "Bearer ts_test_key"
    # 选项顺序固定：respond 在前
    criteria = captured["body"]["questions"]["action"]["criteria"]
    assert list(criteria) == ["respond", "silent"]
    assert captured["body"]["model"] == "jev-latest"


async def test_request_decision_auth_failure_is_not_retried() -> None:
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    async with _client(handler) as client:
        with pytest.raises(JevError, match="HTTP 401"):
            await request_decision(
                client, settings=_settings(max_retries=3), state={}
            )

    # 鉴权失败不重试
    assert calls["count"] == 1


async def test_request_decision_validation_error_is_not_retried() -> None:
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(422, json={"error": "bad request"})

    async with _client(handler) as client:
        with pytest.raises(JevError, match="HTTP 422"):
            await request_decision(
                client, settings=_settings(max_retries=2), state={}
            )

    assert calls["count"] == 1


async def test_request_decision_retries_on_rate_limit_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_choice_payload("silent", 0.7))

    async with _client(handler) as client:
        result = await request_decision(
            client,
            settings=_settings(max_retries=1),
            state={},
        )

    assert calls["count"] == 2
    assert result.action is DecisionAction.SILENT
    assert result.confidence == pytest.approx(0.7)


async def test_request_decision_retries_on_overload() -> None:
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(529, json={"error": "overloaded"})

    async with _client(handler) as client:
        with pytest.raises(JevError, match="529"):
            await request_decision(
                client, settings=_settings(max_retries=1), state={}
            )

    # 首次 + 1 次重试
    assert calls["count"] == 2


async def test_request_decision_handles_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(JevError):
            await request_decision(client, settings=_settings(), state={})


async def test_request_decision_rejects_non_json_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    async with _client(handler) as client:
        with pytest.raises(JevError, match="不是合法 JSON"):
            await request_decision(client, settings=_settings(), state={})


async def test_request_decision_rejects_missing_answer_key() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"other": {}}})

    async with _client(handler) as client:
        with pytest.raises(JevError, match="缺少 answers.action"):
            await request_decision(client, settings=_settings(), state={})


async def test_request_decision_propagates_low_confidence_as_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_choice_payload("respond", 0.2))

    async with _client(handler) as client:
        with pytest.raises(JevError, match="置信度低于门限"):
            await request_decision(
                client,
                settings=_settings(min_confidence=0.6),
                state={},
            )


# ---------------------------------------------------------------------------
# 决策包装
# ---------------------------------------------------------------------------


def test_build_decision_uses_jev_source_and_marks_jev() -> None:
    decision = build_decision(
        DecisionAction.RESPOND,
        0.83,
        local_score=0.05,
        lower_bound=-0.1,
        upper_bound=0.2,
        reasons=["question_or_request"],
    )

    assert decision.action is DecisionAction.RESPOND
    assert decision.source is DecisionSource.JEV
    assert decision.confidence == pytest.approx(0.83)
    assert decision.reasons[-1] == "jev_decided"
    # Jev 不是 sub_actor，不应标记为已使用子决策模型
    assert decision.sub_actor_used is False
    # 本地评分原样保留，便于日志对照
    assert decision.score == pytest.approx(0.05)


def test_build_fallback_decision_is_silent_and_never_sub_actor() -> None:
    decision = build_fallback_decision(
        local_score=0.02,
        lower_bound=-0.3,
        upper_bound=0.34,
        reasons=["question_or_request"],
        detail="jev_api_key_missing",
    )

    assert decision.action is DecisionAction.SILENT
    assert decision.source is DecisionSource.FALLBACK
    assert decision.confidence == pytest.approx(0.0)
    assert decision.reasons[-1] == "jev_api_key_missing"
    assert decision.sub_actor_used is False


# ---------------------------------------------------------------------------
# chatter 接线
# ---------------------------------------------------------------------------


class _JevChatter:
    """只暴露 _decide_gray_zone_with_jev 所需表面的最小替身。"""

    def __init__(self, stream_id: str = "stream-abcdef1234567890") -> None:
        self.stream_id = stream_id

    from ..chatter import _AgenticChatterBase as _Base

    _jev_unavailable = _Base._jev_unavailable
    _decide_gray_zone_with_jev = _Base._decide_gray_zone_with_jev


def _config(
    enabled: bool = True,
    api_key: str = "ts_key",
    *,
    fallback_decider: str = "sub_actor",
    **jev_overrides,
):
    """构造仅含 jev 段的插件配置替身。"""
    jev = SimpleNamespace(
        enabled=enabled,
        api_key=api_key,
        fallback_decider=fallback_decider,
        model=jev_overrides.get("model", "jev-latest"),
        timeout_seconds=jev_overrides.get("timeout_seconds", 8.0),
        total_budget_seconds=jev_overrides.get("total_budget_seconds", 12.0),
        max_history_messages=jev_overrides.get("max_history_messages", 8),
        max_retries=jev_overrides.get("max_retries", 0),
        retry_backoff_seconds=jev_overrides.get("retry_backoff_seconds", 0.0),
        min_confidence=jev_overrides.get("min_confidence", 0.0),
        log_answers=jev_overrides.get("log_answers", False),
    )
    return SimpleNamespace(jev=jev)


def _decision_config(decider: str = "jev") -> SimpleNamespace:
    """构造回复决策配置替身。"""
    return SimpleNamespace(gray_zone_decider=decider, fallback_mode="contextual")


def _chat_stream(bot_id: str = "bot-1") -> SimpleNamespace:
    return SimpleNamespace(bot_id=bot_id)


def _messages() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            sender_id="bot-1",
            sender_name="小蝶",
            sender_role="bot",
            content="刚才提到过天气",
            processed_plain_text=None,
        ),
        SimpleNamespace(
            sender_id="u-1",
            sender_name="小明",
            sender_role="user",
            content="那明天会下雨吗？",
            processed_plain_text=None,
        ),
    ]


def _patch_http(monkeypatch, handler, captured: dict | None = None):
    """把 chatter 使用的共享客户端替换为 MockTransport 版本。

    ``chatter`` 在导入时把 ``get_shared_client`` 绑定到自身命名空间，
    因此必须替换 ``chatter`` 上的符号，而不是 ``decision.jev`` 上的。
    """
    from .. import chatter as chatter_module

    def wrapped(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
        return handler(request)

    original_client = httpx.AsyncClient

    def fake_client(*_args, **kwargs):
        kwargs.pop("timeout", None)
        return original_client(transport=httpx.MockTransport(wrapped), **kwargs)

    monkeypatch.setattr(chatter_module, "get_shared_client", fake_client)


def _ok_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_choice_payload("respond", 0.9))


async def _run_jev(
    monkeypatch,
    config,
    *,
    decider: str = "jev",
    handler=_ok_handler,
    captured: dict | None = None,
):
    """执行一次灰区裁决；handler 默认返回合法应答。"""
    _patch_http(monkeypatch, handler, captured)
    chatter = _JevChatter()
    return await chatter._decide_gray_zone_with_jev(
        config,
        chat_stream=_chat_stream(),
        decision_config=_decision_config(decider),
        history_messages=[],
        unread_msgs=_messages(),
        signal_summary={"question_or_request": 1.0},
        bot_nicknames=("小蝶",),
        bot_message_ids=frozenset(),
        score=0.0,
        lower=-0.2,
        upper=0.2,
        reasons=["question_or_request"],
    )


# --- 裁决者枚举分派 ------------------------------------------------------


async def test_sub_actor_decider_skips_jev_entirely(monkeypatch) -> None:
    """decider=sub_actor 时即使 Jev 完全可用也不发请求。"""
    calls = {"count": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _ok_handler(request)

    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key"),
        decider="sub_actor",
        handler=counting_handler,
    )

    assert decision is None
    assert calls["count"] == 0


async def test_unknown_decider_value_falls_back_to_sub_actor(monkeypatch) -> None:
    calls = {"count": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _ok_handler(request)

    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key"),
        decider="不存在的值",
        handler=counting_handler,
    )

    assert decision is None
    assert calls["count"] == 0


async def test_returns_none_when_config_is_none() -> None:
    chatter = _JevChatter()

    decision = await chatter._decide_gray_zone_with_jev(
        None,
        chat_stream=_chat_stream(),
        decision_config=_decision_config("jev"),
        history_messages=[],
        unread_msgs=_messages(),
        signal_summary={},
        bot_nicknames=("小蝶",),
        bot_message_ids=frozenset(),
        score=0.0,
        lower=0.0,
        upper=0.0,
        reasons=[],
    )

    assert decision is None


# --- 成功路径 ------------------------------------------------------------


async def test_chatter_produces_decision_on_success(monkeypatch) -> None:
    captured: dict = {}

    decision = await _run_jev(
        monkeypatch, _config(True, "ts_key"), captured=captured
    )

    assert decision is not None
    assert decision.action is DecisionAction.RESPOND
    assert decision.source is DecisionSource.JEV
    assert decision.confidence == pytest.approx(0.9)
    assert "jev_decided" in decision.reasons
    assert decision.sub_actor_used is False
    # 本地评分信息被保留，便于与 Jev 结果对照
    assert decision.score == pytest.approx(0.0)
    assert decision.lower_bound == pytest.approx(-0.2)
    assert decision.upper_bound == pytest.approx(0.2)
    # 待判断消息与信号确实送达
    assert len(captured["body"]["state"]["incoming"]) == 2


async def test_chatter_marks_bot_messages_in_state(monkeypatch) -> None:
    captured: dict = {}

    await _run_jev(monkeypatch, _config(True, "ts_key"), captured=captured)

    roles = [entry["role"] for entry in captured["body"]["state"]["incoming"]]
    assert roles == ["bot", "member"]


# --- fallback_decider = sub_actor：返回 None 交由调用方回退 --------------


async def test_returns_none_when_jev_disabled(monkeypatch) -> None:
    decision = await _run_jev(monkeypatch, _config(False, "ts_key"))

    assert decision is None


async def test_returns_none_when_api_key_missing(monkeypatch) -> None:
    decision = await _run_jev(monkeypatch, _config(True, "   "))

    assert decision is None


async def test_returns_none_on_http_failure(monkeypatch) -> None:
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key"),
        handler=lambda _r: httpx.Response(401, json={"error": "unauthorized"}),
    )

    assert decision is None


async def test_returns_none_on_low_confidence(monkeypatch) -> None:
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key", min_confidence=0.7),
        handler=lambda _r: httpx.Response(
            200, json=_choice_payload("respond", 0.3)
        ),
    )

    assert decision is None


# --- fallback_decider = silent：绝不混入 sub_actor ------------------------


async def test_silent_fallback_on_disabled() -> None:
    chatter = _JevChatter()

    decision = await chatter._decide_gray_zone_with_jev(
        _config(False, "ts_key", fallback_decider="silent"),
        chat_stream=_chat_stream(),
        decision_config=_decision_config("jev"),
        history_messages=[],
        unread_msgs=_messages(),
        signal_summary={},
        bot_nicknames=("小蝶",),
        bot_message_ids=frozenset(),
        score=0.02,
        lower=-0.3,
        upper=0.34,
        reasons=["question_or_request"],
    )

    assert decision is not None
    assert decision.action is DecisionAction.SILENT
    assert decision.source is DecisionSource.FALLBACK
    assert decision.reasons[-1] == "jev_disabled"
    assert decision.sub_actor_used is False


async def test_silent_fallback_on_missing_api_key() -> None:
    chatter = _JevChatter()

    decision = await chatter._decide_gray_zone_with_jev(
        _config(True, "", fallback_decider="silent"),
        chat_stream=_chat_stream(),
        decision_config=_decision_config("jev"),
        history_messages=[],
        unread_msgs=_messages(),
        signal_summary={},
        bot_nicknames=("小蝶",),
        bot_message_ids=frozenset(),
        score=0.0,
        lower=0.0,
        upper=0.0,
        reasons=[],
    )

    assert decision is not None
    assert decision.action is DecisionAction.SILENT
    assert decision.reasons[-1] == "jev_api_key_missing"


async def test_silent_fallback_on_http_failure(monkeypatch) -> None:
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key", fallback_decider="silent"),
        handler=lambda _r: httpx.Response(401, json={"error": "unauthorized"}),
    )

    assert decision is not None
    assert decision.action is DecisionAction.SILENT
    assert decision.source is DecisionSource.FALLBACK
    assert decision.reasons[-1].startswith("jev_error:")
    assert decision.sub_actor_used is False


async def test_silent_fallback_on_low_confidence(monkeypatch) -> None:
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key", fallback_decider="silent", min_confidence=0.7),
        handler=lambda _r: httpx.Response(
            200, json=_choice_payload("respond", 0.3)
        ),
    )

    assert decision is not None
    assert decision.action is DecisionAction.SILENT
    assert decision.sub_actor_used is False


async def test_silent_fallback_ignores_unknown_fallback_value(monkeypatch) -> None:
    """非法 fallback_decider 值按 sub_actor 处理，不静默。"""
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key", fallback_decider="bogus"),
        handler=lambda _r: httpx.Response(401, json={"error": "unauthorized"}),
    )

    assert decision is None


# --- 总预算封顶（对抗审查发现：重试会把 decide 阶段阻塞放大到分钟级）----


async def test_total_budget_stops_further_retries() -> None:
    """预算耗尽后不再重试，即使 max_retries 还有余量。"""
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(529, json={"error": "overloaded"})

    async with _client(handler) as client:
        with pytest.raises(JevError):
            await request_decision(
                client,
                settings=_settings(
                    max_retries=5,
                    total_budget_seconds=1.0,
                    retry_backoff_seconds=5.0,
                ),
                state={},
            )

    # 退避 5 秒远超总预算 1 秒，因此不会进入第二次尝试
    assert calls["count"] == 1


async def test_per_attempt_timeout_capped_by_remaining_budget() -> None:
    """单次尝试超时不得超过剩余总预算。"""
    seen: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=_choice_payload("respond", 0.9))

    async with _client(handler) as client:
        result = await request_decision(
            client,
            settings=_settings(timeout_seconds=60.0, total_budget_seconds=5.0),
            state={},
        )

    assert result.action is DecisionAction.RESPOND
    timeout_ext = seen[0]
    assert timeout_ext is not None
    assert timeout_ext["read"] <= 5.0


# --- 共享客户端复用 --------------------------------------------------------


async def test_shared_client_is_reused(monkeypatch) -> None:
    """共享客户端应复用同一实例，避免每次裁决重建连接池。"""
    from ..decision import jev as jev_module

    first = jev_module.get_shared_client()
    second = jev_module.get_shared_client()
    assert first is second
    await jev_module.close_shared_client()


# --- signals 字段自解释（M4：System One 不是推理模型）--------------------


def test_signals_carry_chinese_legend() -> None:
    """本地信号字段必须附带中文说明与取值范围。"""
    state = build_state(
        incoming=[_MessageView(sender="a", text="在吗")],
        history=[],
        signals={
            "question_or_request": 1.0,
            "semantic_continuity": 0.5,
            "unknown_signal": 1.0,
        },
        max_history_messages=5,
    )

    keys = list(state["signals"])
    # 已知字段带图例
    assert any(key.startswith("question_or_request（") for key in keys)
    assert any(key.startswith("semantic_continuity（") for key in keys)
    # 图例需说明取值范围
    legend = next(key for key in keys if key.startswith("semantic_continuity（"))
    assert "0" in legend and "1" in legend
    # 未知字段原样透传，不报错
    assert "unknown_signal" in state["signals"]


# --- usage token 透传 -----------------------------------------------------


async def test_request_decision_exposes_usage_tokens() -> None:
    """token 用量需透传到返回值，供 log_answers 观测成本。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_choice_payload("respond", 0.9))

    async with _client(handler) as client:
        result = await request_decision(client, settings=_settings(), state={})

    assert result.input_tokens == 100
    assert result.output_tokens == 10


# --- base_url 可配置（不得硬编码）---------------------------------------


def test_endpoint_defaults_to_official_url() -> None:
    assert (
        JevSettings(api_key="k").endpoint
        == "https://api.typesafe.ai/v1/systemone"
    )


def test_endpoint_accepts_custom_gateway() -> None:
    settings = JevSettings(api_key="k", base_url="https://my-proxy.internal")

    assert settings.endpoint == "https://my-proxy.internal/v1/systemone"


def test_endpoint_tolerates_trailing_slash() -> None:
    settings = JevSettings(api_key="k", base_url="https://my-proxy.internal/")

    assert settings.endpoint == "https://my-proxy.internal/v1/systemone"


def test_endpoint_does_not_double_path() -> None:
    """用户直接填到完整路径时不应拼成 .../v1/systemone/v1/systemone。"""
    settings = JevSettings(
        api_key="k", base_url="https://api.typesafe.ai/v1/systemone"
    )

    assert settings.endpoint == "https://api.typesafe.ai/v1/systemone"


def test_endpoint_supports_local_proxy_with_path_prefix() -> None:
    settings = JevSettings(
        api_key="k", base_url="http://127.0.0.1:8080/jev-proxy"
    )

    assert settings.endpoint == "http://127.0.0.1:8080/jev-proxy/v1/systemone"


def test_endpoint_falls_back_when_base_url_blank() -> None:
    settings = JevSettings(api_key="k", base_url="   ")

    assert settings.endpoint == "https://api.typesafe.ai/v1/systemone"


async def test_custom_base_url_is_actually_used(monkeypatch) -> None:
    """端到端确认请求真的发往自定义地址，而不是硬编码域名。"""
    captured: dict = {}

    _patch_http(
        monkeypatch,
        _ok_handler,
        captured,
    )
    chatter = _JevChatter()
    config = _config(True, "ts_key")
    config.jev.base_url = "https://my-proxy.internal"

    await chatter._decide_gray_zone_with_jev(
        config,
        chat_stream=_chat_stream(),
        decision_config=_decision_config("jev"),
        history_messages=[],
        unread_msgs=_messages(),
        signal_summary={},
        bot_nicknames=("小蝶",),
        bot_message_ids=frozenset(),
        score=0.0,
        lower=0.0,
        upper=0.0,
        reasons=[],
    )

    assert captured["url"] == "https://my-proxy.internal/v1/systemone"

# --- bot 身份与召唤信号（回归：曾因只发 is_chatbot 导致喊名字不回应）------


def test_state_declares_bot_names() -> None:
    """state 必须告知 Jev 群里怎么称呼它，否则无法识别召唤。"""
    state = build_state(
        incoming=[_MessageView(sender="小明", text="小蝶，在吗")],
        history=[],
        signals={"direct_address": 1.0},
        max_history_messages=5,
        bot_names=("小蝶", "蝶蝶"),
    )

    bot = state["bot"]
    assert bot["is_chatbot"] is True
    assert bot["names"] == ["小蝶", "蝶蝶"]
    # note 不重复具体昵称，只指向 names 字段，避免与 names 不同步
    assert "names" in bot["note"]


def test_state_without_names_falls_back_to_pronoun_note() -> None:
    state = build_state(
        incoming=[],
        history=[],
        signals={},
        max_history_messages=5,
    )

    assert state["bot"]["names"] == []
    assert "没有登记" in state["bot"]["note"]


def test_direct_address_signal_has_legend() -> None:
    """direct_address 必须有图例，否则 Jev 不知道 1 意味着被点名。"""
    state = build_state(
        incoming=[],
        history=[],
        signals={"direct_address": 1.0},
        max_history_messages=5,
    )

    keys = list(state["signals"])
    assert any(key.startswith("direct_address（") for key in keys)


def test_instructions_mention_calling_bot_by_name() -> None:
    """instructions 必须写明以 bot 名字开头即召唤。"""
    from ..decision import jev as jev_module

    text = jev_module._INSTRUCTIONS
    assert "names" in text
    assert "@" in text


# ---------------------------------------------------------------------------
# 失败原因码
#
# 回归背景：此前 chatter 用 f"jev_error:{type(exc).__name__}" 记录失败，而
# JevError 没有子类、全部失败都是同一个类名，日志里恒为
# "jev_error:JevError"，HTTP 状态码、限流、超时、结构非法全部无法区分。
# 下面这组用例逐点锁死"码必须随死因变化"。
# ---------------------------------------------------------------------------


def test_status_failure_code_maps_known_statuses() -> None:
    assert status_failure_code(401) == "jev_error:auth:401"
    assert status_failure_code(403) == "jev_error:auth:403"
    assert status_failure_code(402) == "jev_error:payment:402"
    assert status_failure_code(404) == "jev_error:not_found:404"
    assert status_failure_code(400) == "jev_error:bad_request:400"
    assert status_failure_code(422) == "jev_error:bad_request:422"
    assert status_failure_code(429) == "jev_error:rate_limited:429"
    assert status_failure_code(529) == "jev_error:overloaded:529"


def test_status_failure_code_keeps_unmapped_status_verbatim() -> None:
    """上游新增状态码时不丢原始码，只是不细分类别。"""
    assert status_failure_code(503) == "jev_error:http:503"
    assert status_failure_code(418) == "jev_error:http:418"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "jev_error:auth:401"),
        (402, "jev_error:payment:402"),
        (404, "jev_error:not_found:404"),
        (422, "jev_error:bad_request:422"),
        (500, "jev_error:http:500"),
    ],
)
async def test_request_decision_non_retryable_status_carries_code(status, expected) -> None:
    async with _client(lambda _r: httpx.Response(status, json={"error": "x"})) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(client, settings=_settings(), state={})

    assert excinfo.value.code == expected
    assert excinfo.value.status == status
    assert excinfo.value.attempts == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, "jev_error:rate_limited:429"),
        (529, "jev_error:overloaded:529"),
    ],
)
async def test_request_decision_exhausted_retries_keep_retryable_code(
    status, expected
) -> None:
    """重试用尽后仍要保留限流/过载类别，而不是退化成 unknown。"""
    async with _client(lambda _r: httpx.Response(status, json={"error": "x"})) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(
                client, settings=_settings(max_retries=1), state={}
            )

    assert excinfo.value.code == expected
    assert excinfo.value.status == status
    assert excinfo.value.attempts == 2


async def test_request_decision_transport_error_is_network() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(client, settings=_settings(), state={})

    assert excinfo.value.code == "jev_error:network"
    assert excinfo.value.status is None


async def test_request_decision_timeout_carries_timeout_code() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with _client(handler) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(client, settings=_settings(), state={})

    assert excinfo.value.code == "jev_error:timeout"


async def test_request_decision_budget_exhaustion_keeps_overload_code() -> None:
    """预算先耗尽时，真实死因（529 过载）必须保住，预算只作附加说明。"""
    async with _client(lambda _r: httpx.Response(529, json={})) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(
                client,
                settings=_settings(
                    max_retries=5,
                    total_budget_seconds=1.0,
                    retry_backoff_seconds=5.0,
                ),
                state={},
            )

    assert excinfo.value.code == "jev_error:overloaded:529"
    assert excinfo.value.status == 529
    assert "已达总预算" in str(excinfo.value)
    # attempts 必须记"实际消耗次数"（1），不是配置上限 max_retries+1（6）。
    # 写成上限会让日志宣称"试了 6 次"，与真实请求次数不符。
    assert excinfo.value.attempts == 1


async def test_parse_stage_attempts_counts_real_requests() -> None:
    """解析阶段若发生在第 2 次尝试，attempts 应记 2 而非恒为 1。"""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=_choice_payload("respond", 0.2))

    async with _client(handler) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(
                client,
                settings=_settings(max_retries=2, min_confidence=0.7),
                state={},
            )

    assert excinfo.value.code == "jev_error:low_confidence"
    assert calls["n"] == 2
    assert excinfo.value.attempts == 2


def test_action_text_labels_and_fallback() -> None:
    """动作文案与兜底都会被直接写进日志，必须锁住。"""
    assert action_text(DecisionAction.RESPOND) == "回复"
    assert action_text(DecisionAction.SILENT) == "暂不回复"
    assert action_text("respond") == "回复"
    assert action_text("silent") == "暂不回复"
    # 未收录的值与 None 落到同一个中文兜底，不漏出英文
    assert action_text("bogus") == "未知"
    assert action_text(None) == "未知"


def test_total_budget_default_matches_config() -> None:
    """``config.py`` 是默认值的唯一来源，兜底常量与文档都必须与它一致。

    曾出现过的错：``JevSettings`` 的兜底是 12.0，而 ``config.py`` 的默认值
    是 4.0——按 12.0 调参的用户会被误导到错误的耗时预期。
    """
    from ..config import AgenticChatterConfig

    section = AgenticChatterConfig.JevSection()
    assert section.total_budget_seconds == 4.0
    assert section.timeout_seconds == 8.0


async def test_request_decision_bad_json_code() -> None:
    async with _client(lambda _r: httpx.Response(200, text="not json")) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(client, settings=_settings(), state={})

    assert excinfo.value.code == "jev_error:bad_json"


async def test_request_decision_missing_answer_is_bad_schema() -> None:
    async with _client(
        lambda _r: httpx.Response(200, json={"answers": {"other": {}}})
    ) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(client, settings=_settings(), state={})

    assert excinfo.value.code == "jev_error:bad_schema"


def test_parse_choice_answer_bad_answer_code() -> None:
    with pytest.raises(JevError) as excinfo:
        parse_choice_answer("not-a-dict", min_confidence=0.0)
    assert excinfo.value.code == "jev_error:bad_answer"

    with pytest.raises(JevError) as excinfo:
        parse_choice_answer({"choice": "maybe", "confidence": 0.9}, min_confidence=0.0)
    assert excinfo.value.code == "jev_error:bad_answer"


def test_low_confidence_is_not_classified_as_failure() -> None:
    """置信度低于门限是可调配置，不是调用故障，必须单独成码。"""
    answer = _choice_payload("respond", 0.4)["answers"]["action"]

    with pytest.raises(JevError) as excinfo:
        parse_choice_answer(answer, min_confidence=0.8)

    assert excinfo.value.code == "jev_error:low_confidence"


# --- 端点脱敏 -------------------------------------------------------------
#
# base_url 来自用户配置，自建网关常写成 https://svc:token@gw.internal/...，
# 原样记进日志等于把密码抄给所有能读日志的人。


def test_redact_url_leaves_plain_endpoint_untouched() -> None:
    assert redact_url("https://api.typesafe.ai/v1/systemone") == (
        "https://api.typesafe.ai/v1/systemone"
    )


def test_redact_url_masks_userinfo() -> None:
    assert redact_url("https://svc:secret@gw.internal/v1/systemone") == (
        f"https://{REDACTED}@gw.internal/v1/systemone"
    )


def test_redact_url_masks_user_without_password() -> None:
    assert redact_url("https://svc@gw.internal/v1/systemone") == (
        f"https://{REDACTED}@gw.internal/v1/systemone"
    )


def test_redact_url_masks_userinfo_without_scheme() -> None:
    """省略协议的写法也要能脱敏，否则等于留了个后门。"""
    assert redact_url("svc:secret@gw.internal:8080/v1/systemone") == (
        f"{REDACTED}@gw.internal:8080/v1/systemone"
    )


def test_redact_url_keeps_host_when_password_contains_at_sign() -> None:
    """密码里带 @ 时必须按最后一个 @ 切分，不能把 host 一起吃掉。"""
    assert redact_url("https://svc:p@ss@gw.internal/v1/systemone") == (
        f"https://{REDACTED}@gw.internal/v1/systemone"
    )


@pytest.mark.parametrize(
    ("query", "secret"),
    [
        ("api_key=sk-live-1", "sk-live-1"),
        ("token=abc", "abc"),
        ("access_token=abc", "abc"),
        ("password=abc", "abc"),
    ],
)
def test_redact_url_masks_sensitive_query_params(query, secret) -> None:
    text = redact_url(f"https://gw.internal/v1/systemone?{query}&model=jev-latest")

    assert secret not in text
    assert REDACTED in text
    # 非敏感参数保留，否则运维看不出打的是哪个模型
    assert "model=jev-latest" in text


def test_redact_url_collapses_newline_to_stop_log_forging() -> None:
    """配置里的换行不能把一行日志劈成两行伪造事件。"""
    text = redact_url("https://gw.internal/v1\n[INFO] 伪造的日志行")

    assert "\n" not in text
    assert text.count(" ") <= 4


def test_redact_url_caps_length() -> None:
    text = redact_url("https://gw.internal/" + "a" * 500)

    assert len(text) <= 200
    assert text.endswith("…")


def test_redact_url_masks_sensitive_param_in_fragment() -> None:
    """凭据塞在 fragment 里也要脱敏——只认 ?/& 会漏掉。"""
    text = redact_url("https://gw.internal/v1?a=1#access_token=zzz")

    assert "zzz" not in text
    assert REDACTED in text
    assert "a=1" in text


def test_redact_url_is_idempotent() -> None:
    once = redact_url("https://svc:secret@gw.internal/v1?api_key=sk-live")

    assert redact_url(once) == once


def test_redact_url_handles_blank() -> None:
    assert redact_url("") == ""
    assert redact_url("   ") == ""


def test_jev_error_never_retains_plaintext_credentials() -> None:
    """异常对象本身不能持有明文，否则直接打印异常就会泄漏。"""
    error = JevError(
        "x",
        code="jev_error:network",
        endpoint="https://svc:supersecret@gw.internal/v1/systemone",
    )

    assert "supersecret" not in error.endpoint
    assert error.endpoint == f"https://{REDACTED}@gw.internal/v1/systemone"


def test_format_failure_fields_never_prints_credentials() -> None:
    text = format_failure_fields(
        JevError(
            "x",
            code="jev_error:auth:401",
            attempts=2,
            endpoint="https://svc:supersecret@gw.internal/v1/systemone",
        )
    )

    assert "supersecret" not in text
    assert f"端点 https://{REDACTED}@gw.internal/v1/systemone" in text


async def test_parse_stage_failures_keep_diagnostic_context() -> None:
    """解析阶段的失败也必须带 attempts/status/端点，不能比其它失败类信息更少。"""

    async with _client(
        lambda _r: httpx.Response(200, json=_choice_payload("respond", 0.2))
    ) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(
                client, settings=_settings(min_confidence=0.7), state={}
            )

    assert excinfo.value.code == "jev_error:low_confidence"
    assert excinfo.value.attempts == 1
    assert excinfo.value.status == 200
    assert excinfo.value.endpoint.endswith("/v1/systemone")
    # 原始解析消息不能丢
    assert "置信度低于门限" in str(excinfo.value)


async def test_budget_exhaustion_keeps_upstream_cause() -> None:
    """预算提前收手不能顶替上游的真实死因。

    默认配置（单次 8 秒、总预算 12 秒、重试 2 次）下上游宕机必然先耗尽预算。
    若此时报成 jev_error:budget，运维会被引导去调大预算而不是查上游——
    这正是本次改动要消灭的"日志查不出死因"。
    """

    async def slow_429(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.30)
        return httpx.Response(429, json={"error": "slow down"})

    async with _client(slow_429) as client:
        with pytest.raises(JevError) as excinfo:
            await request_decision(
                client,
                settings=_settings(
                    max_retries=2,
                    total_budget_seconds=0.5,
                    retry_backoff_seconds=0.0,
                ),
                state={},
            )

    # 真实死因是限流，状态码也必须保住
    assert excinfo.value.code == "jev_error:rate_limited:429"
    assert excinfo.value.status == 429
    # 预算作为"提前收手"附加在消息里，不取代死因
    assert "已达总预算" in str(excinfo.value)


def test_format_failure_fields_hides_success_status() -> None:
    """2xx 不是失败信号，写进诊断字段只会让人误以为上游报错。"""
    text = format_failure_fields(
        JevError("x", code="jev_error:bad_schema", status=200, attempts=1)
    )

    assert "HTTP" not in text
    # 其它诊断字段不受影响
    assert "尝试 1 次" in text


def test_endpoint_keeps_query_string_after_path() -> None:
    """带查询串的 base_url 不能把路径补到参数后面。"""
    settings = JevSettings(
        api_key="k", base_url="https://api.typesafe.ai/v1/systemone?key=v"
    )

    assert settings.endpoint == "https://api.typesafe.ai/v1/systemone?key=v"

    settings = JevSettings(api_key="k", base_url="https://gw.internal?key=v")

    assert settings.endpoint == "https://gw.internal/v1/systemone?key=v"


@pytest.mark.parametrize(
    "url",
    [
        "https://g/v1?a=1;api_key=LEAK",
        "https://g/v1?a=1&api%5Fkey=LEAK",
    ],
)
def test_redact_url_covers_odd_parameter_separators(url: str) -> None:
    """分号分隔与百分号编码的键名也要脱敏，否则与工具参数脱敏行为不一致。"""
    assert "LEAK" not in redact_url(url)


def test_reason_text_keeps_unclassified_exception_class() -> None:
    """无法判定类别的异常要保留类名——它往往是定位根因的唯一线索。"""
    from ..decision.models import reason_text

    text = reason_text(classify_failure(ValueError("boom")))

    assert "ValueError" in text
    assert "其他判断信号" not in text


def test_reason_text_still_falls_back_for_unknown_prefix() -> None:
    from ..decision.models import reason_text

    assert reason_text("some_unmapped:ValueError") == "其他判断信号"


def test_classify_failure_passes_through_jev_error_code() -> None:
    error = JevError("x", code="jev_error:auth:401")
    assert classify_failure(error) == "jev_error:auth:401"


def test_classify_failure_keeps_unclassifiable_exception_verbose() -> None:
    """连故障类别都判不出的异常应保留类名，不伪装成已知类别。"""
    assert classify_failure(ValueError("boom")) == "jev_exception:ValueError"


def test_format_failure_fields_omits_empty_diagnostics() -> None:
    assert format_failure_fields(None) == ""
    assert format_failure_fields(JevError("x")) == ""


def test_format_failure_fields_reports_attempts_and_endpoint() -> None:
    text = format_failure_fields(
        JevError(
            "x",
            code="jev_error:auth:401",
            status=401,
            attempts=3,
            elapsed=1.5,
            endpoint="https://api.typesafe.ai/v1/systemone",
        )
    )

    assert "尝试 3 次" in text
    assert "耗时 1.50 秒" in text
    assert "端点 https://api.typesafe.ai/v1/systemone" in text
    # 状态码只出现一次，不因"原因里已含"而重复罗列
    assert text.count("HTTP 401") == 1


# --- 原因码在决策日志里的呈现 --------------------------------------------


def test_describe_decision_renders_jev_failure_code() -> None:
    """带状态码后缀的原因码必须查到中文，不能落成「其他判断信号」。"""
    decision = build_fallback_decision(
        local_score=0.0,
        lower_bound=-0.2,
        upper_bound=0.2,
        reasons=["question_or_request"],
        detail="jev_error:auth:401",
    )

    _action, _source, reasons = describe_decision(decision)

    assert "鉴权失败" in reasons
    assert "其他判断信号" not in reasons


def test_describe_decision_renders_plain_code_without_suffix() -> None:
    decision = build_fallback_decision(
        local_score=0.0,
        lower_bound=-0.2,
        upper_bound=0.2,
        reasons=[],
        detail="jev_disabled",
    )

    _action, _source, reasons = describe_decision(decision)

    assert "Jev 未启用" in reasons


def test_describe_decision_still_falls_back_for_unknown_code() -> None:
    """前缀也匹配不上时才是「其他判断信号」，README 对该症状的说明仍成立。"""
    decision = build_fallback_decision(
        local_score=0.0,
        lower_bound=-0.2,
        upper_bound=0.2,
        reasons=[],
        detail="some_unmapped_reason",
    )

    _action, _source, reasons = describe_decision(decision)

    assert reasons == "其他判断信号"


# --- 失败留痕 -------------------------------------------------------------


@contextmanager
def _capture_chatter_logs(monkeypatch):
    """把 chatter 的日志调用收进列表。

    框架的 ``Logger``（``src/kernel/logger``）是自定义类，既不是标准库
    ``logging.Logger``，也不向 root 传播，因此 ``caplog`` 与 ``addHandler``
    都抓不到记录。这里直接替换实例上的 ``warning`` / ``debug`` 方法，
    只观察调用事实，不依赖日志器的内部实现。
    """
    from .. import chatter as chatter_module

    captured: list[tuple[str, str]] = []
    for level in ("warning", "debug", "info"):
        monkeypatch.setattr(
            chatter_module.logger,
            level,
            lambda message, _level=level: captured.append((_level, message)),
        )
    yield captured


async def test_jev_failure_is_warned_even_when_falling_back(monkeypatch) -> None:
    """默认 sub_actor 回退下也必须 WARNING——否则 Jev 静默失效无人察觉。"""
    with _capture_chatter_logs(monkeypatch) as captured:
        await _run_jev(
            monkeypatch,
            _config(True, "ts_key"),
            handler=lambda _r: httpx.Response(401, json={"error": "unauthorized"}),
        )

    warnings = [text for level, text in captured if level == "warning"]
    assert all("jev_error" not in text for text in warnings), captured
    # 诊断字段随失败一起落盘，运维不必再去翻 traceback
    assert any("尝试 1 次" in text for text in warnings), captured
    assert any("端点 " in text for text in warnings), captured
    # 死因必须是中文，不能把内部原因码原样吐给人看
    assert any("鉴权失败" in text for text in warnings), captured


async def test_jev_disabled_stays_debug_to_avoid_warn_spam(monkeypatch) -> None:
    """未启用不是"请求失败"，按灰区频率打 WARNING 会变成常态噪音。"""
    with _capture_chatter_logs(monkeypatch) as captured:
        await _run_jev(monkeypatch, _config(False, "ts_key"))

    assert not [text for level, text in captured if level == "warning"], captured
    assert any(
        level == "debug" and "Jev 未启用" in text for level, text in captured
    ), captured


async def test_silent_fallback_reason_carries_specific_code(monkeypatch) -> None:
    """silent 模式下决策原因必须是细分后的码，而不是 JevError 类名。"""
    decision = await _run_jev(
        monkeypatch,
        _config(True, "ts_key", fallback_decider="silent"),
        handler=lambda _r: httpx.Response(401, json={"error": "unauthorized"}),
    )

    assert decision is not None
    assert decision.reasons[-1] == "jev_error:auth:401"
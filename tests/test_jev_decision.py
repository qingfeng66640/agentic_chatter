"""Jev 灰区裁决测试。

覆盖四类路径：

- state 构建与截断；
- answer 解析（正常、非法结构、非法选项、置信度门限）；
- HTTP 层的正常应答与失败处理（401/422/429/529/超时/传输错误）；
- ``_decide_gray_zone_with_jev`` 的开关、失败回退与接线行为。

所有 HTTP 交互均通过 httpx MockTransport 拦截，不产生真实网络请求。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from ..decision.jev import (
    JEV_ENDPOINT,
    JevError,
    JevSettings,
    _MessageView,
    build_decision,
    build_fallback_decision,
    build_state,
    parse_choice_answer,
    request_decision,
)
from ..decision.models import DecisionAction, DecisionSource


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
    assert state["bot"] == {"is_chatbot": True}


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
    assert captured["url"] == JEV_ENDPOINT
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

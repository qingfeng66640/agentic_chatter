"""分层回复决策测试。"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.core.models.message import Message
from src.core.models.stream import ChatStream
from src.core.prompt.template import PromptTemplate

from ..chatter import AgenticChatter
from ..decision.actor import _fallback, _parse_decision_result
from ..decision import (
    DecisionAction,
    DecisionFeatures,
    DecisionSource,
    ReplyDecision,
    compute_semantic_relevance,
    cosine_similarity,
    hard_rule_decision,
    score_features,
)
from ..pipeline.stages import (
    STAGE_ACT,
    STAGE_DECIDE,
    STAGE_PERCEIVE,
    STAGE_PLAN,
    STAGE_REFLECT,
    resolve_stage_order,
)
from ..pipeline.state import TurnState
from ..prompts import reply_decision_prompt


async def test_reply_decision_prompt_escapes_json_example() -> None:
    """JSON 示例的花括号不能被模板引擎识别为变量。"""
    rendered = await PromptTemplate(
        name="test_agentic_reply_decision",
        template=reply_decision_prompt,
    ).build()

    assert '"action":"respond|silent"' in rendered
    assert "{{" not in rendered


def test_contextual_fallback_keeps_gray_zone_reachable() -> None:
    """sub_actor 失败时，灰区不能被隐藏阈值再次判为静默。"""
    decision = _fallback(
        "contextual",
        local_score=-0.114,
        reasons=["semantic_unknown"],
    )

    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.FALLBACK


def test_parse_decision_result_accepts_common_wrappers() -> None:
    """兼容模型偶发输出的单元素数组和嵌套 JSON 字符串。"""
    expected = {"action": "respond", "confidence": 0.8}

    assert _parse_decision_result(
        '[{"action":"respond","confidence":0.8}]'
    ) == expected
    assert _parse_decision_result(
        '"{\\"action\\":\\"respond\\",\\"confidence\\":0.8}"'
    ) == expected


def test_parse_decision_result_rejects_non_object() -> None:
    """无法恢复为对象的输出仍应明确失败并走安全回退。"""
    try:
        _parse_decision_result("true")
    except ValueError as exc:
        assert "parsed_type=bool" in str(exc)
    else:
        raise AssertionError("布尔结果不应被接受")


def _message(
    text: str,
    *,
    message_id: str = "m1",
    sender_id: str = "u1",
    reply_to: str | None = None,
    **extra: object,
) -> Message:
    """构造测试消息。"""
    return Message(
        message_id=message_id,
        content=text,
        processed_plain_text=text,
        sender_id=sender_id,
        reply_to=reply_to,
        chat_type="group",
        **extra,
    )


def _decision_config(**overrides: object) -> SimpleNamespace:
    """构造评分所需配置。"""
    values: dict[str, object] = {
        "weight_direct_address": 0.30,
        "weight_semantic_continuity": 0.20,
        "weight_bot_history_continuity": 0.14,
        "weight_participation_momentum": 0.12,
        "weight_question_or_request": 0.10,
        "weight_contribution_value": 0.06,
        "weight_directed_elsewhere": 0.24,
        "weight_interruption_cost": 0.18,
        "weight_topic_closure": 0.16,
        "weight_rhythm_cooldown": 0.12,
        "weight_silence_momentum": 0.08,
        "weight_low_information": 0.06,
        "gray_zone_randomness": 0.0,
        "base_uncertainty": 0.10,
        "local_reply_lower_bound": 0.35,
        "local_silent_upper_bound": -0.35,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_private_message_is_hard_reply() -> None:
    """有效私聊不应消耗 embedding 或 sub_actor。"""
    decision = hard_rule_decision(
        is_private=True,
        bot_id="bot",
        bot_nickname="狐狐",
        unread_messages=[_message("在吗")],
        bot_message_ids=set(),
    )

    assert decision is not None
    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.HARD_RULE


def test_reply_to_bot_is_hard_reply() -> None:
    """平台可靠 reply-to-bot 应直接回复。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot",
        bot_nickname="狐狐",
        unread_messages=[_message("然后呢", reply_to="bot-msg")],
        bot_message_ids={"bot-msg"},
    )

    assert decision is not None
    assert decision.should_respond
    assert decision.reasons == ["reply_to_bot"]


def test_bot_mention_is_hard_reply() -> None:
    """结构化 @ bot 应直接回复。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot",
        bot_nickname="狐狐",
        unread_messages=[
            _message(
                "看看这个",
                at_users=[{"user_id": "bot", "nickname": "狐狐"}],
            )
        ],
        bot_message_ids=set(),
    )

    assert decision is not None
    assert decision.reasons == ["mention_bot"]


def test_empty_messages_are_hard_silent() -> None:
    """无有效正文的批次应静默。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot",
        bot_nickname="狐狐",
        unread_messages=[_message("   ")],
        bot_message_ids=set(),
    )

    assert decision is not None
    assert decision.action == DecisionAction.SILENT


def test_score_uses_confidence_interval() -> None:
    """只有整个区间越过边界才本地直通。"""
    strong = DecisionFeatures(
        direct_address=1.0,
        semantic_continuity=1.0,
        bot_history_continuity=1.0,
        question_or_request=1.0,
    )
    weak = DecisionFeatures(question_or_request=1.0, uncertainty=0.25)

    strong_decision = score_features(strong, _decision_config())
    weak_decision = score_features(weak, _decision_config())

    assert strong_decision is not None
    assert strong_decision.action == DecisionAction.RESPOND
    assert weak_decision is None


def test_score_is_reproducible_with_fixed_rng() -> None:
    """固定 RNG 时灰区随机扰动应可复现。"""
    config = _decision_config(gray_zone_randomness=0.03)
    features = DecisionFeatures(question_or_request=1.0)

    first = score_features(features, config, rng=random.Random(7))
    second = score_features(features, config, rng=random.Random(7))

    assert first == second


def test_cosine_similarity_handles_invalid_vectors() -> None:
    """非法或零向量不应伪造低相关。"""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([], []) is None
    assert cosine_similarity([0.0], [0.0]) is None
    assert cosine_similarity([1.0], [1.0, 2.0]) is None


async def test_embedding_uses_one_batched_request(monkeypatch) -> None:
    """一轮相关性判断只发一个有界批量 embedding 请求。"""
    captured: list[str] = []

    class Request:
        async def send(self):
            return SimpleNamespace(
                embeddings=[
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )

    def create_request(_model_set, *, request_name: str, inputs: list[str]):
        assert request_name == "agentic_reply_relevance"
        captured.extend(inputs)
        return Request()

    monkeypatch.setattr(
        "plugins.agentic_chatter.decision.semantic.llm_api.get_model_set_by_task",
        lambda _name: [{}],
    )
    monkeypatch.setattr(
        "plugins.agentic_chatter.decision.semantic.llm_api.create_embedding_request",
        create_request,
    )

    topic, bot = await compute_semantic_relevance(
        current_text="继续聊部署",
        topic_candidates=["部署问题"],
        bot_candidates=["昨天说的测试"],
        model_task="embedding",
        candidate_limit=1,
        max_chars_per_candidate=40,
    )

    assert len(captured) == 3
    assert topic == 1.0
    assert bot == 0.0


def test_decide_stage_is_before_plan_and_act() -> None:
    """用户误配顺序时也不能在 decide 前行动。"""
    order = resolve_stage_order(
        [STAGE_PLAN, STAGE_ACT, STAGE_PERCEIVE, STAGE_REFLECT],
        enable_perceive=True,
        enable_plan=True,
        enable_reflect=True,
        enable_decide=True,
    )

    assert order.index(STAGE_DECIDE) < order.index(STAGE_PLAN)
    assert order.index(STAGE_DECIDE) < order.index(STAGE_ACT)


async def test_silent_decision_skips_plan_and_act_but_reflects() -> None:
    """成功静默应继续 reflect，而不是进入失败重试或 Stop。"""
    chatter = AgenticChatter(stream_id="stream", plugin=object())
    chatter._stage_perceive = AsyncMock()
    chatter._stage_decide = AsyncMock()
    chatter._stage_plan = AsyncMock()
    chatter._stage_act = AsyncMock()
    chatter._stage_reflect = AsyncMock()

    async def decide(_config, _stream, state: TurnState, _messages) -> None:
        state.decision = ReplyDecision(
            DecisionAction.SILENT,
            DecisionSource.LOCAL,
        )

    chatter._stage_decide.side_effect = decide
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            stage_order=[STAGE_DECIDE, STAGE_PLAN, STAGE_ACT, STAGE_REFLECT],
            enable_perceive=False,
            enable_plan=True,
            enable_reflect=True,
        ),
        decision=SimpleNamespace(enabled=True),
    )
    state = TurnState(stream_id="stream", unread_texts="群聊消息")

    outcome = await chatter._run_pipeline(
        config=config,
        chat_stream=ChatStream("stream", chat_type="group"),
        state=state,
        unread_msgs=[_message("群聊消息")],
    )

    chatter._stage_plan.assert_not_awaited()
    chatter._stage_act.assert_not_awaited()
    chatter._stage_reflect.assert_awaited_once()
    assert outcome.should_wait
    assert not outcome.should_stop
    assert not state.failed

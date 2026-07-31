"""分层回复决策测试。"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.core.models.message import Message
from src.core.models.stream import ChatStream
from src.core.prompt.template import PromptTemplate

from ..chatter import AgenticChatter
from ..decision.actor import (
    _fallback,
    _parse_decision_result,
    _response_diagnostics,
)
from ..decision import (
    DecisionAction,
    DecisionFeatures,
    DecisionSource,
    ReplyDecision,
    compute_semantic_relevance,
    cosine_similarity,
    extract_features,
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
    """没有近期成功回复时，contextual fallback 仍允许进入回复流程。"""
    decision = _fallback(
        "contextual",
        local_score=-0.114,
        reasons=["semantic_unknown"],
    )

    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.FALLBACK



def test_nickname_address_enters_local_scoring() -> None:
    """昵称文本不再绕过评分直接唤醒。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot-id",
        bot_nickname="小蝶",
        unread_messages=[_message("小蝶，帮我看看")],
        bot_message_ids=set(),
    )

    assert decision is None


def test_nickname_request_is_locally_allowed() -> None:
    """昵称加问题或请求时由本地组合规则回复。"""
    features = extract_features(
        unread_messages=[_message("小蝶，帮我看看这个问题")],
        bot_id="bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    decision = score_features(features, _decision_config())

    assert features.direct_address == 1.0
    assert features.question_or_request == 1.0
    assert decision is not None
    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.LOCAL
    assert "contextual_nickname_address" in decision.reasons


def test_standalone_nickname_is_locally_allowed() -> None:
    """独立昵称召唤不应因低信息信号进入灰区。"""
    features = extract_features(
        unread_messages=[_message("小蝶")],
        bot_id="bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    decision = score_features(features, _decision_config())

    assert features.direct_address == 1.0
    assert features.low_information == 1.0
    assert decision is not None
    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.LOCAL


def test_nickname_punctuation_is_locally_allowed() -> None:
    """独立昵称加标点也应视为召唤。"""
    features = extract_features(
        unread_messages=[_message("小蝶！")],
        bot_id="bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    decision = score_features(features, _decision_config())

    assert decision is not None
    assert decision.action == DecisionAction.RESPOND


def test_nickname_local_reply_respects_negative_guards() -> None:
    """明确负向上下文不能被昵称召唤组合直接放行。"""
    for feature in (
        DecisionFeatures(direct_address=1.0, directed_elsewhere=1.0),
        DecisionFeatures(direct_address=1.0, topic_closure=1.0),
        DecisionFeatures(direct_address=1.0, rhythm_cooldown=1.0),
        DecisionFeatures(direct_address=1.0, interruption_cost=1.0),
    ):
        decision = score_features(feature, _decision_config())
        assert decision is None or decision.action == DecisionAction.SILENT


def test_nickname_alone_does_not_use_hard_rule() -> None:
    """单独昵称仍不绕过硬规则层。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot-id",
        bot_nickname="小蝶",
        unread_messages=[_message("小蝶")],
        bot_message_ids=set(),
    )

    assert decision is None


def test_personality_alias_is_direct_address() -> None:
    """人格配置别名应与聊天流显示名同样参与本地称呼判断。"""
    features = extract_features(
        unread_messages=[_message("小蝶！")],
        bot_id="bot-id",
        bot_nickname=("我是打工蝶😭", "小蝶", "蝶宝"),
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    decision = score_features(features, _decision_config())

    assert features.direct_address == 1.0
    assert decision is not None
    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.LOCAL


def test_discussing_nickname_is_not_direct_address() -> None:
    """正文讨论 Bot 名称时不能获得直接称呼信号。"""
    features = extract_features(
        unread_messages=[_message("我们正在讨论小蝶这个角色")],
        bot_id="bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    assert features.direct_address == 0.0
    assert "nickname_address" not in features.reasons


def test_nickname_prefix_inside_a_word_is_not_direct_address() -> None:
    """昵称只是词语前缀时不能被视为直接称呼。"""
    features = extract_features(
        unread_messages=[_message("小蝶结这个道具怎么获得")],
        bot_id="bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=0.0,
        bot_history_continuity=0.0,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    assert features.direct_address == 0.0


def test_other_member_mention_does_not_trigger_hard_reply() -> None:
    """提及其他成员不能被误判为 bot 被唤醒。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="bot-id",
        bot_nickname="小蝶",
        unread_messages=[_message("@我一下", at_users=["other-id"])],
        bot_message_ids=set(),
    )

    assert decision is None


def test_contextual_fallback_suppresses_recent_reply() -> None:
    """近期已成功回复时，contextual fallback 不能再次自动介入。"""
    decision = _fallback(
        "contextual",
        local_score=0.104,
        reasons=["semantic_unknown"],
        suppress_contextual=True,
    )

    assert decision.action == DecisionAction.SILENT
    assert "recent_reply_fallback_suppressed" in decision.reasons


def test_platform_mention_signal_triggers_hard_reply_without_ids() -> None:
    """平台确认 @Bot 时不依赖稳定账号或结构化提及列表。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="",
        bot_nickname="小蝶",
        unread_messages=[_message("@<小蝶：群内身份> @我一下", bot_was_mentioned=True, at_users=[])],
        bot_message_ids=set(),
    )

    assert decision is not None
    assert decision.action == DecisionAction.RESPOND
    assert decision.source == DecisionSource.HARD_RULE
    assert decision.reasons == ["mention_bot"]


def test_platform_mention_signal_overrides_other_member_mentions() -> None:
    """平台确认 @Bot 时，其他成员提及不能标记为面向他人。"""
    message = _message(
        "@<小蝶：群内身份> @我一下",
        bot_was_mentioned=True,
        at_users=[{"user_id": "other-member"}],
    )
    decision = hard_rule_decision(
        is_private=False,
        bot_id="stable-bot-id",
        bot_nickname="小蝶",
        unread_messages=[message],
        bot_message_ids=set(),
    )
    features = extract_features(
        unread_messages=[message],
        bot_id="stable-bot-id",
        bot_nickname="小蝶",
        bot_message_ids=set(),
        participation=SimpleNamespace(last_reply_at=0.0, consecutive_silence=0),
        semantic_continuity=None,
        bot_history_continuity=None,
        participation_window_seconds=60.0,
        rhythm_cooldown_seconds=30.0,
        now=100.0,
    )

    assert decision is not None
    assert decision.reasons == ["mention_bot"]
    assert features.directed_elsewhere == 0.0


def test_platform_false_keeps_legacy_exact_mention_compatibility() -> None:
    """False 不能否定旧 Adapter 的精确结构化提及。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="stable-bot-id",
        bot_nickname="小蝶",
        unread_messages=[_message("你好", bot_was_mentioned=False, at_users=[{"user_id": "stable-bot-id"}])],
        bot_message_ids=set(),
    )

    assert decision is not None
    assert decision.reasons == ["mention_bot"]


def test_non_boolean_platform_mention_signal_does_not_trigger_reply() -> None:
    """字符串等不可信值不能被当成平台确认。"""
    decision = hard_rule_decision(
        is_private=False,
        bot_id="",
        bot_nickname="小蝶",
        unread_messages=[_message("普通消息", bot_was_mentioned="false")],
        bot_message_ids=set(),
    )

    assert decision is None


def test_explicit_fallback_modes_ignore_recent_reply_suppression() -> None:
    """只有 contextual 受近期回复保护，显式模式保持原语义。"""
    assert _fallback(
        "fail_open",
        local_score=0.0,
        reasons=[],
        suppress_contextual=True,
    ).action == DecisionAction.RESPOND
    assert _fallback(
        "fail_closed",
        local_score=0.0,
        reasons=[],
        suppress_contextual=True,
    ).action == DecisionAction.SILENT


def test_response_diagnostics_describes_empty_text_without_leaking_content() -> None:
    """空决策输出日志只记录长度和响应状态，不记录正文。"""
    response = SimpleNamespace(
        message="",
        reasoning_content="隐私推理内容",
        call_list=[object()],
        stop_reason="stop",
    )

    assert _response_diagnostics(response) == (
        "正文长度=0，推理内容长度=6，工具调用数=1，结束原因=stop"
    )


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


def test_local_combinations_handle_clear_reply_and_silence() -> None:
    """明显的本地组合应绕过子决策模型。"""
    closure = score_features(
        DecisionFeatures(topic_closure=1.0, low_information=1.0),
        _decision_config(),
    )
    directed_other = score_features(
        DecisionFeatures(directed_elsewhere=1.0),
        _decision_config(),
    )
    followup_question = score_features(
        DecisionFeatures(question_or_request=1.0, semantic_continuity=0.60),
        _decision_config(),
    )
    strong_followup = score_features(
        DecisionFeatures(
            semantic_continuity=0.75,
            bot_history_continuity=0.55,
            contribution_value=0.25,
        ),
        _decision_config(),
    )

    assert closure is not None
    assert closure.action == DecisionAction.SILENT
    assert closure.reasons[0] == "closure_low_information"
    assert directed_other is not None
    assert directed_other.action == DecisionAction.SILENT
    assert directed_other.reasons[0] == "directed_elsewhere_unrelated"
    assert followup_question is not None
    assert followup_question.action == DecisionAction.RESPOND
    assert followup_question.reasons[0] == "contextual_question_for_bot"
    assert strong_followup is not None
    assert strong_followup.action == DecisionAction.RESPOND
    assert strong_followup.reasons[0] == "strong_contextual_followup"


def test_local_combinations_preserve_ambiguous_messages_for_sub_actor() -> None:
    """泛问题或只与当前话题相关的消息仍需交给子决策模型。"""
    generic_question = score_features(
        DecisionFeatures(question_or_request=1.0),
        _decision_config(),
    )
    topic_only = score_features(
        DecisionFeatures(semantic_continuity=0.75, contribution_value=0.25),
        _decision_config(),
    )

    assert generic_question is None
    assert topic_only is None


def test_local_reply_combination_respects_negative_guards() -> None:
    """面向他人、冷却、收尾或多人快聊不能被连续性强行放行。"""
    for feature in (
        DecisionFeatures(
            question_or_request=1.0,
            semantic_continuity=0.9,
            directed_elsewhere=1.0,
        ),
        DecisionFeatures(
            question_or_request=1.0,
            semantic_continuity=0.9,
            rhythm_cooldown=1.0,
        ),
        DecisionFeatures(
            question_or_request=1.0,
            semantic_continuity=0.9,
            interruption_cost=1.0,
        ),
    ):
        decision = score_features(feature, _decision_config())
        assert decision is None or decision.action == DecisionAction.SILENT

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

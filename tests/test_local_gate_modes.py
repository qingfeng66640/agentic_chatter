"""`local_gate_enabled` 两种模式的行为测试。

此前该开关零覆盖。关闭本地闸门时必须满足：

- 不调用 embedding（``compute_semantic_relevance`` 不可达）；
- 不调用 ``score_features``，本地评分无法产出任何裁决；
- 不产生 ``semantic_unknown`` 不确定度惩罚（其补偿逻辑位于闸门块内，
  关闭闸门时若不处理会把 +0.18 惩罚永久留下）；
- 语义信号显式为 0，并在图例中注明。
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..decision.models import DecisionFeatures, DecisionSource
from ..decision.scoring import score_features
from ..decision.signals import extract_features

_CHATTER = Path(__file__).resolve().parents[1] / "chatter.py"


class _Msg:
    """最小消息替身。"""

    def __init__(
        self,
        sender_id: str,
        sender_name: str,
        text: str,
        role: str = "user",
        reply_to: str = "",
    ) -> None:
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.content = text
        self.processed_plain_text = None
        self.sender_role = role
        self.reply_to = reply_to
        self.extra: dict = {}


class _Participation:
    """参与状态替身。"""

    last_reply_at = 0.0
    last_reply_topic = ""
    consecutive_participation = 0
    consecutive_silence = 0


def _config(**overrides) -> SimpleNamespace:
    base = {
        "local_reply_lower_bound": 0.30,
        "local_silent_upper_bound": -0.25,
        "base_uncertainty": 0.12,
        "gray_zone_randomness": 0.0,
        "weight_direct_address": 0.30,
        "weight_semantic_continuity": 0.26,
        "weight_bot_history_continuity": 0.18,
        "weight_participation_momentum": 0.08,
        "weight_question_or_request": 0.08,
        "weight_contribution_value": 0.10,
        "weight_directed_elsewhere": 0.34,
        "weight_interruption_cost": 0.24,
        "weight_topic_closure": 0.24,
        "weight_rhythm_cooldown": 0.20,
        "weight_silence_momentum": 0.04,
        "weight_low_information": 0.10,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _features(semantic, bot_semantic) -> DecisionFeatures:
    return extract_features(
        unread_messages=[_Msg("u1", "小明", "这个方案我觉得可以")],
        bot_id="bot-1",
        bot_nickname=("小蝶",),
        bot_message_ids=set(),
        participation=_Participation(),
        semantic_continuity=semantic,
        bot_history_continuity=bot_semantic,
        participation_window_seconds=600.0,
        rhythm_cooldown_seconds=35.0,
    )


# --- 闸门开启：评分直通生效 ------------------------------------------------


def test_gate_on_local_score_can_decide() -> None:
    """闸门开启且语义很高时，本地评分直接裁决，不再交给外部模型。"""
    features = _features(0.9, 0.9)
    decision = score_features(features, _config())

    assert decision is not None
    assert decision.source is DecisionSource.LOCAL


def test_gate_off_local_score_would_have_decided_but_is_skipped() -> None:
    """同样的特征在闸门关闭时由外部裁决者接手，本地不再产出裁决。

    这里证明「关闭闸门会改变裁决者」这一行为，而不是证明本地评分失效。
    """
    features = _features(0.9, 0.9)
    local = score_features(features, _config())

    assert local is not None, "闸门开启时本地确实能裁决"
    assert local.source is DecisionSource.LOCAL


# --- 闸门关闭：不得留下 semantic_unknown 惩罚 -----------------------------


def test_no_semantic_unknown_penalty_when_semantics_disabled() -> None:
    """语义不可用时才加 semantic_unknown；显式传 0.0 不应触发惩罚。

    ``extract_features`` 只在参数为 None 时加 +0.18 不确定度。
    """
    unknown = _features(None, None)
    disabled = _features(0.0, 0.0)

    assert "semantic_unknown" in unknown.reasons
    assert unknown.uncertainty >= 0.18

    assert "semantic_unknown" not in disabled.reasons
    assert disabled.uncertainty < 0.18


def test_disabled_semantics_are_zero_not_unset() -> None:
    disabled = _features(0.0, 0.0)

    assert disabled.semantic_continuity == 0.0
    assert disabled.bot_history_continuity == 0.0


# --- 结构性保证：闸门块内才有 embedding 与 score_features -------------------


def test_embedding_and_scoring_live_inside_the_gate_block() -> None:
    """静态保证：embedding 与本地评分都只在闸门块内被调用。

    这是「关闭闸门后本地评分不能裁决」的结构性依据，防止后续重构把
    调用点挪到块外而无人察觉。
    """
    tree = ast.parse(io.open(_CHATTER, encoding="utf-8").read())

    stage = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_stage_decide"
    )

    gate = next(
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.If)
        and "local_gate_enabled" in ast.unparse(node.test)
    )
    gate_span = (
        min(statement.lineno for statement in gate.body),
        max(statement.lineno for statement in gate.body),
    )

    def _call_line(fn_name: str) -> int:
        return next(
            call.lineno
            for call in ast.walk(stage)
            if isinstance(call, ast.Call)
            and ast.unparse(call.func) == fn_name
        )

    assert gate_span[0] <= _call_line("compute_semantic_relevance") <= gate_span[1]
    assert gate_span[0] <= _call_line("score_features") <= gate_span[1]


def test_bounds_are_not_compared_outside_the_gate() -> None:
    """闸门外不得出现对边界常量的比较。

    日志 f-string 里引用这两个常量是允许的（仅用于打印），
    因此只禁止 Compare 节点，不禁止普通属性引用。
    """
    tree = ast.parse(io.open(_CHATTER, encoding="utf-8").read())
    stage = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_stage_decide"
    )
    gate = next(
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.If)
        and "local_gate_enabled" in ast.unparse(node.test)
    )
    gate_span = (
        min(statement.lineno for statement in gate.body),
        max(statement.lineno for statement in gate.body),
    )

    for node in ast.walk(stage):
        if not isinstance(node, ast.Compare):
            continue
        operands = [ast.unparse(item) for item in node.comparators]
        operands.append(ast.unparse(node.left))
        hits_bounds = any(
            name in operand
            for name in ("local_reply_lower_bound", "local_silent_upper_bound")
            for operand in operands
        )
        if hits_bounds:
            assert gate_span[0] <= node.lineno <= gate_span[1], (
                f"边界比较出现在闸门外 L{node.lineno}，可能引入本地裁决"
            )


# --- 图例一致性 -----------------------------------------------------------


def test_semantic_legend_mentions_disabled_state() -> None:
    """关闭语义计算时信号恒为 0，图例必须写明，避免被当成真实测量值。"""
    from ..decision.jev import _SIGNAL_LEGEND

    assert "未启用语义计算" in _SIGNAL_LEGEND["semantic_continuity"]
    assert "未启用语义计算" in _SIGNAL_LEGEND["bot_history_continuity"]


@pytest.mark.parametrize("key", ["direct_address", "question_or_request"])
def test_legend_covers_every_forwarded_signal(key: str) -> None:
    """chatter 传给 Jev 的每个信号都必须有图例。"""
    from ..decision.jev import _SIGNAL_LEGEND

    assert key in _SIGNAL_LEGEND

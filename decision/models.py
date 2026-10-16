"""回复决策领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DecisionAction(StrEnum):
    """回复决策动作。"""

    RESPOND = "respond"
    SILENT = "silent"


class DecisionSource(StrEnum):
    """回复决策来源。"""

    DISABLED = "disabled"
    HARD_RULE = "hard_rule"
    LOCAL = "local"
    SUB_ACTOR = "sub_actor"
    JEV = "jev"
    FALLBACK = "fallback"


@dataclass(slots=True)
class DecisionFeatures:
    """本轮本地决策信号。"""

    direct_address: float = 0.0
    semantic_continuity: float = 0.0
    bot_history_continuity: float = 0.0
    participation_momentum: float = 0.0
    question_or_request: float = 0.0
    contribution_value: float = 0.0
    directed_elsewhere: float = 0.0
    interruption_cost: float = 0.0
    topic_closure: float = 0.0
    rhythm_cooldown: float = 0.0
    silence_momentum: float = 0.0
    low_information: float = 0.0
    uncertainty: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReplyDecision:
    """单轮是否回复的完整决策。"""

    action: DecisionAction
    source: DecisionSource
    score: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    sub_actor_used: bool = False

    @property
    def should_respond(self) -> bool:
        """返回是否应进入回复流程。"""
        return self.action == DecisionAction.RESPOND


_ACTION_TEXT = {
    DecisionAction.RESPOND: "回复",
    DecisionAction.SILENT: "暂不回复",
}
# 子决策 JSON 拦截日志拿到的是原始 action 字符串而非枚举，单独建一份按值索引
_ACTION_TEXT_BY_VALUE = {
    "respond": "回复",
    "silent": "暂不回复",
}
_SOURCE_TEXT = {
    DecisionSource.DISABLED: "已关闭判断",
    DecisionSource.HARD_RULE: "直接唤醒",
    DecisionSource.LOCAL: "本地判断",
    DecisionSource.SUB_ACTOR: "子决策模型",
    DecisionSource.JEV: "Jev 决策",
    DecisionSource.FALLBACK: "故障回退",
}
_REASON_TEXT = {
    "decision_disabled": "已关闭自然参与判断",
    "empty_messages": "没有可处理的消息",
    "private_chat": "私聊消息",
    "reply_to_bot": "回复了我的消息",
    "mention_bot": "@了我",
    "nickname_address": "称呼了我",
    "distracted": "开启了偶尔走神",
    "question_or_request": "包含问题或请求",
    "directed_elsewhere": "消息主要面向其他成员",
    "multi_party_fast_flow": "多人连续对话中",
    "topic_closure": "话题已经收尾",
    "rhythm_cooldown": "我刚回复过，暂缓插话",
    "low_information": "消息信息较少",
    "semantic_unknown": "语义相关性暂不可用",
    "conflicting_signals": "对话信号存在冲突",
    "recent_reply_fallback_suppressed": "刚回复过，避免故障回退时重复插话",
    "sub_actor_fallback": "子决策模型不可用",
    "closure_low_information": "话题收尾且没有新信息",
    "directed_elsewhere_unrelated": "消息面向其他成员且与我无关",
    "fast_flow_no_clear_entry": "多人快速对话中没有明确介入入口",
    "cooldown_low_value": "刚回复过且当前消息价值较低",
    "contextual_question_for_bot": "问题与当前话题或我的发言连续",
    "contextual_nickname_address": "明确称呼了我，且当前没有明显冲突信号",
    "strong_contextual_followup": "高连续性且内容具有可贡献信息",
    "local_score_reply": "本地评分达到回复边界",
    "local_score_silent": "本地评分达到静默边界",
    "jev_decided": "Jev 裁决",
    "jev_disabled": "Jev 未启用",
    "jev_api_key_missing": "Jev 未配置 API Key",
    "jev_error": "Jev 调用失败",
    # Jev 失败原因码。码尾可能再带一段原始状态码（如 jev_error:auth:401），
    # describe_decision 会退回基础码匹配，因此这里只登记基础码即可。
    "jev_error:budget": "Jev 裁决超出总预算",
    "jev_error:timeout": "Jev 请求超时",
    "jev_error:network": "Jev 网络不可达",
    "jev_error:rate_limited": "Jev 被限流",
    "jev_error:overloaded": "Jev 服务过载",
    "jev_error:auth": "Jev 鉴权失败",
    "jev_error:payment": "Jev 额度不足",
    "jev_error:not_found": "Jev 端点或模型不存在",
    "jev_error:bad_request": "Jev 拒绝了请求",
    "jev_error:http": "Jev 返回异常状态码",
    "jev_error:bad_json": "Jev 响应不是合法 JSON",
    "jev_error:bad_schema": "Jev 响应结构非法",
    "jev_error:bad_answer": "Jev 答案结构非法",
    "jev_error:low_confidence": "Jev 置信度低于门限",
    "jev_error:unknown": "Jev 调用失败（原因未分类）",
}


def reason_text(code: str) -> str:
    """把原因码映射为中文说明。

    原因码允许携带细节后缀（``jev_error:auth:401``），因此先整串匹配，
    再从右往左逐段剥离后缀去命中已登记的基础码：
    ``jev_error:auth:401`` → ``jev_error:auth`` → ``jev_error``。

    只做整串匹配会让带后缀的码一律落成「其他判断信号」；只按第一个冒号
    切分又会退到 ``jev_error`` 这条最粗的映射，等于把刚细分出来的死因
    又糊回去——两种简化都会让原因细分白做。

    ``jev_exception:<类名>`` 表示异常连故障类别都判不出。此处保留类名而不
    落成「其他判断信号」：类名往往就是定位根因的唯一线索，抹掉它等于让
    这类失败彻底无从查起。

    Args:
        code: 决策原因码。

    Returns:
        str: 中文说明；无法映射时返回「其他判断信号」。
    """
    text = _REASON_TEXT.get(code)
    if text is not None:
        return text
    base = code
    while ":" in base:
        base = base.rsplit(":", 1)[0]
        text = _REASON_TEXT.get(base)
        if text is not None:
            return text
    if code.startswith("jev_exception:") and len(code) > len("jev_exception:"):
        return f"Jev 调用失败（{code[len('jev_exception:'):]}）"
    return "其他判断信号"


def action_text(action: DecisionAction | str | None) -> str:
    """返回动作对应的中文说明。

    既接受 :class:`DecisionAction` 枚举（裁决结果），也接受原始取值字符串
    （子决策 JSON 拦截日志里的 ``action`` 字段），统一以中文呈现。

    Args:
        action: 动作枚举或 ``"respond"`` / ``"silent"`` 字符串。

    Returns:
        str: 中文说明；无法识别时返回「未知」。
    """
    if action is None:
        return "未知"
    if isinstance(action, DecisionAction):
        return _ACTION_TEXT.get(action, "未知")
    return _ACTION_TEXT_BY_VALUE.get(str(action).strip().lower(), "未知")


def describe_decision(decision: ReplyDecision) -> tuple[str, str, str]:
    """将内部决策枚举与原因转换为中文运行日志文本。"""
    action = _ACTION_TEXT.get(decision.action, "未知")
    source = _SOURCE_TEXT.get(decision.source, "未知方式")
    reasons = [reason_text(reason) for reason in decision.reasons[-2:]]
    return action, source, "、".join(reasons) or "未提供原因"

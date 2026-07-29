"""Agent 主循环。

本模块实现「说话 → 做事 → 观察 → 继续」的 agent 循环，这是本插件
与 DFC 最本质的差异所在。

DFC 的问题（``session.py:729-750``）：一旦某轮的 tool call 全部是
action（也就是模型说完话了），就立刻注入 ``__SUSPEND__`` 并 yield
Wait 结束本轮。模型永远没有机会在说完话之后再想想「要不要顺手查
一下、记一下」。这是工具调用率低的最直接原因。

本模块的循环不做这个截断：说完话之后仍然继续迭代，只有模型显式
调用 ``end_turn`` / ``stop_conversation``，或达到 ``max_iterations``
上限时才结束。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from src.kernel.llm import LLMPayload, ROLE, Text, ToolResult

from ..actions.control import DEFAULT_STOP_MINUTES, MAX_STOP_MINUTES
from ..humanize.segmenter import Segment, segment_reply
from .state import TurnState

# 控制流动作名，这些调用不进入普通工具执行路径
END_TURN_CALL = "action-end_turn"
STOP_CALL = "action-stop_conversation"


class SpeakFn(Protocol):
    """发送一条消息的回调协议。"""

    async def __call__(self, text: str) -> bool:
        """发送文本。

        Args:
            text: 要发送的文本。

        Returns:
            bool: 是否发送成功。
        """
        ...


async def deliver_segments(
    segments: list[Segment],
    speak: SpeakFn,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> int:
    """按打字节奏依次发送各个分段。

    Args:
        segments: 待发送的分段列表。
        speak: 实际发送单条消息的回调。
        sleep: 延迟实现；为 None 时使用 ``asyncio.sleep``。

    Returns:
        int: 成功发送的分段数量。
    """
    if not segments:
        return 0

    waiter = sleep or asyncio.sleep
    sent = 0

    for segment in segments:
        if segment.delay > 0:
            await waiter(segment.delay)
        if await speak(segment.text):
            sent += 1

    return sent


def classify_calls(calls: list[Any]) -> tuple[list[Any], float | None, float | None]:
    """将本轮的 tool call 拆分为普通调用与控制流调用。

    Args:
        calls: LLM 返回的 tool call 列表。

    Returns:
        tuple: ``(普通调用列表, end_turn 秒数, stop 分钟数)``。
            后两者为 None 表示模型未请求对应的控制流。
    """
    normal: list[Any] = []
    end_seconds: float | None = None
    stop_minutes: float | None = None

    for call in calls or []:
        name = str(getattr(call, "name", "") or "")
        args = getattr(call, "args", None)
        args = args if isinstance(args, dict) else {}

        if name == END_TURN_CALL:
            try:
                end_seconds = max(0.0, float(args.get("seconds", 0.0) or 0.0))
            except (TypeError, ValueError):
                end_seconds = 0.0
            continue

        if name == STOP_CALL:
            try:
                stop_minutes = max(
                    0.0,
                    min(
                        MAX_STOP_MINUTES,
                        float(args.get("minutes", DEFAULT_STOP_MINUTES) or DEFAULT_STOP_MINUTES),
                    ),
                )
            except (TypeError, ValueError):
                stop_minutes = DEFAULT_STOP_MINUTES
            continue

        normal.append(call)

    return normal, end_seconds, stop_minutes


def should_continue_loop(state: TurnState, max_iterations: int) -> bool:
    """判断 agent 循环是否应当继续下一次迭代。

    这是与 DFC 行为差异的核心：即使本轮已经说过话（``state.spoke``
    为 True），只要模型没有显式结束，循环仍然继续。模型因此可以在
    说完话之后继续调用工具。

    Args:
        state: 当前回合状态。
        max_iterations: 最大迭代次数上限。

    Returns:
        bool: 是否继续循环。
    """
    if state.end_turn_requested or state.stop_requested:
        return False
    if max_iterations > 0 and state.iterations >= max_iterations:
        return False
    return True


def build_speak_segments(message: str | None, humanize_config: Any) -> list[Segment]:
    """将模型的文本输出转换为待发送的分段。

    Args:
        message: 模型本轮的文本输出。
        humanize_config: 拟人化配置段，需提供分段相关字段。

    Returns:
        list[Segment]: 分段列表；无有效文本时返回空列表。
    """
    return segment_reply(
        message or "",
        enabled=bool(getattr(humanize_config, "enable_segmentation", True)),
        max_segment_chars=int(getattr(humanize_config, "max_segment_chars", 60)),
        max_segments=int(getattr(humanize_config, "max_segments", 4)),
        typing_cps=float(getattr(humanize_config, "typing_cps", 8.0)),
        max_typing_delay=float(getattr(humanize_config, "max_typing_delay", 4.0)),
    )


def append_tool_result(response: Any, call: Any, value: str) -> None:
    """向响应链追加一条工具结果。

    Args:
        response: 当前 LLM 响应对象。
        call: 对应的 tool call。
        value: 结果文本。
    """
    response.add_payload(
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(  # type: ignore[arg-type]
                value=value,
                call_id=getattr(call, "id", None),
                name=getattr(call, "name", None),
            ),
        )
    )


def append_control_tool_results(response: Any, calls: list[Any]) -> None:
    """为控制流调用补齐与最终决策一致的结果。

    Args:
        response: 当前 LLM 响应对象。
        calls: 当前响应中的全部 tool call。
    """
    control_calls = [
        call
        for call in calls
        if str(getattr(call, "name", "") or "") in {END_TURN_CALL, STOP_CALL}
    ]
    effective_call = next(
        (
            call
            for call in reversed(control_calls)
            if str(getattr(call, "name", "") or "") == STOP_CALL
        ),
        control_calls[-1] if control_calls else None,
    )

    for call in control_calls:
        name = str(getattr(call, "name", "") or "")
        if call is not effective_call:
            append_tool_result(response, call, "本次控制请求已被后续或更高优先级请求覆盖。")
        elif name == END_TURN_CALL:
            append_tool_result(response, call, "本轮已结束，等待后续消息。")
        else:
            append_tool_result(response, call, "当前对话已结束并进入冷却。")


def append_interrupted_tool_results(response: Any, calls: list[Any]) -> None:
    """为因新消息中断的工具调用补齐未执行结果。

    Args:
        response: 当前 LLM 响应对象。
        calls: 当前响应中的全部 tool call。
    """
    value = "生成期间收到新消息，本次调用未执行；将基于新消息重新规划。"
    for call in calls:
        append_tool_result(response, call, value)


def normalize_reply_text(text: str) -> str:
    """标准化回复文本，用于本轮复读比较。"""
    normalized = text.casefold()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？、,.!?；;：:~～…]+", "", normalized)
    return normalized


def _character_ngrams(text: str, size: int = 2) -> set[str]:
    """生成字符 n-gram 集合。"""
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def reply_similarity(left: str, right: str) -> tuple[float, float]:
    """返回字符 n-gram Jaccard 与较短文本覆盖率。"""
    left_normalized = normalize_reply_text(left)
    right_normalized = normalize_reply_text(right)
    if not left_normalized or not right_normalized:
        return 0.0, 0.0
    left_grams = _character_ngrams(left_normalized)
    right_grams = _character_ngrams(right_normalized)
    union = left_grams | right_grams
    intersection = left_grams & right_grams
    jaccard = len(intersection) / len(union) if union else 0.0
    if left_normalized in right_normalized:
        containment = 1.0
    elif right_normalized in left_normalized:
        containment = 0.0
    else:
        containment = len(intersection) / max(1, min(len(left_grams), len(right_grams)))
    return jaccard, containment


def is_repeated_reply(
    text: str,
    sent_texts: list[str],
    *,
    similarity_threshold: float,
    containment_threshold: float,
) -> bool:
    """判断文本是否与本轮已发送内容重复。"""
    candidate = normalize_reply_text(text)
    if not candidate:
        return False
    for sent in sent_texts:
        if candidate == normalize_reply_text(sent):
            return True
        similarity, containment = reply_similarity(text, sent)
        if similarity >= similarity_threshold or containment >= containment_threshold:
            return True
    return False


def append_post_speech_nudge(response: Any, *, duplicate: bool = False) -> None:
    """提示模型不要复述已发送文本，并继续工具或结束本轮。"""
    prefix = "刚才的文本与已发送内容重复，已被抑制。" if duplicate else "刚才的文本已经发送。"
    response.add_payload(
        LLMPayload(
            ROLE.USER,
            Text(
                prefix
                + "不要复述或改写已发送内容。若仍需查证或执行任务，只调用相关工具；"
                "只有工具结果带来新信息时才补充，否则调用 end_turn。"
            ),
        )
    )


def append_no_op_nudge(response: Any) -> None:
    """在模型既不说话也不调工具时追加一条推动消息。

    没有这条推动，模型可能陷入空转：既不产出文本也不调用工具，
    循环只能靠迭代上限兜底。

    Args:
        response: 当前 LLM 响应对象。
    """
    response.add_payload(
        LLMPayload(
            ROLE.USER,
            Text(
                "你刚才既没有说话也没有做任何事。"
                "如果你想说点什么就直接输出文本；"
                "如果这一轮不需要你介入，调用 end_turn 结束本轮。"
            ),
        )
    )

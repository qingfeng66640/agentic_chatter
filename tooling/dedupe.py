"""工具重复调用的软去重。

DFC 采用硬去重：同名同参的调用跨轮被直接拒绝执行，并向模型返回
「检测到跨轮重复工具调用，已自动跳过」。这种负反馈会进一步压低
模型本就不高的工具调用意愿。

本模块改为软去重：允许有限次数的重复，但在结果中附上上次的返回值
提醒模型「你刚查过，结果是 X」。这样既避免了真正的无意义空转，
又不会打击调用意愿。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

DedupeMode = Literal["soft", "hard", "off"]

# 附在软去重提醒中的上次结果的最大长度
MAX_ECHO_CHARS = 200


def build_call_key(name: str, args: Any) -> str:
    """构建工具调用的去重键。

    参数中的 ``reason`` 字段会被剔除，因为它只是模型的自述理由，
    不影响调用的实际语义。

    Args:
        name: 工具调用名称。
        args: 工具调用参数。

    Returns:
        str: 稳定的去重键。
    """
    payload = args
    if isinstance(args, dict):
        payload = {key: value for key, value in args.items() if key != "reason"}

    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except TypeError:
        serialized = str(payload)

    return f"{name}:{serialized}"


@dataclass
class DedupeDecision:
    """一次去重判定的结果。

    Attributes:
        allow: 是否允许本次调用真正执行。
        note: 需要附加给模型的提醒文本；为空表示无需提醒。
    """

    allow: bool
    note: str = ""


@dataclass
class CallDeduper:
    """工具调用去重器。

    Attributes:
        mode: 去重模式。soft 允许有限重复并附提醒；hard 直接拒绝重复；
            off 完全不干预。
        soft_limit: soft 模式下同一调用允许出现的最大次数。
        counts: 各去重键出现的次数。
        last_results: 各去重键上一次的执行结果文本，用于生成提醒。
    """

    mode: DedupeMode = "soft"
    soft_limit: int = 3
    counts: dict[str, int] = field(default_factory=dict)
    last_results: dict[str, str] = field(default_factory=dict)

    def check(self, name: str, args: Any) -> DedupeDecision:
        """判定一次工具调用是否放行。

        Args:
            name: 工具调用名称。
            args: 工具调用参数。

        Returns:
            DedupeDecision: 判定结果。
        """
        if self.mode == "off":
            return DedupeDecision(allow=True)

        key = build_call_key(name, args)
        seen = self.counts.get(key, 0)

        if seen == 0:
            self.counts[key] = 1
            return DedupeDecision(allow=True)

        if self.mode == "hard":
            return DedupeDecision(
                allow=False,
                note="你已经用相同参数调用过这个工具了，本次调用已跳过。",
            )

        # soft 模式
        if seen >= max(1, self.soft_limit):
            return DedupeDecision(
                allow=False,
                note=(
                    f"这个工具你已经用相同参数调用 {seen} 次了，"
                    "结果不会变化，请基于已有结果继续，不要再重复调用。"
                ),
            )

        self.counts[key] = seen + 1
        echo = self.last_results.get(key, "")
        if echo:
            return DedupeDecision(
                allow=True,
                note=f"提醒：你刚查过这个，上次的结果是「{echo}」。",
            )
        return DedupeDecision(allow=True)

    def record_result(self, name: str, args: Any, result: Any) -> None:
        """记录一次调用的执行结果，供后续重复调用时回显。

        Args:
            name: 工具调用名称。
            args: 工具调用参数。
            result: 执行结果。
        """
        key = build_call_key(name, args)
        if isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result, ensure_ascii=False, default=str)
            except TypeError:
                text = str(result)

        cleaned = " ".join(text.split())
        if len(cleaned) > MAX_ECHO_CHARS:
            cleaned = cleaned[: MAX_ECHO_CHARS - 1] + "…"
        self.last_results[key] = cleaned

    def reset(self) -> None:
        """清空去重状态，通常在一轮对话结束时调用。"""
        self.counts.clear()
        self.last_results.clear()

"""情绪状态推断。

从对话内容中推断情绪增量，并把结果写入全局心智。情绪不是外显的
数值，而是通过影响用词倾向让 bot 的表现更连贯：在某个群被惹毛了，
去另一个群说话确实会冲一点。

这里刻意不使用 LLM 判断情绪——那会给每轮增加一次调用开销。改用
轻量的关键词信号，配合时间衰减，效果足够且零成本。
"""

from __future__ import annotations

import re

# 正向信号词，命中后情绪上浮
_POSITIVE_PATTERNS = (
    "谢谢", "感谢", "太好了", "哈哈", "笑死", "喜欢", "厉害", "牛", "赞",
    "开心", "好耶", "可爱", "爱了", "有意思", "不错",
)
# 负向信号词，命中后情绪下沉
_NEGATIVE_PATTERNS = (
    "烦", "滚", "闭嘴", "无聊", "傻", "蠢", "废物", "垃圾", "讨厌",
    "生气", "难受", "别说了", "没用",
)

# 单条消息对情绪的最大影响幅度，防止个别极端消息把情绪拉满
MAX_DELTA_PER_TURN = 0.35
# 每命中一个信号词的基础权重
SIGNAL_WEIGHT = 0.12

_LAUGH_PATTERN = re.compile(r"(哈{2,}|233+|hhh+)")


def infer_mood_delta(text: str) -> tuple[float, str]:
    """从文本中推断情绪增量。

    Args:
        text: 待分析的文本，通常是本轮的未读消息拼接。

    Returns:
        tuple[float, str]: (情绪增量, 变化原因简述)。
            增量为 0 时表示无明显信号，原因为空字符串。
    """
    content = str(text or "")
    if not content.strip():
        return 0.0, ""

    positive_hits = sum(1 for word in _POSITIVE_PATTERNS if word in content)
    negative_hits = sum(1 for word in _NEGATIVE_PATTERNS if word in content)

    if _LAUGH_PATTERN.search(content):
        positive_hits += 1

    if positive_hits == negative_hits:
        return 0.0, ""

    raw = (positive_hits - negative_hits) * SIGNAL_WEIGHT
    delta = max(-MAX_DELTA_PER_TURN, min(MAX_DELTA_PER_TURN, raw))

    if delta > 0:
        reason = "刚才聊得挺愉快"
    else:
        reason = "刚才被呛了几句"

    return delta, reason


def describe_mood_for_prompt(mood_value: float) -> str:
    """将情绪值渲染成给模型看的行为引导。

    这段文本不描述数值，而是描述情绪应当如何影响表达，
    避免模型把情绪当成需要复述的状态。

    Args:
        mood_value: 当前情绪值，取值范围 [-1.0, 1.0]。

    Returns:
        str: 行为引导文本；情绪接近基线时返回空字符串。
    """
    if mood_value >= 0.5:
        return "你现在心情不错，说话可以更轻快一些，但不要刻意夸张。"
    if mood_value >= 0.2:
        return "你现在心情还行，正常发挥即可。"
    if mood_value <= -0.5:
        return (
            "你现在有点烦躁，回复会更简短、更直接一些，"
            "但不要迁怒于无关的人，也不要失礼。"
        )
    if mood_value <= -0.2:
        return "你现在情绪一般，回复可以稍微收敛一点。"
    return ""

"""拟人化行为模块。

提供消息分段、打字节奏、情绪推断与注意力模拟能力。
"""

from .attention import should_get_distracted, should_interrupt
from .mood import describe_mood_for_prompt, infer_mood_delta
from .segmenter import (
    CleanReplyResult,
    Segment,
    clean_reply_text,
    clean_reply_text_with_metadata,
    segment_reply,
)

__all__ = [
    "CleanReplyResult",
    "Segment",
    "clean_reply_text",
    "clean_reply_text_with_metadata",
    "describe_mood_for_prompt",
    "infer_mood_delta",
    "segment_reply",
    "should_get_distracted",
    "should_interrupt",
]

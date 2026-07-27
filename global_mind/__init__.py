"""跨流全局心智模块。

导出全局心智单例与渲染函数，供 chatter 主循环使用。
"""

from .render import render_global_awareness
from .store import GlobalMind, MoodState, StreamDigest, get_global_mind

__all__ = [
    "GlobalMind",
    "MoodState",
    "StreamDigest",
    "get_global_mind",
    "render_global_awareness",
]

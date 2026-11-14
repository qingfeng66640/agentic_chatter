"""AgenticChatter 测试入口、事件循环策略与全局心智隔离。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ..global_mind.store import GlobalMind

# plugins/agentic_chatter/tests/conftest.py → parents[3] 为仓库根
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """在 Windows 使用选择器事件循环，避免 socketpair 权限错误。"""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.fixture()
def fresh_mind() -> Iterator[GlobalMind]:
    """提供一个干净的全局心智实例，并在测试结束后复原。"""
    from ..global_mind import get_global_mind

    mind = get_global_mind()
    mind.clear()
    yield mind
    mind.clear()

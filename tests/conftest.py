"""Pytest bootstrap for agentic_chatter tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# plugins/agentic_chatter/tests/conftest.py → parents[3] 为仓库根
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Use selector loops on Windows to avoid socketpair permission failures."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.fixture()
def fresh_mind():
    """提供一个干净的全局心智实例，并在测试结束后复原。"""
    from ..global_mind import get_global_mind

    mind = get_global_mind()
    mind.clear()
    yield mind
    mind.clear()

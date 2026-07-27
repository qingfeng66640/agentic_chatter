"""渐进披露的流隔离与实际展开测试。"""

from __future__ import annotations

from types import SimpleNamespace

from ..tooling.explore import (
    ExploreToolsTool,
    clear_stream_catalog,
    consume_expansion,
    set_stream_catalog,
)


class _FakePlugin:
    """最小插件替身。"""


class _ExpandedTool:
    """待展开的假工具类。"""


async def test_expansion_is_scoped_by_stream() -> None:
    set_stream_catalog(
        "stream_a",
        {"platform": ["get_info"]},
        {"platform": [_ExpandedTool]},
    )
    set_stream_catalog(
        "stream_b",
        {"other": ["other_tool"]},
        {"other": []},
    )

    tool = ExploreToolsTool(_FakePlugin())
    tool._bind_runtime_context(stream_id="stream_a", message=SimpleNamespace(stream_id="stream_a"))

    success, text = await tool.execute("platform")

    assert success
    assert "get_info" in str(text)
    assert consume_expansion("stream_a") == [_ExpandedTool]
    assert consume_expansion("stream_b") == []

    clear_stream_catalog("stream_a")
    clear_stream_catalog("stream_b")


async def test_listing_categories_does_not_expand() -> None:
    set_stream_catalog("stream", {"platform": ["get_info"]}, {"platform": [_ExpandedTool]})
    tool = ExploreToolsTool(_FakePlugin())
    tool._bind_runtime_context(stream_id="stream")

    success, text = await tool.execute("")

    assert success
    assert "platform" in str(text)
    assert consume_expansion("stream") == []
    clear_stream_catalog("stream")

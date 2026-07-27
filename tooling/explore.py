"""渐进披露工具。

``ExploreToolsTool`` 让模型按类别请求展开被折叠的工具。请求会以
``stream_id`` 为键暂存；chatter 在工具执行结束后消费请求，把对应
工具 schema 注入同一条响应链的下一轮。状态按流隔离，避免并发串流。
"""

from __future__ import annotations

from threading import RLock
from typing import Annotated, Any

from src.core.components.base.tool import BaseTool

MAX_EXPANDED_TOOLS = 40

_lock = RLock()
_categories_by_stream: dict[str, dict[str, list[str]]] = {}
_classes_by_stream: dict[str, dict[str, list[type]]] = {}
_pending_by_stream: dict[str, str] = {}


def set_stream_catalog(
    stream_id: str,
    categories: dict[str, list[str]],
    classes: dict[str, list[type]],
) -> None:
    """设置某个流本轮可展开的工具目录。

    Args:
        stream_id: 聊天流 ID。
        categories: 分类到工具短名列表的映射。
        classes: 分类到工具类列表的映射。
    """
    key = str(stream_id or "").strip()
    if not key:
        return
    with _lock:
        _categories_by_stream[key] = {name: list(items) for name, items in categories.items()}
        _classes_by_stream[key] = {name: list(items) for name, items in classes.items()}
        _pending_by_stream.pop(key, None)


def consume_expansion(stream_id: str) -> list[type]:
    """消费某个流最近一次展开请求并返回对应工具类。

    Args:
        stream_id: 聊天流 ID。

    Returns:
        list[type]: 请求分类中的工具类；没有请求时返回空列表。
    """
    key = str(stream_id or "").strip()
    with _lock:
        category = _pending_by_stream.pop(key, None)
        if not category:
            return []
        return list(_classes_by_stream.get(key, {}).get(category, []))[:MAX_EXPANDED_TOOLS]


def clear_stream_catalog(stream_id: str) -> None:
    """清理某个流的渐进披露状态。

    Args:
        stream_id: 聊天流 ID。
    """
    key = str(stream_id or "").strip()
    with _lock:
        _categories_by_stream.pop(key, None)
        _classes_by_stream.pop(key, None)
        _pending_by_stream.pop(key, None)


class ExploreToolsTool(BaseTool):
    """展开被折叠的工具分类。"""

    tool_name = "explore_tools"
    tool_description = (
        "按类别展开当前未加载的工具。需要群管理、用户资料、特殊消息等能力，"
        "但当前列表没有合适工具时调用。不传类别会列出目录；指定类别后，"
        "该类别的工具会在下一步变为可调用。"
    )

    async def execute(
        self,
        category: Annotated[str, "要展开的工具类别名。留空则列出所有类别。"] = "",
    ) -> tuple[bool, str | dict[str, Any]]:
        """列出分类或登记一个实际展开请求。

        Args:
            category: 要展开的类别名；留空时仅返回分类目录。

        Returns:
            tuple[bool, str | dict]: 执行结果。
        """
        stream_id = self.get_current_stream_id()
        with _lock:
            categories = dict(_categories_by_stream.get(stream_id, {}))

        if not categories:
            return True, "当前没有折叠工具，你看到的已经是全部可用工具。"

        target = str(category or "").strip()
        if not target:
            return True, "可展开的工具类别有：" + "、".join(sorted(categories))

        matched_name = next(
            (
                name
                for name in categories
                if name.lower() == target.lower() or target.lower() in name.lower()
            ),
            None,
        )
        if matched_name is None:
            return False, f"没有类别「{target}」。可用类别：{'、'.join(sorted(categories))}。"

        names = categories[matched_name][:MAX_EXPANDED_TOOLS]
        with _lock:
            _pending_by_stream[stream_id] = matched_name

        return True, (
            f"已展开「{matched_name}」类别，下一步可以直接调用其中的工具："
            f"{'、'.join(names)}"
        )

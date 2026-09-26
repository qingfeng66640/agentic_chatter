"""Agentic Chatter 主 Agent 的原生图片内容构建。"""

from __future__ import annotations

import re
from typing import Any

from src.app.plugin_system.types import Message
from src.kernel.llm import Content, Image, Text

_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[图片(?:\(([0-9a-fA-F]+)\)(?::[^]]*)?)?\]")


def _media_items(message: Message) -> list[dict[str, Any]]:
    """读取消息中带原始数据的图片媒体。"""
    candidates: Any = None
    if isinstance(message.content, dict):
        candidates = message.content.get("media")
    if not isinstance(candidates, list) and isinstance(message.extra, dict):
        candidates = message.extra.get("media")
    if not isinstance(candidates, list):
        return []
    return [
        item
        for item in candidates
        if isinstance(item, dict)
        and item.get("type") == "image"
        and item.get("data")
    ]


def _image_index(messages: list[Message]) -> dict[str, dict[str, Any]]:
    """按 image_id 建立图片索引。"""
    index: dict[str, dict[str, Any]] = {}
    for message in messages:
        for item in _media_items(message):
            image_id = item.get("image_id")
            if isinstance(image_id, str) and image_id:
                index[image_id] = item
    return index


def inline_message_images(
    text: str,
    messages: list[Message],
    *,
    preferred_messages: list[Message] | None = None,
    max_images: int = 3,
) -> list[Content]:
    """将消息图片按占位符位置内联为 Text/Image 内容。

    ``preferred_messages`` 用于优先保留本轮新输入，历史图片不占用新输入上限。
    """
    if max_images <= 0:
        return [Text(text)]

    source_messages = list(messages)
    preferred = list(preferred_messages or [])
    ordered_messages = preferred + [message for message in source_messages if message not in preferred]
    index = _image_index(ordered_messages)
    preferred_ids = [
        item.get("image_id")
        for message in preferred
        for item in _media_items(message)
        if isinstance(item.get("image_id"), str)
    ]
    preferred_allowed = set(preferred_ids[:max_images])
    history_allowed = set(index).difference(preferred_ids)
    allowed = {
        image_id: item
        for image_id, item in index.items()
        if image_id in preferred_allowed or image_id in history_allowed
    }
    source_images = [
        item
        for message in ordered_messages
        for item in _media_items(message)
        if item.get("image_id") in allowed
    ]
    remaining_images = iter(source_images)
    content: list[Content] = []
    cursor = 0

    for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(text):
        if match.start() > cursor:
            content.append(Text(text[cursor : match.start()]))
        image_id = match.group(1)
        item = allowed.get(image_id) if image_id else None
        if item is None and image_id is None:
            try:
                item = next(remaining_images)
            except StopIteration:
                item = None
        content.append(Text(match.group(0)))
        if item is not None:
            content.append(Image(str(item["data"])))
        cursor = match.end()

    if cursor < len(text):
        content.append(Text(text[cursor:]))
    if not content:
        content.append(Text(text))
    return content


def image_count(messages: list[Message]) -> int:
    """统计消息中的原始图片数量。"""
    return sum(len(_media_items(message)) for message in messages)

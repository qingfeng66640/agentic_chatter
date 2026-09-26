"""Agentic Chatter 主回合原生图片输入测试。"""

from __future__ import annotations

from src.core.models import Message
from src.kernel.llm import Image, Text

from ..config import AgenticChatterConfig
from ..multimodal import image_count, inline_message_images


_DATA = "aGVsbG8="


def _message(text: str, image_id: str | None = None, data: str | None = None) -> Message:
    media = []
    if image_id is not None:
        media.append({"type": "image", "image_id": image_id, "data": data})
    return Message(
        message_id=image_id or "text",
        content={"text": text, "media": media},
        processed_plain_text=text,
        sender_id="user",
        chat_type="group",
    )


def test_native_multimodal_defaults_to_disabled_with_three_image_limit() -> None:
    config = AgenticChatterConfig()

    assert config.plugin.native_multimodal is False
    assert config.plugin.max_input_images == 3


def test_inline_images_preserves_message_position_and_image_identity() -> None:
    first = _message("看这张 [图片(a1)]", "a1", _DATA)
    second = _message("再看 [图片(b2)]", "b2", _DATA)

    content = inline_message_images(
        "【用户】看这张 [图片(a1)]\n【用户】再看 [图片(b2)]",
        [first, second],
        preferred_messages=[second],
        max_images=3,
    )

    assert [item.text for item in content if isinstance(item, Text)] == [
        "【用户】看这张 ",
        "[图片(a1)]",
        "\n【用户】再看 ",
        "[图片(b2)]",
    ]
    assert [item.value for item in content if isinstance(item, Image)] == [_DATA, _DATA]


def test_inline_images_limits_new_input_to_three_images() -> None:
    messages = [_message(f"[图片(a{index})]", f"a{index}", _DATA) for index in range(5)]
    text = " ".join(f"[图片(a{index})]" for index in range(5))

    content = inline_message_images(text, messages, preferred_messages=messages, max_images=3)

    assert len([item for item in content if isinstance(item, Image)]) == 3
    assert image_count(messages) == 5


def test_image_without_raw_data_remains_text_only() -> None:
    message = _message("[图片(missing)]", "missing", data="")

    content = inline_message_images("[图片(missing)]", [message], max_images=3)

    assert [item.text for item in content if isinstance(item, Text)] == ["[图片(missing)]"]
    assert not any(isinstance(item, Image) for item in content)

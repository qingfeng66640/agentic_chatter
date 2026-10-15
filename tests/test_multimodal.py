"""Agentic Chatter 主回合原生图片输入测试。"""

from __future__ import annotations

import base64

from src.core.models import Message
from src.kernel.llm import Image, Text

from ..config import AgenticChatterConfig
from ..multimodal import (
    MAX_SINGLE_IMAGE_BYTES,
    image_count,
    inline_message_images,
    summarize_budget,
)


_DATA = "aGVsbG8="


def _b64(nbytes: int) -> str:
    """返回长度为 ``nbytes`` 的合法 base64 字符串。"""
    return base64.b64encode(b"\x00" * nbytes).decode()


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


def _msg(
    text: str,
    *,
    mid: str | None,
    iid: str | None = None,
    data: str | None = None,
) -> Message:
    """构造 message_id 与 image_id 独立可控的消息。

    ``mid`` 必须**原样透传**（含空串），不要写 ``mid or "text"`` 兜底——
    那样多条空 id 的消息会塌成同一 id。
    """
    media = []
    if iid is not None:
        media.append({"type": "image", "image_id": iid, "data": data})
    return Message(
        message_id=mid,
        content={"text": text, "media": media},
        processed_plain_text=text,
        sender_id="user",
        chat_type="group",
    )


def _joined(messages: list[Message]) -> str:
    return "\n".join(f"【用户】{message.content['text']}" for message in messages)


def _emitted(content: list) -> list[Image]:
    return [item for item in content if isinstance(item, Image)]


def _texts(content: list) -> list[str]:
    return [item.text for item in content if isinstance(item, Text)]


def _ids(content: list) -> list[str]:
    return [item.value for item in _emitted(content)]


def _sizes(content: list) -> list[int]:
    """返回每张实际发射图片的原始字节数。

    正则占位符的 substring 断言是**恒真**的——被断言的 id 本来就在输入文本里，
    只有真被改写成 ``[图片已省略]`` 才会消失。断言"发的是哪一张"必须看
    真正进入 payload 的内容，所以用可区分的字节数做身份。
    """
    return [len(base64.b64decode(item.value)) for item in _emitted(content)]


# --------------------------------------------------------------------------
# 既有行为
# --------------------------------------------------------------------------


def test_native_multimodal_defaults_to_disabled_with_three_image_limit() -> None:
    config = AgenticChatterConfig()

    assert config.plugin.native_multimodal is False
    assert config.plugin.max_input_images == 3
    assert config.plugin.max_input_image_bytes == 8 * 1024 * 1024


def test_inline_images_preserves_message_position_and_image_identity() -> None:
    first = _message("看这张 [图片(a1)]", "a1", _DATA)
    second = _message("再看 [图片(b2)]", "b2", _DATA)

    content = inline_message_images(
        "【用户】看这张 [图片(a1)]\n【用户】再看 [图片(b2)]",
        [first, second],
        preferred_messages=[second],
        max_images=3,
    )

    assert _texts(content) == [
        "【用户】看这张 ",
        "[图片(a1)]",
        "\n【用户】再看 ",
        "[图片(b2)]",
    ]
    assert _ids(content) == [_DATA, _DATA]


def test_inline_images_limits_new_input_to_three_images() -> None:
    messages = [_message(f"[图片(a{index})]", f"a{index}", _DATA) for index in range(5)]
    text = " ".join(f"[图片(a{index})]" for index in range(5))

    content = inline_message_images(text, messages, preferred_messages=messages, max_images=3)

    assert len(_emitted(content)) == 3
    assert image_count(messages) == 5


def test_image_without_raw_data_remains_text_only() -> None:
    message = _message("[图片(missing)]", "missing", data="")

    content = inline_message_images("[图片(missing)]", [message], max_images=3)

    assert _texts(content) == ["[图片(missing)]"]
    assert not _emitted(content)


# --------------------------------------------------------------------------
# 排序：本轮优先、统一新→旧
# --------------------------------------------------------------------------


def test_ordering_prefers_current_turn_then_newest_history() -> None:
    # 每张图用不同字节数，以便断言"发的是哪几张"而不是只数张数
    history = [
        _msg(
            f"[图片(a{i:02x})]", mid=f"h{i}", iid=f"a{i:02x}", data=_b64(8 + i)
        )
        for i in range(10)
    ]
    current = _msg("[图片(ff01)]", mid="cur", iid="ff01", data=_b64(100))

    content = inline_message_images(
        _joined(history + [current]),
        history + [current],
        preferred_messages=[current],
        max_images=3,
    )

    # 本轮图(100B) + 最新两张历史(17B, 16B)。
    # 发射顺序跟随**正文顺序**（历史在前、本轮在后），故按排序后比较集合。
    assert sorted(_sizes(content)) == [16, 17, 100]


def test_ordering_keeps_newest_history_within_budget_of_two() -> None:
    history = [
        _msg(
            f"[图片(b{i:02x})]", mid=f"h{i}", iid=f"b{i:02x}", data=_b64(8 + i)
        )
        for i in range(5)
    ]
    current = _msg("[图片(ee01)]", mid="cur", iid="ee01", data=_b64(100))

    content = inline_message_images(
        _joined(history + [current]),
        history + [current],
        preferred_messages=[current],
        max_images=2,
    )

    # 本轮图(100B) + 最新一张历史(12B)
    assert sorted(_sizes(content)) == [12, 100]


def test_ordering_single_slot_keeps_current_turn() -> None:
    history = [
        _msg(
            f"[图片(c{i:02x})]", mid=f"h{i}", iid=f"c{i:02x}", data=_b64(8 + i)
        )
        for i in range(5)
    ]
    current = _msg("[图片(dd01)]", mid="cur", iid="dd01", data=_b64(100))

    content = inline_message_images(
        _joined(history + [current]),
        history + [current],
        preferred_messages=[current],
        max_images=1,
    )

    # 只发了本轮那张 100B 的，而不是最旧历史的 8B
    assert _sizes(content) == [100]
    assert "[图片已省略]" in "".join(_texts(content))


def test_ordering_history_only_keeps_newest() -> None:
    history = [
        _msg(
            f"[图片(e{i:02x})]", mid=f"h{i}", iid=f"e{i:02x}", data=_b64(8 + i)
        )
        for i in range(6)
    ]

    content = inline_message_images(_joined(history), history, max_images=3)

    # 最新三张：13B / 12B / 11B
    assert sorted(_sizes(content)) == [11, 12, 13]


def test_dedup_prefers_current_turn_copy_when_history_lost_its_data() -> None:
    """历史副本回填失败（无 data）时，本轮副本必须胜出。"""
    history_copy = _msg("[图片(f101)]", mid="same", iid="f101", data="")
    current_copy = _msg("[图片(f101)]", mid="same", iid="f101", data=_DATA)

    content = inline_message_images(
        _joined([history_copy, current_copy]),
        [history_copy, current_copy],
        preferred_messages=[current_copy],
        max_images=1,
    )

    assert len(_emitted(content)) == 1
    assert _ids(content) == [_DATA]


def test_production_shape_regression_bounds_payload() -> None:
    """复刻生产事故窗口形状：21 引用 / 16 distinct / 含 7.22MB 大图 / 重复 id。"""
    order_new_to_old = [
        "c818e5e0", "7d9dfa5b", "1840c6b2", "844d5113", "d4bb3b04", "c5e679c3",
        "73391500", "b30a5a0b", "deb9b5a0", "3dce779e", "7b234f08", "2b59024f",
        "c843ec70", "726c5927", "c775b727", "852bfdc2",
    ]  # fmt: skip
    sizes = {
        "c818e5e0": 1_604_096,
        "7d9dfa5b": 68_184,
        "1840c6b2": 7_569_272,
        "844d5113": 132_448,
        "d4bb3b04": 225_280,
        "73391500": 136_528,
    }
    total_budget = 4 * 1024 * 1024

    def size_of(image_id: str) -> int:
        return sizes.get(image_id, 104_856)

    # 消息写入顺序是旧→新；候选是倒序出来的，所以本轮要写"新→旧"才能让
    # 最新图落在候选首位。此处刻意让 7.22MB 的 1840c6b2 成为本轮最旧的一条。
    current_ids = list(reversed(order_new_to_old[:3]))
    # 补 5 个重复引用：d4bb3b04 x3、73391500 x2，共 21 引用 / 16 distinct
    history_ids = ["d4bb3b04"] * 3 + ["73391500"] * 2 + list(
        reversed(order_new_to_old[3:])
    )
    history = [
        _msg(f"[图片({iid})]", mid=f"h{index}", iid=iid, data=_b64(size_of(iid)))
        for index, iid in enumerate(history_ids)
    ]
    current = [
        _msg(f"[图片({iid})]", mid=f"u{index}", iid=iid, data=_b64(size_of(iid)))
        for index, iid in enumerate(current_ids)
    ]
    messages = history + current

    assert image_count(messages) == 21

    content = inline_message_images(
        _joined(messages),
        messages,
        preferred_messages=current,
        max_images=3,
        max_bytes=total_budget,
    )

    emitted = _emitted(content)
    text = "".join(_texts(content))
    # 张数硬上限
    assert len(emitted) == 3
    # 体积上界：首张豁免累计但受单张天花板约束
    emitted_bytes = sum(len(item.value) for item in emitted)
    assert emitted_bytes <= max(total_budget, MAX_SINGLE_IMAGE_BYTES)
    # 最新图在存活集，最旧的三个都不在
    assert "[图片(c818e5e0)]" in text
    assert "[图片(852bfdc2)]" not in text
    assert "[图片(c775b727)]" not in text
    assert "[图片(726c5927)]" not in text
    # 7.22MB 的 GIF 应被字节闸门挡下
    assert emitted_bytes < sizes["1840c6b2"]
    # 同一 image_id 只发射一次
    emitted_values = [item.value for item in emitted]
    assert len(set(emitted_values)) == len(emitted_values)


# --------------------------------------------------------------------------
# 张数与去重
# --------------------------------------------------------------------------


def test_count_limit_caps_twenty_history_images() -> None:
    history = [
        _msg(f"[图片(1{i:03x})]", mid=f"h{i}", iid=f"1{i:03x}", data=_b64(8))
        for i in range(20)
    ]

    content = inline_message_images(_joined(history), history, max_images=3)

    assert len(_emitted(content)) == 3


def test_count_limit_caps_current_turn_images() -> None:
    current = [
        _msg(f"[图片(2{i:03x})]", mid=f"u{i}", iid=f"2{i:03x}", data=_b64(8))
        for i in range(5)
    ]

    content = inline_message_images(
        _joined(current), current, preferred_messages=current, max_images=3
    )

    assert len(_emitted(content)) == 3


def test_zero_count_short_circuits_to_plain_text() -> None:
    history = [_msg("[图片(3a01)]", mid="h", iid="3a01", data=_b64(8))]

    content = inline_message_images(_joined(history), history, max_images=0)

    assert _texts(content) == [_joined(history)]
    assert not _emitted(content)


def test_same_image_in_two_messages_emits_once() -> None:
    first = _msg("[图片(4b01)]", mid="m1", iid="4b01", data=_b64(20))
    second = _msg("[图片(4b01)]", mid="m2", iid="4b01", data=_b64(20))

    content = inline_message_images(
        _joined([first, second]), [first, second], max_images=2
    )

    assert _sizes(content) == [20]


def test_repeated_placeholder_emits_once_and_marks_rest() -> None:
    message = _msg("", mid="m", iid="5c01", data=_b64(8))
    text = " ".join(["[图片(5c01)]"] * 5)

    content = inline_message_images(text, [message], max_images=3)

    assert len(_emitted(content)) == 1
    assert "".join(_texts(content)).count("[图片(同图,上文已提供)]") == 4


def test_repeated_placeholder_keeps_description() -> None:
    message = _msg("", mid="m", iid="6d01", data=_b64(8))
    text = "[图片(6d01):猫] 再看 [图片(6d01):猫]"

    content = inline_message_images(text, [message], max_images=3)

    assert len(_emitted(content)) == 1
    assert "[图片(同图,上文已提供):猫]" in _texts(content)


def test_same_message_id_in_history_and_unread_emits_once() -> None:
    history = _msg("[图片(7e01)]", mid="same", iid="7e01", data=_b64(8))
    current = _msg("[图片(7e01)]", mid="same", iid="7e01", data=_b64(8))

    content = inline_message_images(
        _joined([history, current]),
        [history, current],
        preferred_messages=[current],
        max_images=2,
    )

    assert len(_emitted(content)) == 1


def test_empty_message_id_does_not_collapse_messages() -> None:
    first = _msg("[图片(8f01)]", mid="", iid="8f01", data=_b64(8))
    second = _msg("[图片(9a01)]", mid="", iid="9a01", data=_b64(8))

    content = inline_message_images(
        _joined([first, second]), [first, second], max_images=3
    )

    assert len(_emitted(content)) == 2


def test_empty_unread_falls_back_to_history_ordering() -> None:
    history = [
        _msg(f"[图片(a{i:02x})]", mid=f"h{i}", iid=f"a{i:02x}", data=_b64(8))
        for i in range(4)
    ]

    content = inline_message_images(_joined(history), history, max_images=2)

    text = "".join(_texts(content))
    assert "[图片(a03)]" in text
    assert "[图片(a00)]" not in text


# --------------------------------------------------------------------------
# 字节预算
# --------------------------------------------------------------------------


def test_byte_budget_boundary_is_exact() -> None:
    first = _msg("[图片(b101)]", mid="m1", iid="b101", data=_b64(96))
    second = _msg("[图片(b202)]", mid="m2", iid="b202", data=_b64(96))
    messages = [first, second]
    both = len(_b64(96)) * 2

    loose = inline_message_images(_joined(messages), messages, max_images=3, max_bytes=both)
    tight = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=both - 1
    )

    assert len(_emitted(loose)) == 2
    assert len(_emitted(tight)) == 1


def test_first_image_is_always_allowed_within_tight_budget() -> None:
    message = _msg("[图片(c303)]", mid="m", iid="c303", data=_b64(8))

    content = inline_message_images(
        _joined([message]), [message], max_images=1, max_bytes=10
    )

    assert len(_emitted(content)) == 1


def test_single_image_ceiling_applies_to_first_image() -> None:
    oversized = _b64(MAX_SINGLE_IMAGE_BYTES + 1)
    message = _msg("[图片(d404)]", mid="m", iid="d404", data=oversized)

    content = inline_message_images(
        _joined([message]),
        [message],
        max_images=1,
        max_bytes=100 * 1024 * 1024,
    )

    assert not _emitted(content)
    assert "[图片已省略]" in _texts(content)


def test_budget_skips_oversized_and_fills_with_later_smaller() -> None:
    """可判别 continue 与 break 的构造：跳过中间超预算的那张，继续往后填。"""
    newest = _msg("[图片(e501)]", mid="m1", iid="e501", data=_b64(1_800_000))
    middle = _msg("[图片(e602)]", mid="m2", iid="e602", data=_b64(1_800_000))
    oldest = _msg("[图片(e703)]", mid="m3", iid="e703", data=_b64(100))
    messages = [oldest, middle, newest]
    budget = len(_b64(1_800_000)) + len(_b64(100))

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=budget
    )

    text = "".join(_texts(content))
    assert len(_emitted(content)) == 2
    assert "[图片(e501)]" in text
    assert "[图片(e703)]" in text
    assert "[图片已省略]" in text


def test_zero_byte_budget_disables_volume_limit() -> None:
    messages = [
        _msg(f"[图片(f{i:02x})]", mid=f"m{i}", iid=f"f{i:02x}", data=_b64(600))
        for i in range(3)
    ]

    content = inline_message_images(_joined(messages), messages, max_images=3, max_bytes=0)

    assert len(_emitted(content)) == 3


# --------------------------------------------------------------------------
# 占位符标记
# --------------------------------------------------------------------------


def test_trimmed_history_placeholder_is_marked_omitted() -> None:
    history = [
        _msg(f"[图片(1{i:03x}):猫]", mid=f"h{i}", iid=f"1{i:03x}", data=_b64(8))
        for i in range(4)
    ]

    content = inline_message_images(_joined(history), history, max_images=2)

    text = "".join(_texts(content))
    assert "[图片已省略]" in text
    assert ":猫]" not in text.split("[图片已省略]")[0][-10:]


def test_placeholder_with_hex_id_but_no_data_is_marked_omitted() -> None:
    message = _msg("[图片(2a01)]", mid="m", iid="2a01", data="")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert "[图片已省略]" in _texts(content)
    assert not _emitted(content)


def test_desc_only_placeholder_is_marked_omitted() -> None:
    content = inline_message_images("[图片:猫]", [], max_images=3)

    assert "[图片已省略]" in _texts(content)
    assert not _emitted(content)


def test_known_placeholder_forms_are_not_rewritten() -> None:
    """每个样本单独调用，避免 id 占位符先去重而造成假红。"""
    samples = [
        ("[图片(ab12):描述]", "ab12"),
        ("[图片(cd34)]", "cd34"),
        ("[图片已省略]", "ef56"),
        ("[图片(同图,上文已提供):描述]", "a178"),
    ]
    for text, image_id in samples:
        message = _msg("", mid="m", iid=image_id, data=_b64(8))
        content = inline_message_images(text, [message], max_images=3)
        assert text in _texts(content), text


def test_bare_placeholder_does_not_steal_id_referenced_image() -> None:
    """预算只够一张时，裸占位符不得抢走有 id 引用的那张。"""
    history = _msg("[图片]", mid="h", iid="3b01", data=_b64(8))
    current = _msg("[图片(3c02)]", mid="cur", iid="3c02", data=_b64(8))
    messages = [history, current]

    content = inline_message_images(
        _joined(messages), messages, preferred_messages=[current], max_images=1
    )

    text = "".join(_texts(content))
    # 只发了有精确引用、且更新的那张
    assert len(_emitted(content)) == 1
    assert "[图片(3c02)]" in text
    # 裸占位符无池可领 → 显式省略
    assert "[图片已省略]" in text
    assert "[图片]\n" not in text


def test_bare_placeholder_claims_an_unreferenced_image() -> None:
    message = _msg("[图片]", mid="m", iid="4d01", data=_b64(8))

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert len(_emitted(content)) == 1
    assert "[图片已省略]" not in _texts(content)


def test_no_placeholder_in_text_emits_nothing() -> None:
    messages = [
        _msg("", mid=f"m{i}", iid=f"5e{i:02x}", data=_b64(8)) for i in range(3)
    ]

    content = inline_message_images("这里没有任何占位符", messages, max_images=3)

    assert not _emitted(content)


# --------------------------------------------------------------------------
# 健壮性
# --------------------------------------------------------------------------


def test_invalid_base64_does_not_raise() -> None:
    message = _msg("[图片(6f01)]", mid="m", iid="6f01", data="!!!not-base64!!!")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert not _emitted(content)
    assert "[图片已省略]" in _texts(content)


def test_whitespace_data_does_not_produce_empty_image() -> None:
    message = _msg("[图片(7a02)]", mid="m", iid="7a02", data="   ")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert not _emitted(content)


# --------------------------------------------------------------------------
# 预算统计
# --------------------------------------------------------------------------


def test_summarize_budget_reports_all_counters() -> None:
    messages = [
        _msg(f"[图片(8{i:02x})]", mid=f"h{i}", iid=f"8{i:02x}", data=_b64(96))
        for i in range(4)
    ]
    content = inline_message_images(_joined(messages), messages, max_images=2)

    summary = summarize_budget(content, messages, max_bytes=1024 * 1024)

    assert summary["ref_count"] == 4
    assert summary["distinct_count"] == 4
    assert summary["emitted"] == 2
    assert summary["emitted_bytes"] == 2 * len(_b64(96))
    assert summary["trimmed"] is True
    assert summary["over_bytes"] is False
    assert summary["should_warn"] is True
    assert summary["reason"] == "count"


def test_summarize_budget_zero_byte_budget_has_no_bytes_reason() -> None:
    messages = [
        _msg(f"[图片(9{i:02x})]", mid=f"h{i}", iid=f"9{i:02x}", data=_b64(8))
        for i in range(3)
    ]
    content = inline_message_images(_joined(messages), messages, max_images=2, max_bytes=0)

    summary = summarize_budget(content, messages, max_bytes=0)

    assert summary["should_warn"] is True
    assert summary["over_bytes"] is False
    assert "bytes" not in summary["reason"]


def test_summarize_budget_silent_when_nothing_trimmed() -> None:
    messages = [_msg("[图片(a101)]", mid="m", iid="a101", data=_b64(8))]
    content = inline_message_images(_joined(messages), messages, max_images=3)

    summary = summarize_budget(content, messages, max_bytes=1024 * 1024)

    assert summary["trimmed"] is False
    assert summary["over_bytes"] is False
    assert summary["should_warn"] is False
    assert summary["reason"] == ""


def test_summarize_budget_reports_both_reasons() -> None:
    messages = [
        _msg(f"[图片(b{i:03x})]", mid=f"h{i}", iid=f"b{i:03x}", data=_b64(96))
        for i in range(4)
    ]
    # 预算比单张还小：首张仍豁免累计，故实际发出 1 张且体积超限
    tight = len(_b64(96)) - 1
    content = inline_message_images(
        _joined(messages), messages, max_images=2, max_bytes=tight
    )

    summary = summarize_budget(content, messages, max_bytes=tight)

    assert summary["trimmed"] is True
    assert summary["over_bytes"] is True
    assert summary["reason"] == "both"


# --------------------------------------------------------------------------
# 对抗审查补的盲区用例
# --------------------------------------------------------------------------


def test_cumulative_budget_accumulates_across_images() -> None:
    """字节预算必须**累加**——`used = size` 只会看最后一张。

    变异测试发现：所有既有字节用例最多只放 2 张等长图，此时
    ``used += size`` 与 ``used = size`` 数学等价，删掉累加测试也不红。
    """
    messages = [
        _msg(f"[图片(1{i:03x})]", mid=f"m{i}", iid=f"1{i:03x}", data=_b64(600))
        for i in range(3)
    ]
    budget = int(len(_b64(600)) * 2.5)

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=budget
    )

    # 2.5 张的预算只装得下 2 张；第三张会把累计推过上限
    assert _sizes(content) == [600, 600]
    assert sum(len(item.value) for item in _emitted(content)) <= budget


def test_candidate_dedup_prevents_double_counted_budget() -> None:
    """同一 image_id 跨消息只应占用**一次**字节预算。

    若候选列表不去重，``used`` 会为同一张图累加两次，把本该装下的后续图片挤掉。
    下游的 ``intext`` 集合只在发射阶段兜住重复输出，掩盖了预算侧的漏洞。
    """
    dup = _msg("[图片(2a01)]", mid="m1", iid="2a01", data=_b64(600))
    dup2 = _msg("[图片(2a01)]", mid="m2", iid="2a01", data=_b64(600))
    other = _msg("[图片(2b02)]", mid="m3", iid="2b02", data=_b64(600))
    messages = [dup, dup2, other]
    budget = int(len(_b64(600)) * 2.5)

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=budget
    )

    # 去重后占用 600，剩 2.5-1 = 1.5 张额度，other 装得下 → 发 2 张
    assert _sizes(content) == [600, 600]
    assert sum(len(item.value) for item in _emitted(content)) <= budget


def test_bare_placeholders_are_assigned_in_reverse_document_order() -> None:
    """裸占位符必须**从正文最后一个开始**分配。

    正文行序是「历史→本轮」，候选序是「本轮→历史」，二者互为镜像；
    反向分配才能让本轮行优先拿到最新的图。变异测试发现：既有裸占位符用例
    各自只有一个裸占位符，单点分配时正序/逆序根本不可观测。
    """
    history = _msg("[图片]", mid="h", iid="3a01", data=_b64(10))
    current = _msg("[图片]", mid="cur", iid="3b02", data=_b64(200))
    messages = [history, current]

    content = inline_message_images(
        _joined(messages), messages, preferred_messages=[current], max_images=3
    )

    # 正文第一行（历史）拿小图，第二行（本轮）拿大图
    assert _sizes(content) == [10, 200]


def test_images_without_placeholder_do_not_consume_budget() -> None:
    """正文里没有占位符的图不应占用预算名额。

    可达路径：提示词模板缺失时 ``_build_user_prompt`` 退化为只返回本轮文本，
    历史图没有任何占位符，却会白白吃掉全部名额。
    """
    messages = [
        _msg("", mid=f"m{i}", iid=f"4{i:02x}", data=_b64(8 + i)) for i in range(5)
    ]
    wanted = _msg("[图片(4ff)]", mid="want", iid="4ff", data=_b64(200))
    messages = [wanted] + messages

    content = inline_message_images(
        _joined(messages), messages, preferred_messages=[wanted], max_images=1
    )

    assert _sizes(content) == [200]


def test_production_shape_summarize_budget_counts_distinct_separately() -> None:
    """21 引用 / 16 distinct 的形状下，``ref_count`` 与 ``distinct_count`` 必须不同。

    变异测试发现：既有 ``summarize_budget`` 用例的 image_id 互不相同，
    两个计数恒相等，``distinct_count`` 换成 ``ref_count`` 测试也不红。
    """
    order = ["c818e5e0", "7d9dfa5b", "1840c6b2", "844d5113", "d4bb3b04", "c5e679c3",
             "73391500", "b30a5a0b", "deb9b5a0", "3dce779e", "7b234f08", "2b59024f",
             "c843ec70", "726c5927", "c775b727", "852bfdc2"]  # fmt: skip
    rest = ["d4bb3b04"] * 3 + ["73391500"] * 2 + order[3:]
    history = [
        _msg(f"[图片({iid})]", mid=f"h{index}", iid=iid, data=_b64(8))
        for index, iid in enumerate(rest)
    ]
    current = [
        _msg(f"[图片({iid})]", mid=f"u{index}", iid=iid, data=_b64(8))
        for index, iid in enumerate(reversed(order[:3]))
    ]
    messages = history + current
    content = inline_message_images(
        _joined(messages), messages, preferred_messages=current, max_images=3
    )

    summary = summarize_budget(content, messages, max_bytes=8 * 1024 * 1024)

    assert summary["ref_count"] == 21
    assert summary["distinct_count"] == 16
    assert summary["emitted"] == 3
    assert summary["trimmed"] is True
    assert summary["reason"] == "count"


def test_payload_lookalike_path_is_not_read_from_disk() -> None:
    """形似路径的载荷不得被当成文件读进请求体。

    ``Image`` 遇到无法解码的字符串不会抛错，而是当路径读盘并 base64 编码。
    """
    message = _msg("[图片(5a01)]", mid="m", iid="5a01", data="README.md")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert not _emitted(content)
    assert "[图片已省略]" in _texts(content)


def test_bracket_in_description_does_not_swallow_next_placeholder() -> None:
    """描述里的 ``]`` 不得把后面的占位符一并吞掉。"""
    first = _msg("[图片(ab12):图[1]的猫]", mid="m1", iid="ab12", data=_b64(20))
    second = _msg("[图片(cd34)]", mid="m2", iid="cd34", data=_b64(30))
    messages = [first, second]

    content = inline_message_images(
        _joined(messages), messages, max_images=3
    )

    # 第二张仍必须能内联，不能被第一张的描述吞掉
    assert 30 in _sizes(content)

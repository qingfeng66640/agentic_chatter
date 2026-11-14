"""Agentic Chatter 主回合原生图片输入测试。"""

from __future__ import annotations

import base64

from src.core.models import Message
from src.kernel.llm import Content, Image, Text

from ..config import AgenticChatterConfig
from ..multimodal import (
    DEFAULT_MAX_IMAGE_BYTES,
    MAX_SINGLE_IMAGE_BYTES,
    _payload_body,
    _raw_bytes,
    _safe_image,
    build_image_content,
    image_count,
    inline_message_images,
    summarize_budget,
    unusable_media_summary,
)

_DATA = "aGVsbG8="


def _b64(nbytes: int) -> str:
    """返回由 ``nbytes`` 个零字节编码得到的 base64 字符串。"""
    return base64.b64encode(b"\x00" * nbytes).decode()


def _message(
    text: str, image_id: str | None = None, data: str | None = None
) -> Message:
    """构造可选携带一张图片的群聊消息。

    Args:
        text: 消息正文，同时写入纯文本字段。
        image_id: 图片标识；未提供时不添加媒体项。
        data: 图片载荷，可为空以模拟不可用图片。

    Returns:
        以图片标识或固定文本标识作为消息 ID 的消息。
    """
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

    Args:
        text: 消息正文，同时写入纯文本字段。
        mid: 原样保留的消息标识，允许为空或缺失。
        iid: 图片标识；未提供时不添加媒体项。
        data: 图片载荷，允许为空以模拟历史回填失败。

    Returns:
        消息标识与图片标识独立设置的群聊消息。
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
    """按输入顺序为消息正文添加用户前缀并以换行拼接。"""
    return "\n".join(f"【用户】{message.content['text']}" for message in messages)


def _emitted(content: list[Content]) -> list[Image]:
    """按载荷顺序提取实际进入请求体的图片对象。"""
    return [item for item in content if isinstance(item, Image)]


def _texts(content: list[Content]) -> list[str]:
    """按载荷顺序提取文本片段，忽略图片内容。"""
    return [item.text for item in content if isinstance(item, Text)]


def _ids(content: list[Content]) -> list[str]:
    """返回实际发射图片的 base64 载荷，供图片身份断言使用。"""
    return [item.value for item in _emitted(content)]


def _sizes(content: list[Content]) -> list[int]:
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
    """默认关闭原生多模态，并将图片预算设为三张和 8 MiB。"""
    config = AgenticChatterConfig()

    assert config.plugin.native_multimodal is False
    assert config.plugin.max_input_images == 3
    assert config.plugin.max_input_image_bytes == 8 * 1024 * 1024


def test_inline_images_preserves_message_position_and_image_identity() -> None:
    """图片内联后保留原文与占位符顺序，并输出对应的两份图片载荷。"""
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
    """五张输入图片最多内联三张，但原始引用计数仍为五。"""
    messages = [_message(f"[图片(a{index})]", f"a{index}", _DATA) for index in range(5)]
    text = " ".join(f"[图片(a{index})]" for index in range(5))

    content = inline_message_images(
        text, messages, preferred_messages=messages, max_images=3
    )

    assert len(_emitted(content)) == 3
    assert image_count(messages) == 5


def test_image_without_raw_data_remains_text_only() -> None:
    """缺少载荷且标识不匹配十六进制格式时保留原占位文本，不产生图片。"""
    message = _message("[图片(missing)]", "missing", data="")

    content = inline_message_images("[图片(missing)]", [message], max_images=3)

    assert _texts(content) == ["[图片(missing)]"]
    assert not _emitted(content)


# --------------------------------------------------------------------------
# 排序：本轮优先、统一新→旧
# --------------------------------------------------------------------------


def test_ordering_prefers_current_turn_then_newest_history() -> None:
    """三张额度优先保留本轮图片，再选取最新的两张历史图片。"""
    # 每张图用不同字节数，以便断言"发的是哪几张"而不是只数张数
    history = [
        _msg(f"[图片(a{i:02x})]", mid=f"h{i}", iid=f"a{i:02x}", data=_b64(8 + i))
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
    """两张额度同时保留本轮图片与最新的一张历史图片。"""
    history = [
        _msg(f"[图片(b{i:02x})]", mid=f"h{i}", iid=f"b{i:02x}", data=_b64(8 + i))
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
    """仅有一个名额时保留本轮图片，并将未选中的图片标记为省略。"""
    history = [
        _msg(f"[图片(c{i:02x})]", mid=f"h{i}", iid=f"c{i:02x}", data=_b64(8 + i))
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
    """没有本轮图片时，三张额度用于最新的三张历史图片。"""
    history = [
        _msg(f"[图片(e{i:02x})]", mid=f"h{i}", iid=f"e{i:02x}", data=_b64(8 + i))
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
        """返回事故样本指定的图片字节数，未单列的图片统一取 104856 字节。"""
        return sizes.get(image_id, 104_856)

    # 消息写入顺序是旧→新；候选是倒序出来的，所以本轮要写"新→旧"才能让
    # 最新图落在候选首位。此处刻意让 7.22MB 的 1840c6b2 成为本轮最旧的一条。
    current_ids = list(reversed(order_new_to_old[:3]))
    # 补 5 个重复引用：d4bb3b04 x3、73391500 x2，共 21 引用 / 16 distinct
    history_ids = (
        ["d4bb3b04"] * 3 + ["73391500"] * 2 + list(reversed(order_new_to_old[3:]))
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
    """二十张历史图片在三张额度下仅发射三张。"""
    history = [
        _msg(f"[图片(1{i:03x})]", mid=f"h{i}", iid=f"1{i:03x}", data=_b64(8))
        for i in range(20)
    ]

    content = inline_message_images(_joined(history), history, max_images=3)

    assert len(_emitted(content)) == 3


def test_count_limit_caps_current_turn_images() -> None:
    """本轮的五张图片同样受三张总数上限约束。"""
    current = [
        _msg(f"[图片(2{i:03x})]", mid=f"u{i}", iid=f"2{i:03x}", data=_b64(8))
        for i in range(5)
    ]

    content = inline_message_images(
        _joined(current), current, preferred_messages=current, max_images=3
    )

    assert len(_emitted(content)) == 3


def test_zero_count_short_circuits_to_plain_text() -> None:
    """图片数量上限为零时原样返回拼接文本，不发射图片。"""
    history = [_msg("[图片(3a01)]", mid="h", iid="3a01", data=_b64(8))]

    content = inline_message_images(_joined(history), history, max_images=0)

    assert _texts(content) == [_joined(history)]
    assert not _emitted(content)


def test_same_image_in_two_messages_emits_once() -> None:
    """不同消息引用同一图片标识时只发射一份图片载荷。"""
    first = _msg("[图片(4b01)]", mid="m1", iid="4b01", data=_b64(20))
    second = _msg("[图片(4b01)]", mid="m2", iid="4b01", data=_b64(20))

    content = inline_message_images(
        _joined([first, second]), [first, second], max_images=2
    )

    assert _sizes(content) == [20]


def test_repeated_placeholder_emits_once_and_marks_rest() -> None:
    """同一图片的五个占位符仅发射一次，其余四处标记为已提供。"""
    message = _msg("", mid="m", iid="5c01", data=_b64(8))
    text = " ".join(["[图片(5c01)]"] * 5)

    content = inline_message_images(text, [message], max_images=3)

    assert len(_emitted(content)) == 1
    assert "".join(_texts(content)).count("[图片(同图,上文已提供)]") == 4


def test_repeated_placeholder_keeps_description() -> None:
    """重复图片的已提供标记仍保留原占位符中的描述。"""
    message = _msg("", mid="m", iid="6d01", data=_b64(8))
    text = "[图片(6d01):猫] 再看 [图片(6d01):猫]"

    content = inline_message_images(text, [message], max_images=3)

    assert len(_emitted(content)) == 1
    assert "[图片(同图,上文已提供):猫]" in _texts(content)


def test_same_message_id_in_history_and_unread_emits_once() -> None:
    """历史与本轮消息具有相同消息标识时，图片只发射一次。"""
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
    """空消息标识不会使两条包含不同图片的消息被合并。"""
    first = _msg("[图片(8f01)]", mid="", iid="8f01", data=_b64(8))
    second = _msg("[图片(9a01)]", mid="", iid="9a01", data=_b64(8))

    content = inline_message_images(
        _joined([first, second]), [first, second], max_images=3
    )

    assert len(_emitted(content)) == 2


def test_empty_unread_falls_back_to_history_ordering() -> None:
    """未指定本轮消息时按历史新旧排序，保留最新图片而省略最旧图片。"""
    history = [
        _msg(f"[图片(a{i:02x})]", mid=f"h{i}", iid=f"a{i:02x}", data=_b64(8))
        for i in range(4)
    ]

    content = inline_message_images(_joined(history), history)

    text = "".join(_texts(content))
    assert "[图片(a03)]" in text
    assert "[图片(a00)]" not in text


# --------------------------------------------------------------------------
# 字节预算
# --------------------------------------------------------------------------


def test_byte_budget_boundary_is_exact() -> None:
    """边界必须精确到**原始字节**，不能用 base64 长度凑。

    预算口径是原始图片字节，故 ``both`` 取 96×2；``tight`` 少 1 字节即容不下第二张。
    """
    first = _msg("[图片(b101)]", mid="m1", iid="b101", data=_b64(96))
    second = _msg("[图片(b202)]", mid="m2", iid="b202", data=_b64(96))
    messages = [first, second]
    both = 96 * 2

    loose = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=both
    )
    tight = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=both - 1
    )

    assert len(_emitted(loose)) == 2
    assert len(_emitted(tight)) == 1


def test_first_image_is_always_allowed_within_tight_budget() -> None:
    """八字节首图在十字节预算下正常内联。"""
    message = _msg("[图片(c303)]", mid="m", iid="c303", data=_b64(8))

    content = inline_message_images(
        _joined([message]), [message], max_images=1, max_bytes=10
    )

    assert len(_emitted(content)) == 1


def test_single_image_ceiling_applies_to_first_image() -> None:
    """即使累计预算充足，首张图片超过单图上限也会被省略。"""
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
    """字节预算为零时不限制累计体积，三张图片均可内联。"""
    messages = [
        _msg(f"[图片(f{i:02x})]", mid=f"m{i}", iid=f"f{i:02x}", data=_b64(600))
        for i in range(3)
    ]

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=0
    )

    assert len(_emitted(content)) == 3


# --------------------------------------------------------------------------
# 占位符标记
# --------------------------------------------------------------------------


def test_trimmed_history_placeholder_is_marked_omitted() -> None:
    """历史图片被截断后占位符改为省略标记，并移除其附带描述。"""
    history = [
        _msg(f"[图片(1{i:03x}):猫]", mid=f"h{i}", iid=f"1{i:03x}", data=_b64(8))
        for i in range(4)
    ]

    content = inline_message_images(_joined(history), history)

    text = "".join(_texts(content))
    assert "[图片已省略]" in text
    assert ":猫]" not in text.split("[图片已省略]")[0][-10:]


def test_placeholder_with_hex_id_but_no_data_is_marked_omitted() -> None:
    """十六进制图片标识没有可用载荷时改为省略标记，不发射图片。"""
    message = _msg("[图片(2a01)]", mid="m", iid="2a01", data="")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert "[图片已省略]" in _texts(content)
    assert not _emitted(content)


def test_desc_only_placeholder_is_marked_omitted() -> None:
    """仅含描述且无对应媒体的占位符改为省略标记。"""
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
    """裸占位符可以领取未被显式引用的图片，且不产生省略标记。"""
    message = _msg("[图片]", mid="m", iid="4d01", data=_b64(8))

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert len(_emitted(content)) == 1
    # 用拼接串而非 list：否则产出 "[图片已省略] (第2张)" 这类前缀化文本会漏判
    assert "[图片已省略]" not in "".join(_texts(content))


def test_no_placeholder_in_text_emits_nothing() -> None:
    """正文没有图片占位符时，即使消息携带图片也不内联。"""
    messages = [_msg("", mid=f"m{i}", iid=f"5e{i:02x}", data=_b64(8)) for i in range(3)]

    content = inline_message_images("这里没有任何占位符", messages, max_images=3)

    assert not _emitted(content)


# --------------------------------------------------------------------------
# 健壮性
# --------------------------------------------------------------------------


def test_invalid_base64_does_not_raise() -> None:
    """非法 base64 载荷被省略，不产生图片或向调用方抛出异常。"""
    message = _msg("[图片(6f01)]", mid="m", iid="6f01", data="!!!not-base64!!!")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert not _emitted(content)
    assert "[图片已省略]" in _texts(content)


def test_whitespace_data_does_not_produce_empty_image() -> None:
    """仅含空白的图片载荷不会生成空图片。"""
    message = _msg("[图片(7a02)]", mid="m", iid="7a02", data="   ")

    content = inline_message_images(_joined([message]), [message], max_images=3)

    assert not _emitted(content)


# --------------------------------------------------------------------------
# 预算统计
# --------------------------------------------------------------------------


def test_summarize_budget_reports_all_counters() -> None:
    """预算统计区分引用、去重、发射和丢弃数量，并按原始字节报告体积与张数归因。"""
    messages = [
        _msg(f"[图片(8{i:02x})]", mid=f"h{i}", iid=f"8{i:02x}", data=_b64(96))
        for i in range(4)
    ]
    # 走 build_image_content 才有丢弃归因证据
    content, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    assert len(_emitted(content)) == 2
    assert summary["ref_count"] == 4
    assert summary["distinct_count"] == 4
    assert summary["emitted"] == 2
    # emitted_bytes 是**原始图片字节**，不再是 base64 文本长度
    assert summary["emitted_bytes"] == 96 * 2
    assert summary["trimmed"] is True
    assert summary["over_bytes"] is False
    assert summary["should_warn"] is True
    assert summary["reason"] == "count"
    assert summary["dropped"] == {
        "count": 2,
        "bytes": 0,
        "oversize": 0,
        "placeholder_slots": 0,
    }


def test_summarize_budget_zero_byte_budget_has_no_bytes_reason() -> None:
    """关闭字节限制时，图片省略仍告警，但不报告体积超限或字节归因。"""
    messages = [
        _msg(f"[图片(9{i:02x})]", mid=f"h{i}", iid=f"9{i:02x}", data=_b64(8))
        for i in range(3)
    ]
    content = inline_message_images(
        _joined(messages), messages, max_images=2, max_bytes=0
    )

    summary = summarize_budget(content, messages, max_bytes=0)

    assert summary["should_warn"] is True
    assert summary["over_bytes"] is False
    assert "bytes" not in summary["reason"]


def test_summarize_budget_silent_when_nothing_trimmed() -> None:
    """图片全部保留时不标记截断或体积超限，也不产生告警原因。"""
    messages = [_msg("[图片(a101)]", mid="m", iid="a101", data=_b64(8))]
    content = inline_message_images(_joined(messages), messages, max_images=3)

    summary = summarize_budget(content, messages, max_bytes=1024 * 1024)

    assert summary["trimmed"] is False
    assert summary["over_bytes"] is False
    assert summary["should_warn"] is False
    assert summary["reason"] == ""


def test_count_and_bytes_buckets_can_both_be_positive() -> None:
    """``count`` 与 ``bytes`` **不是互斥的**——它们约束不同候选。

    一次遍历中：大图因累计体积被跳过后，后续小图把剩余张数名额占满，
    再后面的候选就因张数满被丢。于是两个桶同时为正，`reason=both`。

    这个用例同时钉住"``both`` 可达"这一事实——早期误以为两桶互斥、
    ``both`` 不可达而删掉该分支，会让此场景退化为只报 ``count``，
    掩盖掉"体积也拒过图"这一事实。
    """
    messages = [
        _msg("[图片(0000)]", mid="h0", iid="0000", data=_b64(96)),
        _msg("[图片(0001)]", mid="h1", iid="0001", data=_b64(96)),
        _msg(
            "[图片(0002)]", mid="h2", iid="0002", data=_b64(MAX_SINGLE_IMAGE_BYTES // 2)
        ),
        _msg("[图片(0003)]", mid="h3", iid="0003", data="这不是base64"),
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    dropped = summary["dropped"]
    assert dropped["count"] > 0
    assert dropped["bytes"] > 0
    assert summary["reason"] == "both"


def test_both_reason_requires_both_evidence() -> None:
    """只有一类证据时不得报 ``both``（对抗审查补的盲区）。

    旧实现只要"体积超标 + 任意省略"就报双超，于是一行日志里同时出现
    ``丢弃(张数)=0`` 与 ``原因=张数与体积双超`` 自相矛盾。
    """
    messages = [
        _msg("[图片(1a01)]", mid="m1", iid="1a01", data=_b64(96)),
        _msg("[图片(2b02)]", mid="m2", iid="2b02", data=_b64(96)),
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=5, max_bytes=1
    )

    assert summary["dropped"]["count"] == 0
    assert summary["reason"] == "bytes"


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
    budget = 600 * 2 + 300  # 2.5 张的原始字节预算

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=budget
    )

    # 2.5 张的预算只装得下 2 张；第三张会把累计推过上限
    assert _sizes(content) == [600, 600]
    assert sum(_sizes(content)) <= budget


def test_candidate_dedup_prevents_double_counted_budget() -> None:
    """同一 image_id 跨消息只应占用**一次**字节预算。

    若候选列表不去重，``used`` 会为同一张图累加两次，把本该装下的后续图片挤掉。
    下游的 ``intext`` 集合只在发射阶段兜住重复输出，掩盖了预算侧的漏洞。
    """
    dup = _msg("[图片(2a01)]", mid="m1", iid="2a01", data=_b64(600))
    dup2 = _msg("[图片(2a01)]", mid="m2", iid="2a01", data=_b64(600))
    other = _msg("[图片(2b02)]", mid="m3", iid="2b02", data=_b64(600))
    messages = [dup, dup2, other]
    budget = 600 * 2 + 300  # 2.5 张的原始字节预算

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=budget
    )

    # 去重后占用 600，剩 1.5 张额度，other 装得下 → 发 2 张
    assert _sizes(content) == [600, 600]
    assert sum(_sizes(content)) <= budget


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
    # 走 build_image_content 才有丢弃归因：只拿 content 时 reason 会是 unknown
    _, summary = build_image_content(
        _joined(messages),
        messages,
        preferred_messages=current,
        max_images=3,
        max_bytes=8 * 1024 * 1024,
    )

    assert summary["ref_count"] == 21
    assert summary["distinct_count"] == 16
    assert summary["emitted"] == 3
    assert summary["trimmed"] is True
    # 张数上限真的触顶且确有图因张数被丢，才允许报 count
    assert summary["reason"] == "count"
    assert summary["dropped"]["count"] > 0


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

    content = inline_message_images(_joined(messages), messages, max_images=3)

    # 第二张仍必须能内联，不能被第一张的描述吞掉
    assert 30 in _sizes(content)


# --------------------------------------------------------------------------
# 计量口径：原始图片字节（而非 base64 文本长度）
# --------------------------------------------------------------------------


def test_raw_bytes_is_exact_not_floor_approximation() -> None:
    """折算必须**精确**，不能用 ``len*3//4`` 近似。

    ``b64encode`` 会补位，``len*3//4`` 对 3,500,000 B 的图会算出 3,500,001。
    多算的 1~2 字节会让"刚好等于预算"的图被误判超限。
    """
    for size in (1, 2, 3, 5, 7, 11, 96, 600, 180_000, 3_500_000, 9_000_000):
        assert _raw_bytes(_b64(size)) == size, f"size={size}"
        assert _raw_bytes(f"base64|{_b64(size)}") == size, f"size={size}"


def test_raw_bytes_ignores_prefix_and_whitespace() -> None:
    """``data:`` / ``base64|`` 前缀与空白都不应计入原始体积。"""
    raw = _b64(120)
    assert _raw_bytes(raw) == 120
    assert _raw_bytes(f"base64|{raw}") == 120
    assert _raw_bytes(f"data:image/png;base64,{raw}") == 120
    assert _raw_bytes("\n".join(raw[i : i + 64] for i in range(0, len(raw), 64))) == 120


def test_raw_bytes_empty_payload_is_zero() -> None:
    """空载荷及仅含 base64 前缀的载荷均计为零字节。"""
    assert _raw_bytes("") == 0
    assert _raw_bytes("base64|") == 0


def test_two_phone_photos_fit_default_budget() -> None:
    """回归事故：两张 3.5MB 手机照（原始 7MB）必须都能内联。

    旧实现按 base64 长度计量时是 9.33MB > 8MB，第二张被无声丢掉。
    """
    messages = [
        _msg(
            f"[图片({iid:04x})]",
            mid=f"m{i}",
            iid=f"{iid:04x}",
            data="base64|" + _b64(3_500_000),
        )
        for i, iid in enumerate((0x1111, 0x2222))
    ]

    content = inline_message_images(
        _joined(messages), messages, max_images=3, max_bytes=DEFAULT_MAX_IMAGE_BYTES
    )

    assert len(_emitted(content)) == 2
    # 用拼接串而非 list：否则产出 "[图片已省略] (第2张)" 这类前缀化文本会漏判
    assert "[图片已省略]" not in "".join(_texts(content))


def test_nine_mib_image_is_not_rejected_by_single_ceiling() -> None:
    """单张硬上限按**原始字节**计：原始 9 MiB 的图不该被 base64 膨胀顶掉。

    旧实现按 base64 长度比 12 MiB，等效原始上限被压到 9 MiB：
    ``12 MiB * 3/4 = 9 MiB``。原始体积一旦超过 9 MiB（约 9.44MB 十进制），
    base64 化后就越过 12 MiB 而被丢弃——尽管原始图并未超过配置宣称的 12MB。

    用 10 MiB 原始体积构造，是旧实现必杀、新实现必放行的区间。
    """
    raw_size = 10 * 1024 * 1024
    # 旧口径下它必然越界，新口径下必须放行
    assert len(_b64(raw_size)) > MAX_SINGLE_IMAGE_BYTES

    message = _msg("[图片(3333)]", mid="m", iid="3333", data="base64|" + _b64(raw_size))

    content = inline_message_images(
        _joined([message]),
        [message],
        max_images=3,
        max_bytes=64 * 1024 * 1024,
    )

    assert len(_emitted(content)) == 1
    assert _sizes(content) == [raw_size]


def test_single_ceiling_still_rejects_oversized_raw_image() -> None:
    """原始体积真的超 12MB 时仍必须被单张上限挡下。"""
    message = _msg(
        "[图片(4444)]", mid="m", iid="4444", data=_b64(MAX_SINGLE_IMAGE_BYTES + 1)
    )

    content = inline_message_images(
        _joined([message]),
        [message],
        max_images=3,
        max_bytes=256 * 1024 * 1024,
    )

    assert not _emitted(content)


# --------------------------------------------------------------------------
# 归因如实化
# --------------------------------------------------------------------------


def test_byte_budget_does_not_report_count_reason() -> None:
    """因**字节**丢图时 reason 必须是 ``bytes``，不能谎报 ``count``。

    这是旧实现最误导的地方：张数远未触顶，日志却写"张数超限"。
    """
    messages = [
        _msg(
            f"[图片({iid:04x})]",
            mid=f"m{i}",
            iid=f"{iid:04x}",
            data="base64|" + _b64(3_500_000),
        )
        for i, iid in enumerate((0x5555, 0x6666, 0x7777))
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=8 * 1024 * 1024
    )

    assert summary["emitted"] == 2
    assert summary["dropped"] == {
        "count": 0,
        "bytes": 1,
        "oversize": 0,
        "placeholder_slots": 0,
    }
    assert summary["reason"] == "bytes"


def test_count_reason_requires_evidence_of_count_cap() -> None:
    """只有"张数确实触顶且确有图因张数被丢"才允许报 ``count``。"""
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"m{i}", iid=f"{iid:04x}", data=_b64(64))
        for i, iid in enumerate((0x8111, 0x8222, 0x8333, 0x8444, 0x8555))
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    assert summary["emitted"] == 2
    assert summary["dropped"]["count"] == 3
    assert summary["reason"] == "count"


def test_single_ceiling_boundary_accepts_exactly_at_limit() -> None:
    """恰好等于单张上限的图必须**放行**（变异测试补的盲区）。

    既有用例只测了 ``MAX+1``（应拦）与 10 MiB（应放行），没有任何用例钉住
    ``== MAX`` 这一侧。把 ``>`` 改成 ``>=`` 时全部测试仍然全绿，而实际行为
    已经改变：12 MiB 的图会被错误丢弃。
    """
    message = _msg(
        "[图片(cc01)]", mid="m", iid="cc01", data=_b64(MAX_SINGLE_IMAGE_BYTES)
    )

    content = inline_message_images(
        _joined([message]),
        [message],
        max_images=1,
        max_bytes=MAX_SINGLE_IMAGE_BYTES + 1,
    )

    assert len(_emitted(content)) == 1
    assert _sizes(content) == [MAX_SINGLE_IMAGE_BYTES]


def test_single_ceiling_boundary_rejects_one_byte_over() -> None:
    """超过上限 1 字节的图必须被拦，与 :func:`test_single_ceiling_boundary_accepts_exactly_at_limit` 构成闭区间。"""
    message = _msg(
        "[图片(cc02)]", mid="m", iid="cc02", data=_b64(MAX_SINGLE_IMAGE_BYTES + 1)
    )

    content = inline_message_images(
        _joined([message]),
        [message],
        max_images=1,
        max_bytes=MAX_SINGLE_IMAGE_BYTES * 2,
    )

    assert not _emitted(content)
    assert "[图片已省略]" in "".join(_texts(content))


def test_placeholder_slot_exhaustion_is_attributed() -> None:
    """裸占位符槽位耗尽必须单独归因，而不是落进 ``unknown``（对抗审查补的盲区）。

    这是最高频的真实死因：正文只有 1 个 ``[图片]``，却有 3 张可用图，
    即使张数与体积都宽裕，多出来的 2 张也无处安放。旧实现直接 ``break``
    且不计数，使这类省略全部退化成"原因不明"。
    """
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"m{i}", iid=f"{iid:04x}", data=_b64(64))
        for i, iid in enumerate((0xA001, 0xA002, 0xA003))
    ]

    _, summary = build_image_content(
        "[图片]", messages, max_images=3, max_bytes=1024 * 1024
    )

    assert summary["emitted"] == 1
    assert summary["dropped"]["placeholder_slots"] == 2
    assert summary["reason"] == "placeholder_slots"
    assert summary["should_warn"] is True


def test_no_placeholder_at_all_is_attributed_to_slots() -> None:
    """正文完全没有占位符时，多余候选同样计入槽位桶。"""
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"m{i}", iid=f"{iid:04x}", data=_b64(64))
        for i, iid in enumerate((0xB001, 0xB002))
    ]

    content, summary = build_image_content(
        "这里没有任何占位符", messages, max_images=5, max_bytes=1024 * 1024
    )

    assert not [item for item in content if isinstance(item, Image)]
    assert summary["emitted"] == 0
    assert summary["dropped"]["placeholder_slots"] == 2
    assert summary["reason"] == "placeholder_slots"


def test_zero_max_images_reports_count_not_unknown() -> None:
    """``max_images=0`` 的正确归因是 ``count``，不得退化成 ``unknown``（对抗审查补的盲区）。

    旧实现在该分支手工构造全零 ``dropped``，把唯一的张数证据自己抹掉了，
    属于诊断能力回退。
    """
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"m{i}", iid=f"{iid:04x}", data=_b64(64))
        for i, iid in enumerate((0xC001, 0xC002, 0xC003))
    ]

    content, summary = build_image_content(_joined(messages), messages, max_images=0)

    assert content == [Text(_joined(messages))]
    assert summary["emitted"] == 0
    assert summary["dropped"]["count"] == 3
    assert summary["reason"] == "count"


def test_count_attribution_survives_invalid_payload() -> None:
    """载荷非法被拒后，张数证据仍须成立（对抗审查补的盲区）。

    旧判据用 ``len(emitted) >= max_images`` 判张数触顶，而发射阶段会因
    载荷非法被拒掉一张，使 ``emitted`` 低于 ``max_images``——张数证据就此丢失。
    判据必须改用 ``dropped["count"]``。
    """
    bad = _msg("[图片(d001)]", mid="m1", iid="d001", data="这不是base64")
    good = _msg("[图片(d002)]", mid="m2", iid="d002", data=_b64(64))
    good2 = _msg("[图片(d003)]", mid="m3", iid="d003", data=_b64(64))
    messages = [bad, good, good2]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    # 非法载荷在发射阶段被拒，故实际内联数少于选中数（3 候选里只内联 2 个可用）
    assert summary["dropped"]["count"] > 0
    assert summary["reason"] == "count"
    assert len(_emitted(content)) == 2


def test_oversize_first_image_still_lets_later_image_through() -> None:
    """首张超限时必须**继续往后选**，不能 ``break`` 提前退出（变异测试补的盲区）。

    ``oversize`` 分支用 ``continue`` 而非 ``break``：一张 20 MiB 的图被单张
    上限拦下后，后面的正常图仍应被选中。若改成 ``break``，``emitted`` 会从 1
    变成 0，而只断言 ``dropped["oversize"]`` 的用例察觉不到。
    """
    huge = _msg(
        "[图片(dd01)]", mid="m1", iid="dd01", data=_b64(MAX_SINGLE_IMAGE_BYTES + 1)
    )
    normal = _msg("[图片(dd02)]", mid="m2", iid="dd02", data=_b64(64))
    messages = [normal, huge]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    # 超限的被跳过，后面那张正常图必须补上
    assert len(_emitted(content)) == 1
    assert summary["emitted"] == 1
    assert _sizes(content) == [64]
    assert summary["dropped"]["oversize"] == 1


def test_single_oversize_image_reports_oversize_reason() -> None:
    """因单张硬上限丢图必须单独归因为 ``oversize``。"""
    messages = [
        _msg(
            "[图片(9111)]", mid="m1", iid="9111", data=_b64(MAX_SINGLE_IMAGE_BYTES + 1)
        ),
        _msg("[图片(9222)]", mid="m2", iid="9222", data=_b64(64)),
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=256 * 1024 * 1024
    )

    assert summary["dropped"]["oversize"] == 1
    assert summary["reason"] == "oversize"


def test_first_image_exemption_does_not_fire_false_over_bytes_alarm() -> None:
    """首张豁免累计预算时不该报"体积超限"。

    单张 9MB > 默认 8MB 预算属既有设计（首张豁免），且没有省略任何图。
    若 ``over_bytes`` 只看体积，就会打出 ``should_warn=True`` 却 ``reason=""``
    的自相矛盾告警。
    """
    message = _msg(
        "[图片(aa11)]", mid="m", iid="aa11", data="base64|" + _b64(9_000_000)
    )

    content, summary = build_image_content(
        _joined([message]), [message], max_images=3, max_bytes=8 * 1024 * 1024
    )

    assert len(_emitted(content)) == 1
    assert summary["over_bytes"] is False
    assert summary["should_warn"] is False
    assert summary["reason"] == ""


def test_reason_is_unknown_when_no_budget_evidence() -> None:
    """有图被省略却无任何预算证据时，必须说 ``unknown`` 而不是猜 ``count``。

    这里刻意用 ``max_images=1`` 制造省略，但不把 ``max_images`` 与 ``dropped``
    传给 ``summarize_budget``——它没有证据区分"张数截断"与其它原因，
    只能如实说 ``unknown``。
    """
    messages = [
        _msg("[图片(b111)]", mid="m1", iid="b111", data=_b64(64)),
        _msg("[图片(b222)]", mid="m2", iid="b222", data=_b64(64)),
    ]
    content = inline_message_images(
        _joined(messages), messages, max_images=1, max_bytes=1024 * 1024
    )

    # 不提供 max_images / dropped 时无法证明张数上限被触及
    summary = summarize_budget(content, messages, max_bytes=1024 * 1024)

    assert len(_emitted(content)) == 1
    assert summary["trimmed"] is True
    assert summary["should_warn"] is True
    assert summary["reason"] == "unknown"


# --------------------------------------------------------------------------
# 静默路径可见化
# --------------------------------------------------------------------------


def test_missing_image_id_is_counted_and_warned() -> None:
    """缺 ``image_id`` 的图片此前被静默省略且零告警——必须可见。"""
    messages = [
        Message(
            message_id=f"m{i}",
            content={"text": "[图片]", "media": [{"type": "image", "data": _b64(64)}]},
            processed_plain_text="[图片]",
            sender_id="user",
            chat_type="group",
        )
        for i in range(2)
    ]

    content, summary = build_image_content(
        "[图片]\n[图片]", messages, max_images=3, max_bytes=1024 * 1024
    )

    assert not _emitted(content)
    assert summary["unusable"]["missing_image_id"] == 2
    assert summary["should_warn"] is True
    assert summary["reason"] == "unusable"


def test_missing_data_is_counted_and_warned() -> None:
    """历史图 ``data`` 回填失败（磁盘已清理）此前静默省略——必须可见。"""
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"h{i}", iid=f"{iid:04x}", data=None)
        for i, iid in enumerate((0xC111, 0xC222))
    ]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=1024 * 1024
    )

    assert not _emitted(content)
    assert summary["unusable"]["missing_data"] == 2
    assert summary["should_warn"] is True
    assert summary["reason"] == "unusable"


def test_unusable_media_summary_ignores_non_image_media() -> None:
    """语音 / 视频 / 非媒体项不计入"图片载荷不可用"。"""
    messages = [
        Message(
            message_id="m0",
            content={
                "text": "[语音]",
                "media": [
                    {"type": "voice", "voice_id": "d1"},
                    {"type": "file", "file_id": "f1"},
                    {"type": "image", "data": _b64(64)},
                ],
            },
            processed_plain_text="[语音]",
            sender_id="user",
            chat_type="group",
        )
    ]

    summary = unusable_media_summary(messages)

    assert summary == {"missing_image_id": 1, "missing_data": 0}


def test_usable_media_does_not_trip_unusable_warning() -> None:
    """一切正常的图片不得误报载荷不可用。"""
    messages = [
        _msg(f"[图片({iid:04x})]", mid=f"m{i}", iid=f"{iid:04x}", data=_b64(64))
        for i, iid in enumerate((0xD111, 0xD222))
    ]

    _, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=1024 * 1024
    )

    assert summary["unusable"] == {"missing_image_id": 0, "missing_data": 0}
    assert summary["should_warn"] is False


def test_raw_bytes_never_negative_for_degenerate_payload() -> None:
    """畸形载荷不得让体积折算为负——否则累计预算会倒退。

    全由 ``=`` 组成的主体会让 ``len*3//4 - padding`` 为负。负体积进入
    ``used`` 累加器等于凭空放大预算，因此必须钳到 0。
    """
    for body in ("=", "==", "===", "====", "=" * 1000):
        assert _raw_bytes(body) >= 0, f"body={body[:16]!r}"
        assert _raw_bytes(f"base64|{body}") >= 0


def test_raw_bytes_clamps_before_accumulating_budget() -> None:
    """负体积不得使 ``used`` 倒退——端到端验证预算闸门不被绕过。

    退化载荷（全 ``=``，修前折算出负数）若未被钳制，就会让 ``used`` 减少，
    等于凭空释放预算。这里放一张退化载荷并**把它的体积算作 0** 的最坏情况：
    两张 1000 字节的图在 1500 字节预算下仍只应进 1 张。若负体积生效让
    ``used`` 倒退到负，第二张就会被错误放行。
    """
    junk = _msg("[图片(0111)]", mid="m0", iid="0111", data="base64|====")
    real = _msg("[图片(0222)]", mid="m1", iid="0222", data=_b64(1000))
    real2 = _msg("[图片(0333)]", mid="m2", iid="0333", data=_b64(1000))
    messages = [real, real2, junk]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=1500
    )

    # 首张 1000 通过；第二张会让累计 2000 > 1500，应被字节闸门拦下
    assert len(_emitted(content)) == 1
    assert summary["emitted_bytes"] == 1000
    assert summary["dropped"]["bytes"] >= 1


def test_payload_body_matches_image_normalization() -> None:
    """``_payload_body`` 必须与 ``File`` 的归一化逐字对齐。

    若这里比 ``Image`` 剥得更宽（含 TAB/VT），``_safe_image`` 的前置校验
    就会把非法字符"洗掉"后误判为合法，校验形同虚设。
    """
    import base64 as _b64mod

    std = _b64mod.b64encode(b"\x89PNG" + b"\x00" * 20).decode()

    # Image 不认的空白必须被保留，让字符集校验如实拒绝
    for ch in ("\t", "\x0b"):
        dirty = std[:10] + ch + std[10:]
        assert _safe_image("k" * 64, {"k" * 64: {"data": dirty}}) is None, repr(ch)

    # Image 认的空白必须被剥掉
    for ch in ("\n", "\r", " "):
        dirty = std[:10] + ch + std[10:]
        assert _safe_image("k" * 64, {"k" * 64: {"data": dirty}}) is not None, repr(ch)


def test_payload_body_only_splits_on_base64_comma() -> None:
    """``data:`` 前缀只在含 ``base64,`` 时才切，不得按首个逗号盲切。"""
    std = "iVBORw0KGgo="
    # 没有 "base64," 时不应切分
    assert _payload_body("data:image/png;base64") == "data:image/png;base64"
    # 有 "base64," 时按它切
    assert _payload_body(f"data:image/png;base64,{std}") == std


def test_invalid_payload_is_attributed_not_unknown() -> None:
    """载荷非法（非 base64）必须单独归因，而不是落到 ``unknown``。

    这类项**有** ``image_id`` 也**有** ``data``，预算侧四个桶全为 0，
    修前只能报 ``unknown``，与真正的"归因不了"混为一谈。
    """
    bad = _msg("[图片(0111)]", mid="m1", iid="0111", data="这不是base64")
    good = _msg("[图片(0222)]", mid="m2", iid="0222", data=_b64(64))
    messages = [bad, good]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=3, max_bytes=1024 * 1024
    )

    assert len(_emitted(content)) == 1
    assert summary["invalid_payloads"] == 1
    assert summary["reason"] == "invalid_payload"
    assert summary["should_warn"] is True


def test_invalid_payload_counter_is_scoped_per_call() -> None:
    """拒绝计数必须按调用取差值，不能跨请求累积。"""
    bad = _msg("[图片(1111)]", mid="m1", iid="1111", data="这不是base64")
    good = _msg("[图片(2222)]", mid="m2", iid="2222", data=_b64(64))

    _, first = build_image_content(
        _joined([bad, good]), [bad, good], max_images=3, max_bytes=1024 * 1024
    )
    _, second = build_image_content(
        _joined([good]), [good], max_images=3, max_bytes=1024 * 1024
    )

    assert first["invalid_payloads"] == 1
    # 第二次调用没有非法载荷，计数必须归零
    assert second["invalid_payloads"] == 0
    assert second["should_warn"] is False


def test_invalid_payload_beats_unknown_when_both_present() -> None:
    """载荷非法优先于 unknown 归因——它是有明确成因的。

    注意构造：必须让非法载荷**进入发射阶段**（``max_images`` 足够大）才会触发
    ``_safe_image`` 的拒绝。若名额不足，非法载荷会在选择阶段就被挤掉，
    ``_safe_image`` 根本不会被调用，归因也就落在 ``count`` 上——那同样是正确的。
    """
    bad = _msg("[图片(3111)]", mid="m1", iid="3111", data="这不是base64")
    other = _msg("[图片(3222)]", mid="m2", iid="3222", data=_b64(64))
    messages = [bad, other]

    content, summary = build_image_content(
        _joined(messages), messages, max_images=2, max_bytes=1024 * 1024
    )

    assert len(_emitted(content)) == 1
    assert summary["invalid_payloads"] == 1
    assert summary["reason"] == "invalid_payload"


def test_invalid_payload_never_reports_unknown() -> None:
    """载荷非法时 ``reason`` 不得是 ``unknown``——它有明确成因。"""
    for max_images in (2, 3):
        bad = _msg("[图片(3111)]", mid="m1", iid="3111", data="这不是base64")
        other = _msg("[图片(3222)]", mid="m2", iid="3222", data=_b64(64))
        messages = [bad, other]

        _, summary = build_image_content(
            _joined(messages),
            messages,
            max_images=max_images,
            max_bytes=1024 * 1024,
        )

        assert summary["reason"] != "unknown", f"max_images={max_images}"


def test_build_image_content_zero_limit_is_plain_text() -> None:
    """``max_images <= 0`` 时退化为纯文本，但仍如实汇报载荷不可用。"""
    message = Message(
        message_id="m0",
        content={"text": "[图片]", "media": [{"type": "image", "data": _b64(64)}]},
        processed_plain_text="[图片]",
        sender_id="user",
        chat_type="group",
    )

    content, summary = build_image_content("[图片]", [message], max_images=0)

    assert content == [Text("[图片]")]
    assert summary["emitted"] == 0
    assert summary["unusable"]["missing_image_id"] == 1
    assert summary["should_warn"] is True

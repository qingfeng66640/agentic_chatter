"""Agentic Chatter 主 Agent 的原生图片内容构建。"""

from __future__ import annotations

import binascii
import re
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import Message
from src.kernel.llm import Content, Image, Text

logger = get_logger("agentic_chatter")

#: 单张图片 base64 的硬上限（字节）。任何图片超过它都会被舍弃，
#: 包括第一张——首张只豁免累计预算，不豁免本上限。
MAX_SINGLE_IMAGE_BYTES = 12 * 1024 * 1024

#: 单次请求图片 base64 的默认总预算（字节）。``0`` 表示不按体积限制。
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: 带 media_id 的占位符（``[图片(hex)]`` / ``[图片(hex):描述]``）与裸占位符（``[图片]``）。
#: id 字符集刻意保持为十六进制：``converter`` 产出的 ``image_id`` 恒为 sha256，
#: 放宽只会让嵌套文本把 id 污染。
#:
#: 描述部分禁止出现 ``[`` 与 ``]``。描述是 VLM 自由文本，若允许 ``]``，则
#: ``[图片(a1):a]b[图片(cd34)]`` 会在第一个 ``]`` 处截断并把后面的
#: ``[图片(cd34)]`` 一并吞掉，导致**预算内、本该内联的图凭空消失**。
#: 禁止方括号后，这类文本退化为"整段不匹配、原样透传"，不会破坏结构。
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[图片(?:\(([0-9a-fA-F]+)\)(?::[^\[\]]*)?)?\]")

#: 无 media_id 的 ``[图片:描述]`` 形态（``converter.py`` 在有描述但无 media_id 时产出）。
#: 要求「图片」后紧跟「:」加至少一个非空白非右括号字符，因此**不会**误匹配
#: ``[图片]``、``[图片(ab12):描述]``、``[图片已省略]``、``[图片(同图,上文已提供):描述]``。
_IMAGE_DESC_ONLY_PATTERN = re.compile(r"\[图片:[^\]\s]+\]")

#: base64 字符集（含补位符）。用于在构造 ``Image`` 之前先确认载荷确实是 base64，
#: 而不是文件路径——``Image`` 遇到无法解码的字符串会**当成路径读盘**并编码进请求体。
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")

_OMITTED_TEXT = "[图片已省略]"


def _media_items(message: Message) -> list[dict[str, Any]]:
    """读取消息中带**可用**图片载荷的媒体项。

    这里要求 ``data`` 是非空白的字符串，使引用计数、去重计数与预算选择
    三者口径完全一致——否则会出现"计入候选却永远选不上"的永久误告警。
    """
    candidates: Any = None
    if isinstance(getattr(message, "content", None), dict):
        candidates = message.content.get("media")
    if not isinstance(candidates, list) and isinstance(
        getattr(message, "extra", None), dict
    ):
        candidates = message.extra.get("media")
    if not isinstance(candidates, list):
        return []
    return [
        item
        for item in candidates
        if isinstance(item, dict)
        and item.get("type") == "image"
        and isinstance(item.get("data"), str)
        and item["data"].strip()
    ]


def _image_ids(message: Message) -> list[str]:
    """返回该消息中可用图片的 id 列表（与 :func:`_media_items` 同源）。"""
    return [
        item["image_id"]
        for item in _media_items(message)
        if isinstance(item.get("image_id"), str) and item["image_id"]
    ]


def _scan_placeholders(text: str) -> tuple[set[str], int]:
    """扫描正文占位符，返回 ``(被显式引用的 image_id 集合, 裸占位符个数)``。"""
    referenced: set[str] = set()
    bare = 0
    for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(text):
        image_id = match.group(1)
        if image_id:
            referenced.add(image_id)
        else:
            bare += 1
    return referenced, bare


def _collect_candidates(
    messages: list[Message],
    preferred_messages: list[Message] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """按「新→旧」返回按 ``image_id`` 去重的图片候选。

    排序契约：``chat_stream.context.history_messages`` 为旧→新
    （``stream_manager.py`` 以 ``order_by("-id")`` + ``reversed`` 装配，
    ``models/stream.py`` 超限时保留尾部），因此**尾部才是最新**。

    输入约定：``messages`` 已是 ``history + unread``，而 ``preferred_messages``
    就是 ``unread``。去重胜出方与输出顺序必须解耦——

    * 去重优先保留**本轮**实例：本轮消息刚到达、``data`` 一定完整；
      历史副本的 ``data`` 依赖磁盘回填，回填失败时就没有 ``data``。
    * 输出顺序要求本轮组排在新→旧序列的**前面**。

    Args:
        messages: 本轮参与决策的全部消息（历史 + 未读）。
        preferred_messages: 本轮新输入消息，优先级高于历史。

    Returns:
        按新→旧排序、``image_id`` 去重后的 ``(image_id, media_item)`` 列表。
    """
    preferred = list(preferred_messages or [])
    # 按对象身份排除，避免依赖 Message 是否定义 __eq__
    history = [
        message
        for message in messages
        if not any(message is item for item in preferred)
    ]

    # 1) 去重：本轮优先。message_id 为空时退化为按对象身份，不参与去重。
    seen_ids: set[str] = set()
    kept_preferred: list[Message] = []
    kept_history: list[Message] = []
    for bucket, sink in ((preferred, kept_preferred), (history, kept_history)):
        for message in bucket:
            message_id = getattr(message, "message_id", None)
            if message_id:
                if message_id in seen_ids:
                    continue
                seen_ids.add(message_id)
            sink.append(message)

    # 2) 排序：本轮组倒序在前，历史组倒序在后；组内均为新→旧。
    #    image_id 去重必须发生在这里——重复候选会让字节预算被重复计量。
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen_image_ids: set[str] = set()
    ordered = list(reversed(kept_preferred)) + list(reversed(kept_history))
    for message in ordered:
        for item in _media_items(message):
            image_id = item.get("image_id")
            if not isinstance(image_id, str) or not image_id or image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            candidates.append((image_id, item))
    return candidates


def _select_within_budget(
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    id_referenced: set[str],
    bare_slots: int,
    max_images: int,
    max_bytes: int,
) -> list[str]:
    """按预算挑选可内联的图片。

    预算语义：``max_images`` 为张数上限、``max_bytes`` 为 base64 字节上限，
    ``max_bytes <= 0`` 表示不按体积限制。**首张豁免累计预算**（避免预算偏小时
    模型完全看不到图），但任何图都受 :data:`MAX_SINGLE_IMAGE_BYTES` 约束。

    因此实际总量上界为::

        min(max_images * MAX_SINGLE_IMAGE_BYTES,
            (max_bytes if max_bytes > 0 else inf) + MAX_SINGLE_IMAGE_BYTES)

    只有"正文里确实有位置用到"的候选才占用预算：被显式引用的图优先，
    未被引用的图仅在还有裸占位符可认领时才计入。否则正文里根本没有占位符的
    历史图会白白吃掉名额，导致真正要发的图被挤掉。

    Args:
        candidates: 按新→旧排序的候选。
        id_referenced: 正文中被显式引用的 image_id 集合。
        bare_slots: 正文中的裸占位符个数。
        max_images: 张数上限。
        max_bytes: 字节上限，``<= 0`` 表示不限制。

    Returns:
        入选的 ``image_id`` 列表；被引用的在前，其余按新→旧在后。
    """
    if max_bytes <= 0:
        max_bytes = float("inf")

    referenced = [item for item in candidates if item[0] in id_referenced]
    unreferenced = [item for item in candidates if item[0] not in id_referenced]

    shortlist: list[str] = []
    used = 0

    for pool, slots in ((referenced, None), (unreferenced, min(bare_slots, max_images))):
        remaining_slots = slots
        for image_id, item in pool:
            if len(shortlist) >= max_images:
                break
            if remaining_slots is not None:
                if remaining_slots <= 0:
                    break
                remaining_slots -= 1
            data = item["data"]
            size = len(data)
            if size > MAX_SINGLE_IMAGE_BYTES:
                continue  # 单张天花板，对首张同样生效
            if shortlist and used + size > max_bytes:
                continue  # 累计预算不足：跳过这张，继续用后面的填额度
            shortlist.append(image_id)
            used += size

    return shortlist


def _payload_body(data: str) -> str:
    """剥掉 ``data:`` / ``base64|`` 前缀与空白，只留载荷主体。"""
    payload = data
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    elif payload.startswith("base64|"):
        payload = payload[len("base64|"):]
    return "".join(payload.split())


def _safe_image(
    image_id: str,
    items_by_id: dict[str, dict[str, Any]],
) -> Image | None:
    """构造 :class:`Image`；载荷非法时降级为 ``None``。

    两道防线：

    1. **先确认是 base64**。:class:`Image` 遇到无法解码的字符串不会抛错，而是
       当成**文件路径**读盘并 base64 编码——那会把任意本地文件塞进请求体。
    2. **再兜住异常**。构造时仍会对 base64 做严格校验，失败时降级而不是炸掉回合。

    校验放在**发射时**而非选择阶段：选择阶段校验需对全部候选完整解码，
    而实际只会发射 ``max_images`` 张。
    """
    item = items_by_id.get(image_id)
    if item is None:
        return None
    data = item.get("data")
    if not isinstance(data, str):
        return None
    body = _payload_body(data)
    if not body or len(body) % 4 != 0 or not _BASE64_PATTERN.match(body):
        logger.warning(
            "多模态图片载荷不是 base64，已省略 "
            f"事件=多模态图片载荷非法 图片标识={image_id}"
        )
        return None
    try:
        return Image(data)
    except (TypeError, ValueError, binascii.Error) as exc:
        logger.warning(
            "多模态图片载荷非法，已省略 "
            f"事件=多模态图片载荷非法 图片标识={image_id} 异常类型={type(exc).__name__}"
        )
        return None


def _append_source_text(content: list[Content], chunk: str) -> None:
    """追加**来自原文**的文本片段，并改写无 media_id 的 ``[图片:描述]`` 形态。

    只对原文片段调用。程序自己生成的 :data:`_OMITTED_TEXT` 与
    ``[图片(同图,上文已提供)…]`` 不经过这里，因此不会被二次改写。
    """
    if chunk:
        content.append(Text(_IMAGE_DESC_ONLY_PATTERN.sub(_OMITTED_TEXT, chunk)))


def _resolve_placeholder(
    text: str,
    shortlist: list[str],
    items_by_id: dict[str, dict[str, Any]],
    id_referenced: set[str],
    bare_slots: int,
) -> list[Content]:
    """按占位符位置内联 Text/Image，并为被舍弃的占位符改写标记。

    Args:
        text: 含占位符的正文。
        shortlist: 入选的 ``image_id`` 列表。
        items_by_id: ``image_id`` 到 media 项的映射。
        id_referenced: 正文中被显式引用的 image_id 集合。
        bare_slots: 正文中的裸占位符个数。

    Returns:
        交错排列的 ``Text`` / ``Image`` 内容列表。
    """
    # 裸占位符只能认领「入选且未被带 id 占位符引用」的图。
    # 正文行序是「历史→本轮」，候选序是「本轮→历史」，二者互为镜像；
    # 因此从正文**最后一个**裸占位符开始反向分配，让本轮行优先拿到最新的图。
    bare_pool = [image_id for image_id in shortlist if image_id not in id_referenced]
    bare_offsets = [
        match.start()
        for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(text)
        if match.group(1) is None
    ]
    bare_assign: dict[int, str | None] = {}
    pool_index = 0
    for offset in reversed(bare_offsets[:bare_slots]):
        bare_assign[offset] = (
            bare_pool[pool_index] if pool_index < len(bare_pool) else None
        )
        pool_index += 1

    content: list[Content] = []
    intext: set[str] = set()
    cursor = 0

    for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(text):
        _append_source_text(content, text[cursor:match.start()])
        image_id = match.group(1)

        if image_id is None:
            target = bare_assign.get(match.start())
            image = _safe_image(target, items_by_id) if target else None
            if image is not None and target is not None:
                content.append(Text(match.group(0)))
                content.append(image)
                intext.add(target)
            else:
                content.append(Text(_OMITTED_TEXT))
        elif image_id in intext:
            desc = match.group(0)[len(f"[图片({image_id})"):]
            content.append(Text(f"[图片(同图,上文已提供){desc}"))
        else:
            image = (
                _safe_image(image_id, items_by_id) if image_id in shortlist else None
            )
            if image is not None:
                content.append(Text(match.group(0)))
                content.append(image)
                intext.add(image_id)
            else:
                content.append(Text(_OMITTED_TEXT))

        cursor = match.end()

    _append_source_text(content, text[cursor:])
    if not content:
        content.append(Text(text))
    return content


def summarize_budget(
    user_content: list[Content],
    messages: list[Message],
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """统计图片预算使用情况。

    抽成纯函数，便于在不启动整个回合管线的前提下单测告警逻辑。

    ``max_bytes <= 0`` 时"不按体积限制"，故 ``over_bytes`` 恒为假。

    Args:
        user_content: :func:`inline_message_images` 的输出。
        messages: 本轮参与决策的全部消息。
        max_bytes: 配置的字节上限，``<= 0`` 表示不限制。

    Returns:
        含 ``ref_count`` / ``distinct_count`` / ``emitted`` / ``emitted_bytes``
        / ``trimmed`` / ``over_bytes`` / ``should_warn`` / ``reason`` 的字典。
    """
    emitted = [item for item in user_content if isinstance(item, Image)]
    emitted_bytes = sum(len(item.value) for item in emitted)
    ref_count = image_count(messages)
    distinct_count = len(
        {image_id for message in messages for image_id in _image_ids(message)}
    )

    trimmed = len(emitted) < distinct_count
    over_bytes = max_bytes > 0 and emitted_bytes > max_bytes
    if trimmed and over_bytes:
        reason = "both"
    elif trimmed:
        reason = "count"
    elif over_bytes:
        reason = "bytes"
    else:
        reason = ""

    return {
        "ref_count": ref_count,
        "distinct_count": distinct_count,
        "emitted": len(emitted),
        "emitted_bytes": emitted_bytes,
        "trimmed": trimmed,
        "over_bytes": over_bytes,
        "should_warn": trimmed or over_bytes,
        "reason": reason,
    }


def inline_message_images(
    text: str,
    messages: list[Message],
    *,
    preferred_messages: list[Message] | None = None,
    max_images: int = 3,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> list[Content]:
    """将消息图片按占位符位置内联为 Text/Image 内容。

    ``max_images`` 与 ``max_bytes`` 是**单次请求的总预算**，本轮新图与历史图共享。
    按「本轮优先、统一新→旧」选取；同一张图只内联一次。

    预算不足而被舍弃的图片，其占位符改写为 ``[图片已省略]``；同一张图重复
    出现时改写为 ``[图片(同图,上文已提供):描述]`` 并保留描述。

    Args:
        text: 含占位符的正文。
        messages: 本轮参与决策的全部消息（历史 + 未读）。
        preferred_messages: 本轮新输入消息，优先级高于历史。
        max_images: 单次请求图片总数上限，``<= 0`` 表示不传入任何图片。
        max_bytes: 单次请求图片 base64 总体积上限，``<= 0`` 表示不按体积限制。

    Returns:
        交错排列的 ``Text`` / ``Image`` 内容列表。
    """
    if max_images <= 0:
        return [Text(text)]

    id_referenced, bare_slots = _scan_placeholders(text)
    preferred = list(preferred_messages or [])
    candidates = _collect_candidates(messages, preferred)
    shortlist = _select_within_budget(
        candidates,
        id_referenced=id_referenced,
        bare_slots=bare_slots,
        max_images=max_images,
        max_bytes=max_bytes,
    )
    return _resolve_placeholder(
        text,
        shortlist,
        {image_id: item for image_id, item in candidates},
        id_referenced,
        bare_slots,
    )


def image_count(messages: list[Message]) -> int:
    """统计消息中带可用图片载荷的媒体项数量。"""
    return sum(len(_media_items(message)) for message in messages)

"""Agentic Chatter 主 Agent 的原生图片内容构建。"""

from __future__ import annotations

import binascii
import re
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import Message
from src.kernel.llm import Content, Image, Text

logger = get_logger("agentic_chatter")

#: 单张图片**原始字节**的硬上限（字节）。任何图片超过它都会被舍弃，
#: 包括第一张——首张只豁免累计预算，不豁免本上限。
#:
#: 计量口径是**原始图片字节**，不是 base64 文本长度。base64 会把二进制膨胀到
#: 4/3 倍，若按 base64 长度比 12MB，原始 9MB 的高清图就会被误判为超限
#: （9MB × 4/3 = 12MB > 12MB），而 3.5MB 的手机随手拍 base64 后有 4.7MB，
#: 8MB 预算只够 1 张——这正是"没到张数上限却被拦"的成因。
MAX_SINGLE_IMAGE_BYTES = 12 * 1024 * 1024

#: 单次请求图片**原始字节**的默认总预算。``0`` 表示不按体积限制。
#:
#: 按原始字节计，8MB 预算可容纳 2 张手机随手拍（3.5MB）或一批表情包/截图。
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: base64 载荷长度 → 原始字节数的折算分子（len(body) * 3 // 4）。
#:
#: 用整数除法而非浮点，避免边界上出现 8,388,607.5 这类中间值。
_BASE64_RAW_NUMERATOR = 3
_BASE64_RAW_DENOMINATOR = 4

#: 带 media_id 的占位符（``[图片(hex)]`` / ``[图片(hex):描述]``）与裸占位符（``[图片]``）。
#: id 字符集刻意保持为十六进制：``converter`` 产出的 ``image_id`` 恒为 sha256，
#: 放宽只会让嵌套文本把 id 污染。
#:
#: 描述部分禁止出现 ``[`` 与 ``]``。描述是 VLM 自由文本，若允许 ``]``，则
#: ``[图片(a1):a]b[图片(cd34)]`` 会在第一个 ``]`` 处截断并把后面的
#: ``[图片(cd34)]`` 一并吞掉，导致**预算内、本该内联的图凭空消失**。
#: 禁止方括号后，这类文本退化为"整段不匹配、原样透传"，不会破坏结构。
_IMAGE_PLACEHOLDER_PATTERN = re.compile(
    r"\[图片(?:\(([0-9a-fA-F]+)\)(?::[^\[\]]*)?)?\]"
)

#: 无 media_id 的 ``[图片:描述]`` 形态（``converter.py`` 在有描述但无 media_id 时产出）。
#: 要求「图片」后紧跟「:」加至少一个非空白非右括号字符，因此**不会**误匹配
#: ``[图片]``、``[图片(ab12):描述]``、``[图片已省略]``、``[图片(同图,上文已提供):描述]``。
_IMAGE_DESC_ONLY_PATTERN = re.compile(r"\[图片:[^\]\s]+\]")

#: base64 字符集（含补位符）。用于在构造 ``Image`` 之前先确认载荷确实是 base64，
#: 而不是文件路径——``Image`` 遇到无法解码的字符串会**当成路径读盘**并编码进请求体。
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")

_OMITTED_TEXT = "[图片已省略]"

#: 发射阶段被 :func:`_safe_image` 拒绝的载荷计数（按原因分桶）。
#:
#: 这类项**有** ``image_id`` 也**有** ``data``，因此既进不了
#: :func:`_unusable_media`（那两类缺项），也进不了预算侧的 ``dropped`` 桶，
#: 结果只会表现为"有图被省略但四个桶全为 0"，只能报 ``unknown``。
#: 用模块级计数器在 :func:`build_image_content` 前后取差值即可把它补进归因。
#:
#: 计数在单次事件循环内自增自减，无跨请求共享状态。
_REJECTED_PAYLOADS: dict[str, int] = {"invalid": 0}


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


def _unusable_media(message: Message) -> dict[str, int]:
    """统计该消息中**无法内联**的图片媒体项及其原因。

    这两类项既进不了 :func:`_media_items`（缺 ``data``），也进不了候选列表
    （缺 ``image_id``），因此既不计入 :func:`image_count` 也不计入预算统计，
    会让"图片被省略"这件事**完全静默**——正文里已经写了 ``[图片已省略]``，
    日志却一个字都没有。必须单独计量才能让它可见。

    Args:
        message: 待检查的消息。

    Returns:
        含 ``missing_image_id`` 与 ``missing_data`` 的字典，值均为个数。
    """
    result = {"missing_image_id": 0, "missing_data": 0}
    candidates: Any = None
    if isinstance(getattr(message, "content", None), dict):
        candidates = message.content.get("media")
    if not isinstance(candidates, list) and isinstance(
        getattr(message, "extra", None), dict
    ):
        candidates = message.extra.get("media")
    if not isinstance(candidates, list):
        return result

    for item in candidates:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        image_id = item.get("image_id")
        data = item.get("data")
        has_data = isinstance(data, str) and bool(data.strip())
        has_id = isinstance(image_id, str) and bool(image_id)
        if not has_id:
            result["missing_image_id"] += 1
        elif not has_data:
            result["missing_data"] += 1
    return result


def unusable_media_summary(messages: list[Message]) -> dict[str, int]:
    """汇总全部消息中无法内联的图片媒体项。

    Args:
        messages: 本轮参与决策的全部消息。

    Returns:
        含 ``missing_image_id`` 与 ``missing_data`` 的字典。
    """
    total = {"missing_image_id": 0, "missing_data": 0}
    for message in messages:
        for key, value in _unusable_media(message).items():
            total[key] += value
    return total


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
            if (
                not isinstance(image_id, str)
                or not image_id
                or image_id in seen_image_ids
            ):
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
) -> tuple[list[str], dict[str, int]]:
    """按预算挑选可内联的图片。

    预算语义：``max_images`` 为张数上限、``max_bytes`` 为**原始图片字节**上限，
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
        max_bytes: 原始图片字节上限，``<= 0`` 表示不限制。

    Returns:
        ``(入选的 image_id 列表, 丢弃归因计数)``。归因计数字段为

        ``count``
            因张数上限被丢弃；
        ``bytes``
            因累计字节预算被丢弃；
        ``oversize``
            因单张原始体积硬上限被丢弃；
        ``placeholder_slots``
            正文里没有占位符可用（裸占位符槽位耗尽），因而即使预算宽裕也
            没有位置放。旧实现对此直接 ``break`` 且不计数，使这类省略
            全部落进 ``unknown``——而它是最高频的真实死因。

        前三者互斥；四者之和不等于候选总数，因为槽位耗尽会提前终止遍历。
    """
    if max_bytes <= 0:
        max_bytes = float("inf")

    referenced = [item for item in candidates if item[0] in id_referenced]
    unreferenced = [item for item in candidates if item[0] not in id_referenced]

    shortlist: list[str] = []
    dropped = {"count": 0, "bytes": 0, "oversize": 0, "placeholder_slots": 0}
    used = 0

    for pool, slots in (
        (referenced, None),
        (unreferenced, min(bare_slots, max_images)),
    ):
        remaining_slots = slots
        for index, (image_id, item) in enumerate(pool):
            if remaining_slots is not None and remaining_slots <= 0:
                # 裸占位符槽位耗尽：本池剩余候选都无处安放，计入独立桶
                dropped["placeholder_slots"] += len(pool) - index
                break
            if len(shortlist) >= max_images:
                dropped["count"] += 1
                continue
            if remaining_slots is not None:
                remaining_slots -= 1
            data = item["data"]
            size = _raw_bytes(data)
            if size > MAX_SINGLE_IMAGE_BYTES:
                dropped["oversize"] += 1
                continue  # 单张天花板，对首张同样生效
            if shortlist and used + size > max_bytes:
                dropped["bytes"] += 1
                continue  # 累计预算不足：跳过这张，继续用后面的填额度
            shortlist.append(image_id)
            used += size

    return shortlist, dropped


def _payload_body(data: str) -> str:
    """按 :class:`~src.kernel.llm.payload.content.File` 的口径归一化载荷。

    **必须与 ``File`` 的归一化逐字对齐**，否则 :func:`_safe_image` 的前置校验
    会被喂进一个"它以为合法、实际 ``Image`` 会拒"的 body，校验形同虚设：

    * 前缀判定用 ``"base64," in s``（含逗号才切），不是按首个逗号切；
    * ``base64|`` 判定是**独立** if，不与 ``data:`` 互斥；
    * 只剥 ``\\n`` / ``\\r`` / 空格三种空白——``\\t`` / ``\\x0b`` 等
      ``Image`` 不认的字符必须**保留**，让字符集校验如实拒绝它。

    Args:
        data: 媒体项的原始 ``data`` 字段。

    Returns:
        归一化后的 base64 主体；不符合任何已知形态时原样返回。
    """
    payload = data
    if payload.startswith("data:") and "base64," in payload:
        payload = payload.split("base64,", 1)[1]
    if payload.startswith("base64|"):
        payload = payload[len("base64|") :]
    return payload.replace("\n", "").replace("\r", "").replace(" ", "")


def _raw_bytes(data: str) -> int:
    """精确还原该载荷的**原始图片字节数**。

    base64 每 4 个字符编码 3 字节，但末尾的 ``=`` 是补位符、不携带数据。
    因此原始长度不是 ``len(body) * 3 / 4``——对带补位的载荷那样算会
    **多算 1~2 字节**（如 3,500,000 B 的图折成 3,500,001 B）。

    这里先按 :func:`_payload_body` 取主体，再减去补位符贡献::

        raw = len(body) * 3 // 4 - padding

    ``padding`` 由末尾 ``=`` 个数给出，恰好等于补位多算的字节数。

    **下界钳制为 0**：畸形载荷（如全由 ``=`` 组成）会让上式算出负数。
    负体积会让累计预算 ``used`` 倒退，等于凭空放大预算，因此必须钳到 0。
    这类载荷本就会被 :func:`_safe_image` 的长度与字符集校验拒绝，钳制只是
    保证预算计算本身不出现负数。

    Args:
        data: 媒体项的原始 ``data`` 字段，可能带 ``data:`` / ``base64|`` 前缀。

    Returns:
        原始图片字节数，不小于 0。
    """
    body = _payload_body(data)
    if not body:
        return 0
    # 末尾补位符不携带数据，须从折算值里扣掉
    padding = len(body) - len(body.rstrip("="))
    return max(
        0, len(body) * _BASE64_RAW_NUMERATOR // _BASE64_RAW_DENOMINATOR - padding
    )


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

    拒绝计数记入 :data:`_REJECTED_PAYLOADS`，由 :func:`build_image_content`
    在调用前后取差值汇入统计——否则这类省略只留下一条孤立日志，
    在预算侧完全归因不出来（``reason`` 只能报 ``unknown``）。
    """
    item = items_by_id.get(image_id)
    if item is None:
        return None
    data = item.get("data")
    if not isinstance(data, str):
        return None
    body = _payload_body(data)
    if not body or len(body) % 4 != 0 or not _BASE64_PATTERN.match(body):
        _REJECTED_PAYLOADS["invalid"] += 1
        logger.warning(
            "多模态图片载荷不是 base64，已省略 "
            f"事件=多模态图片载荷非法 图片标识={image_id}"
        )
        return None
    try:
        return Image(data)
    except (TypeError, ValueError, binascii.Error) as exc:
        _REJECTED_PAYLOADS["invalid"] += 1
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
        _append_source_text(content, text[cursor : match.start()])
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
            desc = match.group(0)[len(f"[图片({image_id})") :]
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
    dropped: dict[str, int] | None = None,
    invalid_payloads: int = 0,
) -> dict[str, Any]:
    """统计图片预算使用情况与**真实**丢弃原因。

    抽成纯函数，便于在不启动整个回合管线的前提下单测告警逻辑。

    ``max_bytes <= 0`` 时"不按体积限制"，故 ``over_bytes`` 恒为假。

    ``emitted_bytes`` 是**原始图片字节**总和（由 :func:`_raw_bytes` 精确折算，
    无近似误差），与 :data:`DEFAULT_MAX_IMAGE_BYTES` 的口径一致。

    Args:
        user_content: :func:`inline_message_images` 的输出。
        messages: 本轮参与决策的全部消息。
        max_bytes: 配置的字节上限，``<= 0`` 表示不限制。
        dropped: :func:`_select_within_budget` 返回的丢弃归因计数。
        invalid_payloads: 发射阶段被 :func:`_safe_image` 拒绝的载荷数。

    Returns:
        含 ``ref_count`` / ``distinct_count`` / ``emitted`` / ``emitted_bytes``
        / ``trimmed`` / ``over_bytes`` / ``should_warn`` / ``reason``
        / ``dropped`` / ``unusable`` / ``invalid_payloads`` 的字典。

        ``reason`` 只在**证据支持**时给出具体值：张数维度要求
        ``dropped["count"] > 0``（确有图因张数上限被丢弃），体积维度取自
        ``dropped``。无法归因时为 ``"unknown"`` 而**不是**谎称 ``count``——
        旧实现把"字节不够"一律报成"张数超限"，把排查方向完全带偏。
    """
    emitted = [item for item in user_content if isinstance(item, Image)]
    emitted_bytes = sum(_raw_bytes(item.value) for item in emitted)
    ref_count = image_count(messages)
    distinct_count = len(
        {image_id for message in messages for image_id in _image_ids(message)}
    )

    unusable = unusable_media_summary(messages)
    dropped_counts = (
        dict(dropped)
        if dropped
        else {"count": 0, "bytes": 0, "oversize": 0, "placeholder_slots": 0}
    )

    trimmed = len(emitted) < distinct_count
    # 张数触顶判据用**选中数**而非实际发射数：发射阶段还可能被
    # ``_safe_image`` 拒掉非法载荷，若用 ``emitted`` 判定，发射数会因
    # 载荷问题而低于 ``max_images``，把"因张数被丢"误报成无法归因。
    count_capped = dropped_counts["count"] > 0
    # 首张豁免累计预算：单张图可以超过 max_bytes 而仍然内联。此时体积确实
    # 超标，但**没有任何图被省略**，若据此置 should_warn 就会打出
    # "should_warn=True 却 reason=''" 的自相矛盾告警。故 over_bytes 额外要求
    # 确有图被舍弃（trimmed 或单张超限），避免首张豁免被误报为超限事件。
    over_bytes = (
        max_bytes > 0
        and emitted_bytes > max_bytes
        and (trimmed or dropped_counts["oversize"] > 0)
    )

    # ``both`` 需**同时**具备两类证据。仅凭"体积超标 + 任意省略"就报双超，
    # 会在同一行日志里出现 ``丢弃(张数)=0`` 与 ``原因=张数与体积双超``
    # 自相矛盾，把排查方向引向"调大 max_input_images"这一无效动作。
    #
    # 注意 ``count`` 与 ``bytes`` 并非互斥：它们约束的是**不同候选**。一次遍历
    # 中某个候选因张数满被丢（count），另一个候选因体积不足被丢（bytes），
    # 两者可以同时为正——例如"大图因体积被跳过后，后续候选把张数名额占满"。
    both_capped = count_capped and dropped_counts["bytes"] > 0

    if both_capped:
        reason = "both"
    elif count_capped:
        reason = "count"
    elif dropped_counts["oversize"]:
        reason = "oversize"
    elif dropped_counts["bytes"]:
        reason = "bytes"
    elif dropped_counts["placeholder_slots"]:
        reason = "placeholder_slots"
    elif invalid_payloads:
        reason = "invalid_payload"
    elif unusable["missing_image_id"] or unusable["missing_data"]:
        reason = "unusable"
    elif trimmed:
        # 有图被省略，但没有任何预算证据支撑——必须如实说"不知道"
        reason = "unknown"
    else:
        reason = ""

    should_warn = (
        trimmed
        or over_bytes
        or invalid_payloads > 0
        or dropped_counts["placeholder_slots"] > 0
        or unusable["missing_image_id"] > 0
        or unusable["missing_data"] > 0
    )

    return {
        "ref_count": ref_count,
        "distinct_count": distinct_count,
        "emitted": len(emitted),
        "emitted_bytes": emitted_bytes,
        "trimmed": trimmed,
        "over_bytes": over_bytes,
        "should_warn": should_warn,
        "reason": reason,
        "dropped": dropped_counts,
        "unusable": unusable,
        "invalid_payloads": invalid_payloads,
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
        max_bytes: 单次请求图片**原始字节**上限，``<= 0`` 表示不按体积限制。

    Returns:
        交错排列的 ``Text`` / ``Image`` 内容列表。
    """
    if max_images <= 0:
        return [Text(text)]

    id_referenced, bare_slots = _scan_placeholders(text)
    preferred = list(preferred_messages or [])
    candidates = _collect_candidates(messages, preferred)
    shortlist, _dropped = _select_within_budget(
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


def build_image_content(
    text: str,
    messages: list[Message],
    *,
    preferred_messages: list[Message] | None = None,
    max_images: int = 3,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> tuple[list[Content], dict[str, Any]]:
    """内联图片并同时返回预算统计。

    这是调用侧真正需要的形态：只有内容列表时无法区分"因张数上限被丢"
    与"因字节预算被丢"，日志就会给出错误的排查方向。

    Args:
        text: 含占位符的正文。
        messages: 本轮参与决策的全部消息（历史 + 未读）。
        preferred_messages: 本轮新输入消息，优先级高于历史。
        max_images: 单次请求图片总数上限，``<= 0`` 表示不传入任何图片。
        max_bytes: 单次请求图片**原始字节**上限，``<= 0`` 表示不限制。

    Returns:
        ``(内容列表, summarize_budget 结果)``。
    """
    if max_images <= 0:
        content = [Text(text)]
        # 该分支一张都不发，所有可用图片都因"张数上限为 0"被舍弃。
        # 必须把张数证据补回去，否则会退化成 unknown，丢失"上限为 0"
        # 这一明确成因。该分支直接返回文本，不经过 _safe_image，无发射阶段拒绝。
        available = len(
            {image_id for message in messages for image_id in _image_ids(message)}
        )
        dropped = {
            "count": available,
            "bytes": 0,
            "oversize": 0,
            "placeholder_slots": 0,
        }
        summary = summarize_budget(
            content,
            messages,
            max_bytes=max_bytes,
            dropped=dropped,
            invalid_payloads=0,
        )
        return content, summary

    id_referenced, bare_slots = _scan_placeholders(text)
    preferred = list(preferred_messages or [])
    candidates = _collect_candidates(messages, preferred)
    shortlist, dropped = _select_within_budget(
        candidates,
        id_referenced=id_referenced,
        bare_slots=bare_slots,
        max_images=max_images,
        max_bytes=max_bytes,
    )
    # 发射阶段会因载荷非法拒绝图片，取计数器差值把它并入归因
    rejected_before = _REJECTED_PAYLOADS["invalid"]
    content = _resolve_placeholder(
        text,
        shortlist,
        {image_id: item for image_id, item in candidates},
        id_referenced,
        bare_slots,
    )
    invalid_payloads = _REJECTED_PAYLOADS["invalid"] - rejected_before
    summary = summarize_budget(
        content,
        messages,
        max_bytes=max_bytes,
        dropped=dropped,
        invalid_payloads=invalid_payloads,
    )
    return content, summary


def image_count(messages: list[Message]) -> int:
    """统计消息中带可用图片载荷的媒体项数量。"""
    return sum(len(_media_items(message)) for message in messages)

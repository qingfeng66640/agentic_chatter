"""不可拆单元识别。

分句时不能按字符位置硬切：颜文字与 emoji 序列被切开会变成无意义的碎片，
例如 ``👨‍👩‍👧`` 被切成 ``👨`` 与 ``👩‍👧``，在聊天平台上会渲染成两个表情。

本模块把文本中不可拆的最小单位识别出来，供分段器把切分点对齐到这些边界。

两类单元：

- **表情簇**：完整 grapheme cluster，涵盖肤色修饰、ZWJ 组合、变体选择符、
  keycap 与 tag 序列（如旗帜）。
- **颜文字**：由成对括号或波浪号包裹的"表情面"，以及 ``QAQ`` 这类网络缩写。
  判定依据是面内含组合符号（``Sk``/``Sm``/``So``/``Mn`` 等），因此普通的
  ``(注)``、``(abc)``、``[1]`` 不会被误判。

依赖 ``regex`` 提供 ``\\X`` 与 Unicode 属性；缺失时自动降级为手写状态机，
准确率略低但保证功能不中断。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

try:  # pragma: no cover - 由运行环境决定
    import regex as _regex
except ImportError:  # pragma: no cover - 降级路径
    _regex = None

import re as _re

# 组合记号：附着在前一字符上，不应成为独立切分点
_COMBINING = "̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃰"
# 变体选择符、keycap 组合键、ZWJ
_VARIATION = "︎️⃣‍"
# 区域指示符与 tag 序列（旗帜、国旗）
_TAG = "\U000E0020-\U000E007F"

_MODIFIERS = _COMBINING + _VARIATION
_CLUSTER_BODY = "[^" + _MODIFIERS + _TAG + "]"

# 优先使用 regex 的 \X（完整 grapheme cluster）；
# 降级路径用手写状态机，规则等价于「基字符 + 若干修饰符 + 若干 ZWJ 连接段」。
_GRAPHEME_RE = (
    _regex.compile(r"\X") if _regex is not None else _re.compile(
        "(?:" + _CLUSTER_BODY + "[" + _MODIFIERS + "]*(?:‍" + _CLUSTER_BODY
        + "[" + _MODIFIERS + "]*)*)*"
    )
)

# emoji 主判定。\X 已足够完成切分，这里仅用于"是否含表情"的诊断与统计。
# keycap 序列（🄞🄞🄞）的基础字符不在 Extended_Pictographic 内，需单独覆盖。
_EMOJI_RE = (
    _regex.compile(r"\p{Extended_Pictographic}|[⃣\U0001F100-\U0001F1FF]" + _TAG)
    if _regex is not None
    else _re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️⃣]")
)

# emoji 基字符：用于判断当前字符是否应交给 \X 聚合。
# 注意 ``＃``/``*`` 在无变体选择符时不是 emoji，故只认带 U+FE0F 的组合由 \X 处理。
_EMOJI_BASE_RE = (
    _regex.compile(r"\p{Extended_Pictographic}|[\U0001F100-\U0001F1FF" + _TAG + "⃣]")
    if _regex is not None
    else _re.compile(
        "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️⃣\U0001F100-\U0001F1FF]"
    )
)

# keycap 序列（1️⃣、#️⃣）的基字符是 ASCII 数字/井号/星号，
# 仅当其后紧跟变体选择符或 U+20E3 时才算 emoji。
_KEYCAP_TAIL_RE = (
    _regex.compile("[" + _VARIATION + "]")
    if _regex is not None
    else _re.compile("[" + _VARIATION + "]")
)

# 括号与波浪号：颜文字的包裹符号
_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    "（": "）",
    "［": "］",
    "｛": "｝",
    "【": "】",
    "〔": "〕",
    "〖": "〗",
}
_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}
_WAVES = "~〜～"

# 中点类字符：颜文字常用作眼珠或分隔，Unicode 类别为 Po
_MIDDLE_DOTS = frozenset({"·", "・", "•", "･"})
# 裸颜文字的包裹符：在正文里是顿号，只在"中点对称"形态下才按颜文字处理
_BARE_WRAPPERS = _MIDDLE_DOTS | {"、"}

# 网络缩写颜文字：全大写或全小写的字母数字组合。
# 不能用 \b 分界：Python 的 re 把汉字也视为单词字符，\bQAQ\b 在
# "甲QAQ乙" 上不成立，因此改用前后瞻排除相邻的拉丁字母与数字。
_LATIN_EMOTICON = _re.compile(
    r"(?<![A-Za-z0-9])(?:QAQ|TAT|OTZ|orz|Orz|ORZ|XD|xd|xD|TT|tt"
    r"|233|2333|555|HHH|hhh)(?![A-Za-z0-9])"
)

# 单元长度上限：超过则不按颜文字处理，避免把整段正文误判成一个单元
_MAX_FACE_LEN = 24
# 括号包裹的面允许的最大嵌套深度，足够覆盖 (<(=･ω･=)>) 与 ((((*´ω`*) )))) 这类堆叠写法
_MAX_NEST = 6


@dataclass(frozen=True)
class Unit:
    """一个不可拆单元。

    Attributes:
        start: 在原文中的起始下标（含）。
        end: 在原文中的结束下标（不含）。
        text: 单元文本。
        kind: 单元类别，``emoji`` 或 ``kaomoji``。
    """

    start: int
    end: int
    text: str
    kind: str


def _is_face_char(char: str) -> bool:
    """判断字符是否属于颜文字"面"的构成。

    颜文字的面由嘴形符号、修饰记号、几何图形组成，同时大量使用半角
    中点、小写片假名浊音符、阿拉伯字母这类在正常正文里极少连排的字符。
    判定同时看 Unicode 类别与码点区间，两者都不落入时才视为非面字符。

    这是区分 ``(´･ω･`)`` 与 ``(注)`` 的关键：后者内部全是汉字或拉丁
    字母，不含任何面字符。
    """
    category = unicodedata.category(char)
    if category in {"Sk", "Sm", "So", "Lm"}:
        return True
    # 组合记号（重音等）随基字符出现，单独出现也算面的一部分
    if category == "Mn" or unicodedata.combining(char):
        return True
    # 泰文数字：颜文字的经典眼珠
    if "\u0e50" <= char <= "\u0e59":
        return True
    # 中点类：(´･ω･`) [･ω･] {・ω・}
    if char in _MIDDLE_DOTS:
        return True
    # 小写假名与片假名浊音符：(っ・ω・っ)、(＾▽＾)ヽ
    if "\u3041" <= char <= "\u309f" or "\u30a1" <= char <= "\u30ff":
        return True
    # 希腊字母：(°ω°) (-ω-) (＊´ω｀＊)
    if "\u0370" <= char <= "\u03ff":
        return True
    # 阿拉伯字母、谚文字母：و ـ 等
    if "\u0600" <= char <= "\u06ff":
        return True
    # 彝文、西里尔补充等表情常用区段：(꒰ᐢ꒱)
    if "\ua000" <= char <= "\ua4cf" or "\u1400" <= char <= "\u167f":
        return True
    # 连接符类泪纹：(:﹏﹏:)
    if category == "Pc":
        return True
    return False


def _starts_keycap(text: str, position: int) -> bool:
    """判断 ``text[position]`` 是否是 keycap 序列的基字符。

    keycap 形如 ``1️⃣``、``#️⃣``，基字符是 ASCII 数字、井号或星号，
    不属于 Extended_Pictographic，必须单独识别。

    Args:
        text: 原文。
        position: 候选基字符下标。

    Returns:
        bool: 该位置开启一个 keycap 序列时为 ``True``。
    """
    if position >= len(text) or text[position] not in "0123456789#*":
        return False
    tail = text[position + 1 : position + 2]
    return tail != "" and _KEYCAP_TAIL_RE.match(tail) is not None


def _is_strong_face_char(char: str) -> bool:
    """判断字符是否为颜文字的标志性构件。

    颜文字的面必须含有明确的"五官"：嘴型修饰符、几何眼形、泰文眼珠、
    或希腊字母这类极少出现在中文正文里的字符。仅有符号（如 ``(1+2)``
    里的 ``+``）或仅有普通文字都不足以判定为颜文字。

    Args:
        char: 待判定字符。

    Returns:
        bool: 属于强面字符时为 ``True``。
    """
    category = unicodedata.category(char)
    codepoint = ord(char)

    # 嘴型与语调修饰符：(´ω`) (｀ω´) (＾▽＾) (*´ω`*)
    if category == "Sk":
        return True
    # 几何眼形与口部：(°ω°) (╹◡╹) (◠‿◠) (▽ω▽)
    if 0x25a0 <= codepoint <= 0x25ff or 0x2500 <= codepoint <= 0x257f:
        return True
    # 泰文眼珠：(๑•̀ㅂ•́)
    if 0x0e50 <= codepoint <= 0x0e59:
        return True
    # 希腊字母：颜文字最常见的嘴型 ω
    if 0x0370 <= codepoint <= 0x03ff:
        return True
    # 方言音符、彝文等表格文字
    if 0xa000 <= codepoint <= 0xa4cf:
        return True
    # 组合记号：(´◡` ) 中的重音符
    if category == "Mn" or unicodedata.combining(char):
        return True
    # 假名浊音符与小写假名：(っ・ω・っ)
    if 0x3041 <= codepoint <= 0x309f or 0x30a1 <= codepoint <= 0x30ff:
        return True
    # 连接符类泪纹：(:﹏﹏:) 里的 ﹏
    if category == "Pc":
        return True
    # 彝文与加拿大音节文字：(꒰ᐢ꒱) 的眼形部件
    if 0xa000 <= codepoint <= 0xa4cf or 0x1400 <= codepoint <= 0x167f:
        return True
    return False


def _has_face_signature(inner: str) -> bool:
    """判断括号内容是否具备颜文字的"五官"特征。

    单独的符号不足以判定——``(1+2)`` 与 ``(a+b)`` 也含符号。
    真正的颜文字至少要有一个眼嘴部件（修饰符、几何形、希腊字母等），
    或出现中点这类只在颜文字里成串出现的字符。

    Args:
        inner: 括号内部的文本。

    Returns:
        bool: 具备颜文字特征时为 ``True``。
    """
    for char in inner:
        category = unicodedata.category(char)
        codepoint = ord(char)
        # 明确的眼嘴部件
        if category == "Sk":
            return True
        if 0x25a0 <= codepoint <= 0x25ff or 0x2500 <= codepoint <= 0x257f:
            return True
        if 0x0e50 <= codepoint <= 0x0e59 or 0x0370 <= codepoint <= 0x03ff:
            return True
        if category == "Mn" or unicodedata.combining(char):
            return True
        if 0x3041 <= codepoint <= 0x309f or 0x30a1 <= codepoint <= 0x30ff:
            return True
        if 0xa000 <= codepoint <= 0xa4cf or 0x1400 <= codepoint <= 0x167f:
            return True
        # 连接符类泪纹：(:﹏﹏:)
        if category == "Pc":
            return True
    # 无眼嘴件时，要求至少两个中点，符合 [･ω･]、{・ω・} 的形态
    dots = sum(1 for char in inner if char in _MIDDLE_DOTS)
    return dots >= 2


def _scan_bracketed(text: str, start: int) -> int | None:
    """从 ``text[start]`` 处的括号开始，返回配对括号后的下标。

    仅当括号内部构成"面"时才视为颜文字；否则返回 ``None``。支持
    ``(<(=･ω･=)>)`` 这类嵌套写法。

    Args:
        text: 原文。
        start: 开括号所在下标。

    Returns:
        int | None: 闭合括号之后的下标；非颜文字时为 ``None``。
    """
    opener = text[start]
    closer = _BRACKET_PAIRS[opener]
    stack = [closer]
    index = start + 1
    face = False
    content = 0

    while index < len(text) and stack:
        char = text[index]
        if char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                break
        elif len(stack) > _MAX_NEST:
            # 嵌套过深，已不像颜文字，提前放弃
            return None
        else:
            content += 1
            if _is_face_char(char):
                face = True
        index += 1

    if stack or not face:
        return None

    inner = text[start + 1 : index]
    # 全是普通文字时不构成颜文字，如 (注)、(abc)、(123)、[1]、<div>
    if not content or len(inner) > _MAX_FACE_LEN:
        return None
    # 必须含颜文字特征，否则 (a+b)、(1+2) 这类普通括号内容会被误判
    if not _has_face_signature(inner):
        return None
    return index + 1


def _find_inner_wave(text: str, bracket: int, limit: int) -> int | None:
    """定位括号内收尾波浪号所在的整体结束位置。

    ``〜(´ω｀〜)`` 这种写法把收尾波浪号放在括号内部，括号因内含波浪号
    而无法配对。找到该波浪号后，连同其后的闭括号一起作为结束位置。

    Args:
        text: 原文。
        bracket: 开括号下标。
        limit: 搜索上界（含）。

    Returns:
        int | None: 整体结束下标；未找到时为 ``None``。
    """
    for offset in range(bracket + 1, min(len(text), limit + 1)):
        if text[offset] not in _WAVES:
            continue
        tail = offset + 1
        if tail < len(text) and text[tail] in _CLOSE_TO_OPEN:
            tail += 1
        return tail
    return None


def _scan_waved(text: str, start: int) -> int | None:
    """扫描 ``~`` / ``〜`` / ``～`` 包裹的颜文字。

    波浪号自身不成对使用，因此以"波浪号 + 表情面 + 波浪号"为整体。

    Args:
        text: 原文。
        start: 起始波浪号下标。

    Returns:
        int | None: 结束波浪号之后的下标；不构成颜文字时为 ``None``。
    """
    closer_offset = None
    cursor = start + 1
    limit = min(len(text), start + _MAX_FACE_LEN + 2)
    # 波浪号可能落在括号内（〜(´ω｀〜)），也可能落在括号外（~(´ω｀)~）。
    # 因此先按普通字符前进，遇到括号则跳到配对括号之后，两种写法都能覆盖。
    while cursor < limit:
        char = text[cursor]
        if char in _BRACKET_PAIRS:
            # 〜(´ω｀〜)：收尾波浪号落在括号内，整体作为一个单元
            inner_wave = _find_inner_wave(text, cursor, limit)
            if inner_wave is not None:
                return inner_wave
            nested = _scan_bracketed(text, cursor)
            if nested is not None:
                # 括号内容本身是颜文字，继续向后寻找收尾波浪号
                cursor = nested
                continue
            cursor += 1
            continue
        if char in _WAVES:
            closer_offset = cursor
            break
        cursor += 1
    if closer_offset is None or closer_offset <= start:
        return None

    inner = text[start + 1 : closer_offset]
    if not inner or len(inner) > _MAX_FACE_LEN:
        return None
    # 括号本身不参与判定：~(´ω｀)~ 的面全部在括号内
    core = [
        char
        for char in inner
        if char not in _BRACKET_PAIRS and char not in _CLOSE_TO_OPEN
    ]
    if not core:
        return None
    if not _has_face_signature("".join(core)):
        return None
    return closer_offset + 1


def _extend_tail(text: str, end: int) -> int:
    """把颜文字后外挂的装饰字符并入单元。

    ``(๑•̀ㅂ•́)و✧`` 这类写法在闭括号后还挂着 ``و`` ``✧`` ``ヽ`` 等字符，
    它们与颜文字同属一个语义单位，切开同样会破坏观感。

    Args:
        text: 原文。
        end: 已识别单元的结束下标。

    Returns:
        int: 并入外挂字符后的结束下标。
    """
    index = end
    while index < len(text) and _is_face_char(text[index]):
        index += 1
    return index


def find_units(text: str) -> list[Unit]:
    """识别文本中全部不可拆单元。

    Args:
        text: 待分析文本。

    Returns:
        list[Unit]: 按起始下标升序排列的单元列表；无单元时为空列表。
    """
    if not text:
        return []

    units: list[Unit] = []
    position = 0
    length = len(text)

    while position < length:
        char = text[position]

        # 1) 表情簇：仅在当前字符本身是 emoji 基字符时交给 \X 处理。
        #    普通字符（含颜文字的括号）必须落到下面的颜文字分支，
        #    因此这里不能无条件推进 position。
        if _EMOJI_BASE_RE.match(char) is not None or _starts_keycap(text, position):
            cluster = _GRAPHEME_RE.match(text, position)
            if cluster is not None and cluster.end() > position:
                units.append(
                    Unit(position, cluster.end(), cluster.group(0), "emoji")
                )
                position = cluster.end()
                continue

        # 2) 颜文字：波浪号包裹（须先于括号判断，否则 ~(´ω｀)~ 会被截断）
        if char in _WAVES:
            end = _scan_waved(text, position)
            if end is not None:
                units.append(
                    Unit(position, end, text[position:end], "kaomoji")
                )
                position = end
                continue

        # 3) 颜文字：成对括号
        if char in _BRACKET_PAIRS:
            end = _scan_bracketed(text, position)
            if end is not None:
                end = _extend_tail(text, end)
                units.append(
                    Unit(position, end, text[position:end], "kaomoji")
                )
                position = end
                continue

        # 5) 网络缩写
        match = _LATIN_EMOTICON.match(text, position)
        if match is not None and match.end() > position:
            units.append(
                Unit(position, match.end(), match.group(0), "kaomoji")
            )
            position = match.end()
            continue

        # 6) 无括号包裹的裸颜文字：、´ω´、（´ω｀ 等以中点或颜文字构件成串出现。
        #    这类写法没有成对括号可依赖，只能靠"中点对称"来界定范围。
        bare_end = _scan_bare(text, position)
        if bare_end > position:
            units.append(Unit(position, bare_end, text[position:bare_end], "kaomoji"))
            position = bare_end
            continue

        position += 1

    return units


def _scan_bare(text: str, start: int) -> int:
    """识别无括号包裹的裸颜文字。

    典型写法是"中点包住颜文字构件"，如 ``、´ω´、``、``。´ω´。``：
    两端各有一个中点，内部由颜文字构件组成。这类写法没有成对括号，
    因此以"首个中点 + 若干面字符 + 收尾中点"界定范围。

    Args:
        text: 原文。
        start: 候选起始下标。

    Returns:
        int: 裸颜文字的结束下标；不构成时为 ``start``。
    """
    if text[start] not in _BARE_WRAPPERS:
        return start

    index = start + 1
    body = 0
    strong = False
    while index < len(text) and index - start <= _MAX_FACE_LEN:
        char = text[index]
        if char in _BARE_WRAPPERS:
            # 收尾包裹符：内部必须已有面构件，且不能紧邻起始符
            if body and strong:
                return index + 1
            return start
        if not _is_face_char(char):
            return start
        body += 1
        if _is_strong_face_char(char):
            strong = True
        index += 1

    return start


def unit_boundaries(units: list[Unit], length: int) -> set[int]:
    """把单元边界换算成"禁止作为内部切点"的下标集合。

    Args:
        units: 单元列表。
        length: 原文长度。

    Returns:
        set[int]: 落在单元内部的下标集合，集合元素均落在 ``1..length-1``。
    """
    blocked: set[int] = set()
    for unit in units:
        # start < i < end 的位置都在单元内部
        for index in range(unit.start + 1, unit.end):
            blocked.add(index)
    return blocked


def split_into_atoms(text: str, units: list[Unit]) -> list[tuple[str, bool]]:
    """把文本切成"原子"，每个原子是不可拆单元或单个普通字符。

    原子序列与原文严格一一对应，拼接后还原原文，不丢字符也不改内容。

    Args:
        text: 原文。
        units: 不可拆单元列表。

    Returns:
        list[tuple[str, bool]]: ``(文本, 是否为不可拆单元)`` 组成的序列。
    """
    atoms: list[tuple[str, bool]] = []
    position = 0
    for unit in units:
        if unit.start > position:
            atoms.extend((char, False) for char in text[position : unit.start])
        atoms.append((unit.text, True))
        position = unit.end
    if position < len(text):
        atoms.extend((char, False) for char in text[position:])
    return atoms


def reconstruct(atoms: list[tuple[str, bool]]) -> str:
    """由原子序列还原原文，供完整性自检使用。

    Args:
        atoms: ``split_into_atoms`` 的返回值。

    Returns:
        str: 还原后的文本。
    """
    return "".join(text for text, _ in atoms)


def has_regex_backend() -> bool:
    """报告当前是否在使用 ``regex`` 的完整 grapheme 分割算法。

    Returns:
        bool: ``regex`` 可用时为 ``True``，否则为 ``False``。
    """
    return _regex is not None

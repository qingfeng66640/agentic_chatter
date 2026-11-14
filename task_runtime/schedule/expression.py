"""调度表达式的解析与计算。

对外只暴露 ``parse_schedule`` 与 ``next_run_after``，内部优先使用
croniter；缺失时降级到内置的五段 cron 解析器，保证插件在未安装可选
依赖的环境下依然可用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

try:  # pragma: no cover - 由运行环境决定
    from croniter import croniter as _croniter
except ImportError:  # pragma: no cover - 降级路径
    _croniter = None


class ScheduleError(ValueError):
    """调度表达式非法。"""


def has_croniter() -> bool:
    """是否可用 croniter。"""
    return _croniter is not None


_FIELD_RANGES: tuple[tuple[int, int], ...] = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),  # day of week
)

_MONTH_ALIASES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_DOW_ALIASES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}


# Windows 与精简容器镜像可能没有 IANA 时区数据库（缺 tzdata 包）。
# 这类环境无法加载任意 IANA 时区，退回到「固定 UTC 偏移」子集：
# 既覆盖实际最常用的时区，又不静默把错误偏移当成正确结果。
_FIXED_OFFSETS: dict[str, timedelta] = {
    "UTC": timedelta(0),
    "Etc/UTC": timedelta(0),
    "Asia/Shanghai": timedelta(hours=8),
    "Asia/Chongqing": timedelta(hours=8),
    "Asia/Hong_Kong": timedelta(hours=8),
    "Asia/Macau": timedelta(hours=8),
    "Asia/Taipei": timedelta(hours=8),
    "Asia/Singapore": timedelta(hours=8),
    "Asia/Tokyo": timedelta(hours=9),
    "Asia/Seoul": timedelta(hours=9),
    "Asia/Dubai": timedelta(hours=4),
    "Asia/Kolkata": timedelta(hours=5, minutes=30),
    "Europe/London": timedelta(0),
    "Europe/Moscow": timedelta(hours=3),
    "America/New_York": timedelta(hours=-5),
    "America/Chicago": timedelta(hours=-6),
    "America/Denver": timedelta(hours=-7),
    "America/Los_Angeles": timedelta(hours=-8),
}


def _resolve_timezone(name: str | None) -> Any:
    """解析 IANA 时区名，非法时抛出 :class:`ScheduleError`。

    环境缺少 IANA 时区数据库时，对常见时区降级为固定 UTC 偏移，
    保证 cron 仍可按用户预期的小时级粒度触发。
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        return None
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover - Python < 3.9
        return _fallback_timezone(cleaned)
    try:
        return ZoneInfo(cleaned)
    except Exception:  # noqa: BLE001 - 统一转成调度错误或降级
        return _fallback_timezone(cleaned)


def _fallback_timezone(name: str) -> Any:
    """缺少时区数据库时的降级解析。"""
    offset = _FIXED_OFFSETS.get(name)
    if offset is not None:
        return timezone(offset)
    raise ScheduleError(
        f"无法识别的时区：{name}（当前环境缺少 IANA 时区数据库，"
        "请安装 tzdata 或改用 UTC）"
    )


def _parse_field(raw: str, index: int) -> set[int]:
    """解析单个 cron 字段为取值集合。"""
    low, high = _FIELD_RANGES[index]
    allowed = (
        set(_MONTH_ALIASES.values())
        if index == 3
        else set(_DOW_ALIASES.values())
        if index == 4
        else set(range(low, high + 1))
    )
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ScheduleError(f"cron 第 {index + 1} 段存在空项")
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                raise ScheduleError(f"cron 第 {index + 1} 段的步长非法：{step_text}")
            step = int(step_text)
            part = part.strip() or "*"
        if part == "*":
            start, end = low, high
        elif "-" in part.lstrip("-"):
            start_text, _, end_text = part.partition("-")
            start = _alias_value(start_text, index, allowed)
            end = _alias_value(end_text, index, allowed)
        else:
            start = end = _alias_value(part, index, allowed)
        if start > end:
            raise ScheduleError(f"cron 第 {index + 1} 段区间起点大于终点：{part}")
        for number in range(start, end + 1, step):
            if number in allowed:
                values.add(number)
    if not values:  # pragma: no cover - 防御性兜底：越界取值已在更早处被拒绝
        raise ScheduleError(f"cron 第 {index + 1} 段没有任何有效取值：{raw}")
    return values


def _alias_value(text: str, index: int, allowed: set[int]) -> int:
    """解析单个取值，支持三字母别名。"""
    token = text.strip().lower()
    aliases = _MONTH_ALIASES if index == 3 else _DOW_ALIASES if index == 4 else {}
    if token in aliases:
        number = aliases[token]
    elif token.isdigit():
        number = int(token)
    else:
        raise ScheduleError(f"cron 第 {index + 1} 段取值非法：{text}")
    low, high = _FIELD_RANGES[index]
    if index == 4 and number == 7:
        # cron 中 7 与 0 都表示周日，归一到 0 后再校验上界
        number = 0
    if not low <= number <= high:
        raise ScheduleError(f"cron 第 {index + 1} 段取值越界：{text}")
    return number


@dataclass(frozen=True, slots=True)
class CronExpression:
    """内置五段 cron 解析结果。"""

    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    day_restricted: bool
    weekday_restricted: bool
    raw: str

    def next_after(self, moment: datetime) -> datetime | None:
        """返回严格晚于 ``moment`` 的下一个匹配时刻。

        Args:
            moment: 基准时刻（需带时区）。

        Returns:
            datetime | None: 下一个匹配时刻；四年内无匹配时返回 None。
        """
        cursor = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = moment + timedelta(days=366 * 4)
        while cursor <= limit:
            if cursor.month not in self.months:
                cursor = _advance_month(cursor)
                continue
            if not self._matches_day(cursor):
                cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if cursor.hour not in self.hours:
                cursor = (cursor + timedelta(hours=1)).replace(minute=0)
                continue
            if cursor.minute not in self.minutes:
                cursor = cursor + timedelta(minutes=1)
                continue
            return cursor
        return None

    def _matches_day(self, moment: datetime) -> bool:
        """判断日期是否命中。

        ``day of month`` 与 ``day of week`` 同时被限定时按 cron 标准取并集。
        """
        # datetime.weekday(): 周一=0；cron: 周日=0
        weekday = (moment.weekday() + 1) % 7
        day_hit = moment.day in self.days
        weekday_hit = weekday in self.weekdays
        if self.day_restricted and self.weekday_restricted:
            return day_hit or weekday_hit
        return day_hit and weekday_hit


def _advance_month(moment: datetime) -> datetime:
    """跳到下一个月同一时刻的零点。"""
    year, month = (
        (moment.year + 1, 1) if moment.month == 12 else (moment.year, moment.month + 1)
    )
    return moment.replace(year=year, month=month, day=1, hour=0, minute=0)


def parse_cron(expression: str) -> CronExpression:
    """解析五段 cron 表达式。

    Args:
        expression: 五段 cron 字符串。

    Returns:
        CronExpression: 解析结果。

    Raises:
        ScheduleError: 表达式非法时抛出。
    """
    raw = str(expression or "").strip()
    if not raw:
        raise ScheduleError("cron 表达式不能为空")
    if not re.fullmatch(r"\S+(\s+\S+){4}", raw):
        raise ScheduleError("cron 表达式必须是五段：分 时 日 月 周")
    parts = raw.split()
    minutes, hours, days, months, weekdays = (
        _parse_field(parts[index], index) for index in range(5)
    )
    return CronExpression(
        minutes=frozenset(minutes),
        hours=frozenset(hours),
        days=frozenset(days),
        months=frozenset(months),
        weekdays=frozenset(weekdays),
        day_restricted=parts[2] != "*",
        weekday_restricted=parts[4] != "*",
        raw=raw,
    )


_DURATION_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> timedelta:
    """解析相对延迟，如 ``30m``、``2h``、``1d``。

    Args:
        text: 延迟字符串。

    Returns:
        timedelta: 解析出的时间差。

    Raises:
        ScheduleError: 格式非法时抛出。
    """
    match = _DURATION_PATTERN.match(str(text or ""))
    if not match:
        raise ScheduleError(f"无法解析相对延迟：{text}（示例 30m、2h、1d）")
    amount = int(match.group(1))
    if amount <= 0:
        raise ScheduleError("相对延迟必须大于 0")
    return timedelta(seconds=amount * _DURATION_UNITS[match.group(2).lower()])


def parse_absolute_time(text: str) -> datetime:
    """解析一次性绝对时间。

    接受 Unix 毫秒时间戳或 ISO 日期时间字符串；无时区的字符串按本地
    时区解释。

    Args:
        text: 时间戳或日期时间字符串。

    Returns:
        datetime: 带时区的时刻。

    Raises:
        ScheduleError: 格式非法时抛出。
    """
    raw = str(text or "").strip()
    if not raw:
        raise ScheduleError("绝对时间不能为空")
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScheduleError(f"无法解析绝对时间：{text}") from exc
    return parsed if parsed.tzinfo else parsed.astimezone()


def next_run_after(
    expression: str, moment: datetime, *, timezone_name: str | None = None
) -> datetime | None:
    """计算某 cron 表达式在 ``moment`` 之后的下一个时刻。

    Args:
        expression: 五段 cron 表达式。
        moment: 基准时刻。
        timezone_name: IANA 时区名，缺省沿用 ``moment`` 的时区。

    Returns:
        datetime | None: 下一个匹配时刻；不可达时返回 None。

    Raises:
        ScheduleError: 表达式或时区非法时抛出。
    """
    tzinfo = _resolve_timezone(timezone_name)
    local_moment = moment.astimezone(tzinfo) if tzinfo else moment
    if _croniter is not None:
        following = _croniter(str(expression).strip(), local_moment).get_next(datetime)
        return following.astimezone(moment.tzinfo) if following.tzinfo else following
    return parse_cron(str(expression)).next_after(local_moment)


@dataclass(frozen=True, slots=True)
class Schedule:
    """归一化后的调度定义。

    三种形态互斥：循环 ``cron``、一次性相对 ``after``、一次性绝对 ``at``。
    """

    kind: str  # cron | after | at
    cron_expression: str = ""
    duration_seconds: float = 0.0
    absolute_at_ms: int = 0
    timezone_name: str = ""

    @property
    def is_recurring(self) -> bool:
        """是否循环触发。"""
        return self.kind == "cron"

    def next_after(self, moment: datetime) -> datetime | None:
        """计算严格晚于 ``moment`` 的下一次触发时刻。

        一次性调度返回绝对时刻且仅在尚未过期时有值。

        Args:
            moment: 基准时刻。

        Returns:
            datetime | None: 下一次触发时刻；无后续触发时返回 None。
        """
        if self.kind == "at":
            target = datetime.fromtimestamp(self.absolute_at_ms / 1000, tz=timezone.utc)
            return target if target > moment else None
        if self.kind == "after":
            return None  # 相对延迟在创建时已折算成绝对时刻
        tzinfo = _resolve_timezone(self.timezone_name) or moment.tzinfo
        local_moment = moment.astimezone(tzinfo) if tzinfo else moment
        if _croniter is not None:
            following = _croniter(self.cron_expression, local_moment).get_next(datetime)
            return (
                following.astimezone(moment.tzinfo) if following.tzinfo else following
            )
        return parse_cron(self.cron_expression).next_after(local_moment)


def parse_schedule(
    *,
    schedule: str | None = None,
    after: str | None = None,
    at: str | None = None,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> tuple[Schedule, datetime | None]:
    """解析三选一的调度参数。

    Args:
        schedule: 五段 cron 表达式。
        after: 相对延迟，如 ``30m``。
        at: 一次性绝对时间。
        timezone_name: IANA 时区名，影响 cron 解释。
        now: 折算相对延迟的基准时刻，缺省取当前时间。

    Returns:
        tuple: ``(Schedule, 首次触发时刻)``。

    Raises:
        ScheduleError: 参数组合非法或格式错误时抛出。
    """
    provided = [
        name
        for name, value in (("schedule", schedule), ("after", after), ("at", at))
        if str(value or "").strip()
    ]
    if len(provided) != 1:
        raise ScheduleError("schedule、after、at 必须且只能提供一个")

    moment = now or datetime.now().astimezone()
    provided_name = provided[0]

    if provided_name == "schedule":
        # 提前校验时区，避免创建成功后才在调度时报错
        _resolve_timezone(timezone_name)
        parse_cron(str(schedule).strip())
        parsed = Schedule(
            kind="cron",
            cron_expression=str(schedule).strip(),
            timezone_name=str(timezone_name or "").strip(),
        )
        return parsed, parsed.next_after(moment)

    if provided_name == "after":
        offset = parse_duration(str(after))
        target = moment + offset
        return (
            Schedule(kind="after", duration_seconds=offset.total_seconds()),
            target,
        )

    target = parse_absolute_time(str(at))
    return (
        Schedule(kind="at", absolute_at_ms=int(target.timestamp() * 1000)),
        target,
    )

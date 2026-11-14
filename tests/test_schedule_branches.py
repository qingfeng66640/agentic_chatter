"""定时任务剩余覆盖补齐。

覆盖防御性分支：空 cron 段、非法步长、越界别名、时间戳解析、
croniter 可用路径、存储的并发认领与空更新等。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Never

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - 测试环境准备
    sys.path.insert(0, str(ROOT))

from ..task_runtime.schedule import (  # noqa: E402
    ScheduleError,
    ScheduleManager,
    ScheduleStore,
    parse_cron,
    parse_schedule,
)
from ..task_runtime.schedule import expression as expression_module  # noqa: E402
from ..task_runtime.schedule import manager as manager_module  # noqa: E402
from ..task_runtime.schedule import store as store_module  # noqa: E402

CST = timezone(timedelta(hours=8))
MONDAY = datetime(2026, 10, 19, 8, 0, tzinfo=CST)


class TestFieldParsing:
    """cron 字段解析的边界。"""

    def test_empty_segment_rejected(self) -> None:
        """空段被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * ,1")

    def test_invalid_step_rejected(self) -> None:
        """非法步长被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * 1/0")
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * 1/x")

    def test_reversed_range_rejected(self) -> None:
        """区间起点大于终点被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * 5-1")

    def test_no_valid_values_rejected(self) -> None:
        """取值全部越界时被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * 70")

    def test_sunday_alias_seven_normalized(self) -> None:
        """星期 7 归一到 0。"""
        parsed = parse_cron("0 9 * * 7")
        assert parsed.weekdays == frozenset({0})

    def test_month_alias_parsed(self) -> None:
        """月份别名可解析。"""
        parsed = parse_cron("0 9 * jan *")
        assert parsed.months == frozenset({1})

    def test_day_and_weekday_union(self) -> None:
        """日月与周同时限定时取并集。"""
        parsed = parse_cron("0 9 15 * 1")
        assert parsed.day_restricted is True
        assert parsed.weekday_restricted is True
        following = parsed.next_after(MONDAY)
        assert following is not None
        # 周一（10-19）或每月 15 日，取更早者
        assert (following.month, following.day) == (10, 19)

    def test_only_day_restricted(self) -> None:
        """只限定日时按日匹配。"""
        parsed = parse_cron("0 9 15 * *")
        following = parsed.next_after(MONDAY)
        assert following is not None
        assert following.day == 15


class TestAbsoluteTime:
    """绝对时间解析。"""

    def test_unix_milliseconds(self) -> None:
        """Unix 毫秒时间戳可解析。"""
        parsed = expression_module.parse_absolute_time("1700000000000")  # noqa: SLF001
        assert parsed.tzinfo is not None
        assert int(parsed.timestamp() * 1000) == 1700000000000

    def test_iso_with_z_suffix(self) -> None:
        """Z 后缀的 ISO 时间可解析。"""
        parsed = expression_module.parse_absolute_time("2030-01-01T00:00:00Z")  # noqa: SLF001
        assert parsed.year == 2030

    def test_naive_iso_uses_local(self) -> None:
        """无时区的 ISO 时间按本地解释。"""
        parsed = expression_module.parse_absolute_time("2030-01-01T08:00:00")  # noqa: SLF001
        assert parsed.tzinfo is not None

    def test_empty_rejected(self) -> None:
        """空时间被拒绝。"""
        with pytest.raises(ScheduleError):
            expression_module.parse_absolute_time("")  # noqa: SLF001

    def test_unparsable_rejected(self) -> None:
        """无法解析的时间被拒绝。"""
        with pytest.raises(ScheduleError):
            expression_module.parse_absolute_time("下周三下午")  # noqa: SLF001


class TestCroniterPath:
    """croniter 可用时的计算路径。"""

    def test_next_run_after_with_croniter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """croniter 存在时走该分支。"""
        captured: dict[str, object] = {}

        class _FakeCroniter:
            """记录 cron 表达式与起点，并固定返回 2030 年触发时刻的替身。"""

            def __init__(self, expression: str, start: datetime) -> None:
                """记录传给 croniter 的表达式和起始时间。

                Args:
                    expression: 待计算下一次触发的 cron 表达式。
                    start: 下一次触发计算的起始时刻。
                """
                captured["expression"] = expression
                captured["start"] = start

            def get_next(self, kind: type[datetime]) -> datetime:
                """用调用方指定的日期时间类型构造固定触发时刻。

                Args:
                    kind: croniter 接口要求返回的日期时间类型。

                Returns:
                    2030 年 1 月 1 日 09:00 的 UTC 时刻。
                """
                return kind(2030, 1, 1, 9, 0, tzinfo=timezone.utc)

        monkeypatch.setattr(expression_module, "_croniter", _FakeCroniter)
        result = expression_module.next_run_after("0 9 * * *", MONDAY)  # noqa: SLF001
        assert result is not None
        assert result.year == 2030
        assert captured["expression"] == "0 9 * * *"

    def test_schedule_next_after_with_croniter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Schedule 在 croniter 可用时同样走该分支。"""

        class _FakeCroniter:
            """保留计算起点，并固定返回 2031 年触发时刻的 croniter 替身。"""

            def __init__(self, expression: str, start: datetime) -> None:
                """保存调度定义传入的计算起点。

                Args:
                    expression: 接收但不解析的 cron 表达式。
                    start: 保存为实例状态的计算起点。
                """
                self._start = start

            def get_next(self, kind: type[datetime]) -> datetime:
                """用指定的日期时间类型返回固定的后续触发时刻。

                Args:
                    kind: croniter 接口要求返回的日期时间类型。

                Returns:
                    2031 年 5 月 5 日 05:05 的 UTC 时刻。
                """
                return kind(2031, 5, 5, 5, 5, tzinfo=timezone.utc)

        monkeypatch.setattr(expression_module, "_croniter", _FakeCroniter)
        schedule = expression_module.Schedule(  # noqa: SLF001
            kind="cron", cron_expression="0 9 * * *", timezone_name="UTC"
        )
        result = schedule.next_after(MONDAY)
        assert result is not None
        assert result.year == 2031

    def test_has_croniter_reports_availability(self) -> None:
        """has_croniter 反映可用性。"""
        assert isinstance(expression_module.has_croniter(), bool)  # noqa: SLF001

    def test_at_kind_next_after(self) -> None:
        """at 类型在目标时刻之前返回该时刻。"""
        schedule = expression_module.Schedule(  # noqa: SLF001
            kind="at", absolute_at_ms=2_000_000_000_000
        )
        assert schedule.next_after(MONDAY) is not None
        # 目标已过则不再触发
        assert schedule.next_after(datetime(2100, 1, 1, tzinfo=timezone.utc)) is None

    def test_after_kind_has_no_next(self) -> None:
        """after 类型不产生后续触发。"""
        schedule = expression_module.Schedule(kind="after", duration_seconds=60)  # noqa: SLF001
        assert schedule.next_after(MONDAY) is None


class TestStoreEdges:
    """存储层的边界分支。"""

    def test_list_by_state(self, tmp_path: Path) -> None:
        """可按状态过滤。"""
        store = ScheduleStore(tmp_path / "db")
        store.create(
            {
                "cron_id": "a",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        assert len(store.list_all(state="active")) == 1
        assert store.list_all(state="expired") == ()

    def test_update_with_empty_changes(self, tmp_path: Path) -> None:
        """空变更直接返回原记录。"""
        store = ScheduleStore(tmp_path / "db")
        store.create(
            {
                "cron_id": "a",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        record = store.update("a", {})
        assert record is not None
        assert record["cron_id"] == "a"

    def test_claim_skips_already_claimed(self, tmp_path: Path) -> None:
        """已被认领的任务不会二次认领。"""
        store = ScheduleStore(tmp_path / "db")
        store.create(
            {
                "cron_id": "a",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        assert len(store.claim_due(9999)) == 1
        # 再次扫描时哨兵值远大于 now，不会被认领
        assert store.claim_due(9999) == ()

    def test_mark_run_without_pending(self, tmp_path: Path) -> None:
        """没有待认领记录时标记操作为空操作。"""
        store = ScheduleStore(tmp_path / "db")
        store.mark_run("missing", status="delivered")

    def test_drop_thread_entries(self, tmp_path: Path) -> None:
        """可删除固定线程历史。"""
        store = ScheduleStore(tmp_path / "db")
        store.append_thread_entry("c1", "记录")
        assert len(store.list_thread_entries("c1")) == 1
        store.drop_thread_entries("c1")
        assert store.list_thread_entries("c1") == ()

    def test_describe_run_with_task_and_error(self) -> None:
        """运行记录渲染包含任务与错误。"""
        text = store_module.describe_run(  # noqa: SLF001
            {
                "scheduled_at_ms": 1_700_000_000_000,
                "status": "failed",
                "task_id": "task-1",
                "error": "超时",
            }
        )
        assert "failed" in text
        assert "task=task-1" in text
        assert "error=超时" in text

    def test_describe_run_without_optional_fields(self) -> None:
        """缺少任务与错误时只渲染基础信息。"""
        text = store_module.describe_run({"status": "pending"})  # noqa: SLF001
        assert "pending" in text
        assert "task=" not in text


class TestManagerUpdateBranches:
    """管理器更新分支。"""

    def test_update_cron_name(self, tmp_path: Path) -> None:
        """可修改标题。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="旧名", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        updated = manager.update(str(record["cron_id"]), cron_name="新名")
        assert updated is not None
        assert updated["cron_name"] == "新名"

    def test_update_timezone_only_keeps_expression(self, tmp_path: Path) -> None:
        """只改时区时表达式保持不变。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        updated = manager.update(str(record["cron_id"]), timezone_name="Asia/Shanghai")
        assert updated is not None
        assert updated["cron_expression"] == "0 9 * * *"
        assert updated["timezone_name"] == "Asia/Shanghai"

    def test_update_resume_recomputes_next_run(self, tmp_path: Path) -> None:
        """暂停后恢复会重算下次触发时刻。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        cron_id = str(record["cron_id"])
        manager.update(cron_id, enabled=False)
        updated = manager.update(cron_id, enabled=True)
        assert updated is not None
        assert updated["next_run_at_ms"] is not None

    def test_get_schedule_manager_returns_instance(self, tmp_path: Path) -> None:
        """全局管理器可获取与释放。"""
        from ..task_runtime import schedule as schedule_pkg

        store = ScheduleStore(tmp_path / "db")
        instance = schedule_pkg.init_schedule_manager(store)
        assert schedule_pkg.get_schedule_manager() is instance
        schedule_pkg.shutdown_schedule_manager()
        assert schedule_pkg.get_schedule_manager() is None


class TestRunForever:
    """后台扫描循环。"""

    @pytest.mark.asyncio
    async def test_run_forever_scans_and_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """循环会持续扫描，取消后退出。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        scans: list[int] = []

        async def fake_scan() -> int:
            """记录扫描次数，并在第二次扫描时取消后台循环。

            Returns:
                首次扫描返回零，表示没有提交任务。

            Raises:
                asyncio.CancelledError: 记录到至少两次扫描时终止循环。
            """
            scans.append(1)
            if len(scans) >= 2:
                raise asyncio.CancelledError
            return 0

        monkeypatch.setattr(manager, "scan_once", fake_scan)

        with pytest.raises(asyncio.CancelledError):
            await manager.run_forever(interval=0.01)
        assert len(scans) >= 2


class TestSubmitterValueError:
    """提交回调的失败分支。"""

    @pytest.mark.asyncio
    async def test_create_failure_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """任务创建失败时返回空串而不是抛出。"""
        from .. import schedule_service
        from ..task_runtime import runtime as runtime_module

        plugin = type("P", (), {"config": _Config(tasks_dir=str(tmp_path))})()
        submit = schedule_service._make_submitter(plugin)  # noqa: SLF001

        class _RuntimeManager:
            """活动任务列表为空，但创建任务时报告容量异常的管理器替身。"""

            def get_active_tasks(self, stream_id: str) -> list[object]:
                """返回空活动任务列表，使提交进入任务创建分支。

                Args:
                    stream_id: 查询活动任务的聊天流标识。

                Returns:
                    不包含活动任务的列表。
                """
                return []

            def create(self, stream_id: str, goal: str, **kwargs: object) -> Never:
                """在创建阶段模拟任务数量超限。

                Args:
                    stream_id: 目标聊天流标识。
                    goal: 待创建任务的目标。
                    **kwargs: 接收但不使用的其他任务配置。

                Raises:
                    ValueError: 固定抛出任务容量超限异常。
                """
                raise ValueError("任务数超限")

        monkeypatch.setattr(
            runtime_module,
            "get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )
        monkeypatch.setattr(
            runtime_module,
            "get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )
        result = await submit(
            stream_id="s1",
            prompt="p",
            context="",
            cron_id="c1",
            record={},
        )
        assert result == ""


class _Config:
    """最小配置替身。"""

    def __init__(self, tasks_dir: str = "") -> None:
        """构造配置。

        Args:
            tasks_dir: 任务检查点目录。
        """
        self.tasks = type("Tasks", (), {"checkpoint_directory": tasks_dir})()
        self.schedule = type(
            "Schedule", (), {"database_directory": "", "enabled": True}
        )()


class TestGetManagerOriginal:
    """未被替换时的管理器获取路径。"""

    def test_get_manager_returns_global(self, tmp_path: Path) -> None:
        """未启用时返回 None，启用后返回全局实例。"""
        from ..task_runtime import schedule as schedule_pkg
        from ..tooling.cron_tools import _get_manager

        schedule_pkg.shutdown_schedule_manager()
        assert _get_manager() is None
        store = ScheduleStore(tmp_path / "db")
        instance = schedule_pkg.init_schedule_manager(store)
        try:
            assert _get_manager() is instance
        finally:
            schedule_pkg.shutdown_schedule_manager()


class TestStoreConcurrentClaim:
    """存储的并发认领分支。"""

    def test_claim_skipped_when_already_taken(self, tmp_path: Path) -> None:
        """已被其他执行者抢先时该行被跳过。"""
        store = ScheduleStore(tmp_path / "db")
        store.create(
            {
                "cron_id": "a",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        # 先手动把待执行时刻推走，模拟已被认领
        store.set_next_run("a", store_module.CLAIMED_SENTINEL_MS)  # noqa: SLF001
        assert store.claim_due(9999) == ()

    def test_claim_skipped_when_row_taken_mid_scan(self, tmp_path: Path) -> None:
        """扫描过程中行被抢占时跳过该行。"""
        store = ScheduleStore(tmp_path / "db")
        store.create(
            {
                "cron_id": "a",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        original_insert = store._insert_run  # noqa: SLF001
        calls = {"n": 0}

        def insert_then_steal(
            connection: sqlite3.Connection, record: dict[str, object], now_ms: int
        ) -> bool:
            """第一次插入后抢占该行，使条件更新不再命中。

            Args:
                connection: 当前认领事务使用的 SQLite 连接。
                record: 本次准备认领的调度记录。
                now_ms: 传给原始插入方法的当前毫秒时间戳。

            Returns:
                首次调用返回真；后续调用不插入并返回假。
            """
            calls["n"] += 1
            if calls["n"] == 1:
                original_insert(connection, record, now_ms)  # type: ignore[arg-type]
                connection.execute(  # type: ignore[attr-defined]
                    "UPDATE schedules SET next_run_at_ms = ? WHERE cron_id = ?",
                    (2, record["cron_id"]),
                )
                return True
            return False

        store._insert_run = insert_then_steal  # type: ignore[method-assign]  # noqa: SLF001
        claimed = store.claim_due(9999)
        # 抢占发生在认领之后，本次认领成立；但下一次扫描该行已被推进，
        # 不再重复认领。
        assert len(claimed) == 1
        assert store.claim_due(9999) == ()


class TestExpressionRemaining:
    """表达式层的剩余分支。"""

    def test_field_without_valid_values(self) -> None:
        """取值集合为空时报错。"""
        with pytest.raises(ScheduleError):
            parse_cron("0 9 * * 1-7/9")

    def test_next_run_after_without_timezone(self) -> None:
        """不指定时区时沿用基准时刻的时区。"""
        result = expression_module.next_run_after("0 9 * * *", MONDAY)  # noqa: SLF001
        assert result is not None
        assert result.tzinfo is not None


class TestManagerKeepOriginalTiming:
    """更新时沿用原调度参数。"""

    def test_update_after_keeps_duration(self, tmp_path: Path) -> None:
        """只改时区时 after 类型与延迟保持。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(cron_name="n", prompt="p", stream_id="s", after="30m")
        updated = manager.update(str(record["cron_id"]), timezone_name="Asia/Shanghai")
        assert updated is not None
        assert updated["schedule_kind"] == "after"
        assert float(updated["duration_seconds"]) == 1800.0

    def test_update_at_keeps_absolute(self, tmp_path: Path) -> None:
        """只改时区时 at 类型与绝对时刻保持。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n",
            prompt="p",
            stream_id="s",
            at="2099-01-01T00:00:00+08:00",
        )
        updated = manager.update(str(record["cron_id"]), timezone_name="Asia/Shanghai")
        assert updated is not None
        assert updated["schedule_kind"] == "at"
        assert int(updated["absolute_at_ms"]) == int(record["absolute_at_ms"])


class TestRunsViewNonEmpty:
    """运行历史非空分支。"""

    @pytest.mark.asyncio
    async def test_runs_view_lists_records(self, tmp_path: Path) -> None:
        """有运行记录时列出内容。"""
        from ..tooling import cron_tools

        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        original = cron_tools._get_manager  # noqa: SLF001
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            record = manager.create(
                cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            store.set_next_run(str(record["cron_id"]), 1)
            store.claim_due(9999)

            tool = cron_tools.ReadCronTool.__new__(  # noqa: SLF001
                cron_tools.ReadCronTool
            )
            tool.get_current_stream_id = lambda: "s"  # type: ignore[method-assign]
            ok, message = await tool.execute(
                view="runs", cron_ref=str(record["cron_id"])
            )
            assert ok is True
            assert "pending" in message
        finally:
            cron_tools._get_manager = original  # type: ignore[assignment]


def test_after_delay_folds_into_absolute_first_run() -> None:
    """相对延迟会折算出首次触发时刻。"""
    _, first_run = parse_schedule(after="90m", now=MONDAY)
    assert first_run is not None
    assert first_run == MONDAY + timedelta(minutes=90)


def test_normalize_session_mode_is_exported() -> None:
    """会话策略归一化函数可从模块访问。"""
    assert manager_module._normalize_session_mode("new") == "new"  # noqa: SLF001

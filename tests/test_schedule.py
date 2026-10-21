"""定时任务子系统测试。

覆盖调度表达式解析、SQLite 持久化、调度管理器的触发与可靠性语义，
以及四个 LLM 工具的关键约束。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - 测试环境准备
    sys.path.insert(0, str(ROOT))

from agentic_chatter.task_runtime.schedule import (  # noqa: E402
    ScheduleError,
    ScheduleManager,
    ScheduleStore,
    parse_cron,
    parse_schedule,
)
from agentic_chatter.task_runtime.schedule.store import (  # noqa: E402
    CLAIMED_SENTINEL_MS,
)

CST = timezone(timedelta(hours=8))
MONDAY_MORNING = datetime(2026, 10, 19, 8, 0, tzinfo=CST)


@pytest.fixture
def store(tmp_path: Path) -> ScheduleStore:
    """构造一个临时存储。"""
    return ScheduleStore(tmp_path / "db")


@pytest.fixture
def manager(store: ScheduleStore) -> ScheduleManager:
    """构造一个不提交任务的管理器。"""
    return ScheduleManager(store, submitter=None)


# ==================== 表达式解析 ====================


class TestExpression:
    """调度表达式解析。"""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("0 9 * * 1-5", datetime(2026, 10, 19, 9, 0, tzinfo=CST)),
            ("*/15 * * * *", datetime(2026, 10, 19, 8, 15, tzinfo=CST)),
            ("30 14 * * 1", datetime(2026, 10, 19, 14, 30, tzinfo=CST)),
            ("0 9 * * 0", datetime(2026, 10, 25, 9, 0, tzinfo=CST)),
            ("0 0 1 * *", datetime(2026, 11, 1, 0, 0, tzinfo=CST)),
        ],
    )
    def test_builtin_cron_matches(self, expression: str, expected: datetime) -> None:
        """内置解析器给出正确的下一个触发时刻。"""
        actual = parse_cron(expression).next_after(MONDAY_MORNING)
        assert actual is not None
        assert actual.replace(tzinfo=CST) == expected.replace(tzinfo=CST)

    def test_cron_rolls_over_year(self) -> None:
        """跨年边界正确推进。"""
        end_of_year = datetime(2026, 12, 31, 23, 50, tzinfo=CST)
        actual = parse_cron("0 9 * * *").next_after(end_of_year)
        assert actual is not None
        assert (actual.month, actual.day, actual.hour) == (1, 1, 9)

    def test_impossible_expression_returns_none(self) -> None:
        """永不相遇的表达式返回 None 而不是无限循环。"""
        assert parse_cron("0 9 30 2 *").next_after(MONDAY_MORNING) is None

    @pytest.mark.parametrize(
        "expression",
        ["", "0 9 * *", "* * * * * *", "99 9 * * *", "0 9 * * abc", "0 9 * * 8"],
    )
    def test_invalid_cron_rejected(self, expression: str) -> None:
        """非法表达式被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_cron(expression)

    @pytest.mark.parametrize("value", ["mon", "MON", "1-5", "1,3,5", "0-4/2"])
    def test_aliases_and_lists(self, value: str) -> None:
        """别名、列表与步长可解析。"""
        assert parse_cron(f"0 9 * * {value}").next_after(MONDAY_MORNING) is not None

    @pytest.mark.parametrize("value", ["0-30", "8", "70 * * * *"])
    def test_out_of_range_rejected(self, value: str) -> None:
        """越界取值被拒绝（含区间终点越界）。"""
        with pytest.raises(ScheduleError):
            parse_cron(value)

    def test_timezone_shifts_result(self) -> None:
        """时区影响 cron 解释结果。"""
        utc_noon = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
        shanghai = parse_schedule(
            schedule="0 9 * * *", timezone_name="Asia/Shanghai", now=utc_noon
        )[1]
        assert shanghai is not None
        # 09:00 +08:00 == 01:00 UTC，下一次应是次日
        assert shanghai.astimezone(timezone.utc).hour == 1

    def test_unknown_timezone_rejected(self) -> None:
        """未知时区被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_schedule(
                schedule="0 9 * * *", timezone_name="Mars/Phobos", now=MONDAY_MORNING
            )

    @pytest.mark.parametrize("value", ["30m", "2h", "1d", "45s", "1w"])
    def test_duration_parsing(self, value: str) -> None:
        """相对延迟可解析。"""
        parsed, first_run = parse_schedule(after=value, now=MONDAY_MORNING)
        assert parsed.kind == "after"
        assert parsed.is_recurring is False
        assert first_run is not None and first_run > MONDAY_MORNING

    @pytest.mark.parametrize("value", ["0m", "-5m", "abc", "10x", ""])
    def test_invalid_duration_rejected(self, value: str) -> None:
        """非法延迟被拒绝。"""
        with pytest.raises(ScheduleError):
            parse_schedule(after=value, now=MONDAY_MORNING)

    def test_absolute_time_parsing(self) -> None:
        """绝对时间可解析。"""
        parsed, first_run = parse_schedule(
            at="2030-06-01T09:00:00+08:00", now=MONDAY_MORNING
        )
        assert parsed.kind == "at"
        assert parsed.is_recurring is False
        assert first_run is not None and first_run.year == 2030

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"schedule": "0 9 * * *", "after": "30m"},
            {"schedule": "0 9 * * *", "at": "2030-01-01T00:00:00"},
            {"after": "30m", "at": "2030-01-01T00:00:00"},
        ],
    )
    def test_exactly_one_trigger_required(self, kwargs: dict[str, str]) -> None:
        """三个参数必须且只能提供一个。"""
        with pytest.raises(ScheduleError):
            parse_schedule(now=MONDAY_MORNING, **kwargs)


# ==================== 存储 ====================


class TestStore:
    """SQLite 存储。"""

    def test_create_and_read(self, store: ScheduleStore) -> None:
        """创建后可读回。"""
        record = store.create(
            {
                "cron_id": "c1",
                "cron_name": "测试",
                "prompt": "做点什么",
                "stream_id": "s1",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        assert record["cron_id"] == "c1"
        assert store.get("c1") is not None
        assert store.find_by_name("测试") is not None

    def test_claim_due_single_winner(self, store: ScheduleStore) -> None:
        """同一触发点只会被认领一次。"""
        store.create(
            {
                "cron_id": "c1",
                "cron_name": "测试",
                "prompt": "p",
                "stream_id": "s1",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1000,
            }
        )
        first = store.claim_due(2000)
        second = store.claim_due(2000)
        assert len(first) == 1
        assert second == ()

    def test_disabled_and_undeleted_not_claimed(self, store: ScheduleStore) -> None:
        """暂停或已删除的任务不参与认领。"""
        store.create(
            {
                "cron_id": "paused",
                "cron_name": "暂停",
                "prompt": "p",
                "stream_id": "s1",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "enabled": 0,
                "next_run_at_ms": 1000,
            }
        )
        assert store.claim_due(9999) == ()

    def test_generation_increments(self, store: ScheduleStore) -> None:
        """世代计数随修改递增。"""
        store.create(
            {
                "cron_id": "c1",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        assert store.bump_generation("c1") == 1
        assert store.bump_generation("c1") == 2

    def test_soft_delete_keeps_runs(self, store: ScheduleStore) -> None:
        """删除保留运行历史。"""
        store.create(
            {
                "cron_id": "c1",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        store.claim_due(5000)
        assert store.soft_delete("c1") is True
        assert store.get("c1") is None
        assert len(store.list_runs("c1")) == 1

    def test_stale_claim_recovered(self, store: ScheduleStore) -> None:
        """认领后崩溃遗留的哨兵值会在启动时被恢复。"""
        store.create(
            {
                "cron_id": "c1",
                "cron_name": "n",
                "prompt": "p",
                "stream_id": "s",
                "session_mode": "new",
                "schedule_kind": "cron",
                "cron_expression": "0 9 * * *",
                "next_run_at_ms": 1,
            }
        )
        assert len(store.claim_due(5000)) == 1
        stuck = store.get("c1")
        assert stuck is not None
        assert int(stuck["next_run_at_ms"]) == CLAIMED_SENTINEL_MS

        # 模拟重启：恢复后应重新参与排程
        assert store.reset_stale_claims() == 1
        recovered = store.get("c1")
        assert recovered is not None
        assert int(recovered["next_run_at_ms"]) < CLAIMED_SENTINEL_MS

    def test_reset_stale_claims_is_idempotent(self, store: ScheduleStore) -> None:
        """没有滞留认领时恢复操作为空操作。"""
        assert store.reset_stale_claims() == 0

    def test_thread_history_trimmed(self, store: ScheduleStore) -> None:
        for index in range(8):
            store.append_thread_entry("c1", f"第{index}次", max_entries=3)
        entries = store.list_thread_entries("c1", limit=10)
        assert len(entries) == 3
        # 时间正序，保留的是最近三条
        assert entries[0]["summary"] == "第5次"


# ==================== 管理器 ====================


class TestManager:
    """调度管理器。"""

    def test_create_cron_records_next_run(self, manager: ScheduleManager) -> None:
        """创建循环任务时写入下次触发时刻。"""
        record = manager.create(
            cron_name="晨报",
            prompt="整理今日待办",
            stream_id="s1",
            schedule="0 9 * * *",
        )
        assert record["state"] == "active"
        assert record["next_run_at_ms"] is not None
        assert record["schedule_kind"] == "cron"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cron_name": "", "prompt": "p", "stream_id": "s"},
            {"cron_name": "n", "prompt": "", "stream_id": "s"},
            {"cron_name": "n", "prompt": "p", "stream_id": ""},
        ],
    )
    def test_create_rejects_empty_fields(
        self, manager: ScheduleManager, kwargs: dict[str, str]
    ) -> None:
        """必填字段为空时拒绝。"""
        with pytest.raises(ScheduleError):
            manager.create(schedule="0 9 * * *", **kwargs)

    def test_me_session_target_rejected(self, manager: ScheduleManager) -> None:
        """me 不是合法的会话目标。"""
        with pytest.raises(ScheduleError):
            manager.create(
                cron_name="n",
                prompt="p",
                stream_id="s",
                session_mode="session",
                session_target="me",
                schedule="0 9 * * *",
            )

    def test_new_mode_clears_session_target(self, manager: ScheduleManager) -> None:
        """new 模式不保留会话目标。"""
        record = manager.create(
            cron_name="n",
            prompt="p",
            stream_id="s",
            session_mode="new",
            session_target="leftover",
            schedule="0 9 * * *",
        )
        assert record["session_target"] == ""

    def test_resolve_by_prefix(self, manager: ScheduleManager) -> None:
        """可按唯一名称前缀解析。"""
        record = manager.create(
            cron_name="工作日晨报", prompt="p", stream_id="s", schedule="0 9 * * 1-5"
        )
        found = manager.resolve_id(str(record["cron_id"])[:8])
        assert found is not None
        assert found["cron_id"] == record["cron_id"]
        assert manager.resolve_id("工作日晨报") is not None
        assert manager.resolve_id("不存在") is None

    def test_resolve_ambiguous_prefix_raises(self, manager: ScheduleManager) -> None:
        """前缀匹配到多个任务时报错而非随意挑一个。"""
        manager.create(cron_name="日报A", prompt="p", stream_id="s", schedule="0 9 * * *")
        manager.create(cron_name="日报B", prompt="p", stream_id="s", schedule="0 10 * * *")
        with pytest.raises(ScheduleError):
            manager.resolve_id("日报")

    def test_update_bumps_generation_on_timing_change(
        self, manager: ScheduleManager
    ) -> None:
        """修改表达式时递增世代。"""
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        before = int(record["generation"])
        updated = manager.update(str(record["cron_id"]), schedule="30 10 * * *")
        assert updated is not None
        assert int(updated["generation"]) == before + 1
        assert updated["cron_expression"] == "30 10 * * *"

    def test_update_prompt_keeps_generation(self, manager: ScheduleManager) -> None:
        """只改指令不递增世代。"""
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        before = int(record["generation"])
        updated = manager.update(str(record["cron_id"]), prompt="新指令")
        assert updated is not None
        assert int(updated["generation"]) == before
        assert updated["prompt"] == "新指令"

    def test_update_preserves_schedule_when_only_prompt_changes(
        self, manager: ScheduleManager
    ) -> None:
        """改指令不会把调度表达式清空。"""
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        updated = manager.update(str(record["cron_id"]), prompt="新指令")
        assert updated is not None
        assert updated["schedule_kind"] == "cron"
        assert updated["cron_expression"] == "0 9 * * *"
        assert updated["next_run_at_ms"] is not None

    def test_disable_clears_next_run(self, manager: ScheduleManager) -> None:
        """暂停后不再有待执行时刻。"""
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        updated = manager.update(str(record["cron_id"]), enabled=False)
        assert updated is not None
        assert updated["next_run_at_ms"] is None

    def test_delete_removes_from_list(self, manager: ScheduleManager) -> None:
        """删除后不再出现在列表中。"""
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        assert manager.delete(str(record["cron_id"])) is True
        assert manager.read_list() == ()

    @pytest.mark.asyncio
    async def test_failure_does_not_disable_task(
        self, store: ScheduleStore
    ) -> None:
        """触发失败只记录，不禁用任务，下次仍会触发。"""

        async def failing(**_: object) -> str:
            raise RuntimeError("提交失败")

        manager = ScheduleManager(store, submitter=failing)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="*/1 * * * *"
        )
        manager._store.set_next_run(str(record["cron_id"]), 1)  # noqa: SLF001
        await manager.scan_once()

        refreshed = manager.read_detail(str(record["cron_id"]))
        assert refreshed is not None
        assert refreshed["state"] == "active"
        assert refreshed["enabled"] == 1
        # 仍然排着下一次
        assert refreshed["next_run_at_ms"] is not None

    @pytest.mark.asyncio
    async def test_catchup_does_not_replay(self, store: ScheduleStore) -> None:
        """宕机期间错过多个触发点时只补一次，并重同步到未来。"""
        calls: list[str] = []

        async def submitter(**_: object) -> str:
            calls.append("run")
            return "task-1"

        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="*/5 * * * *"
        )
        cron_id = str(record["cron_id"])
        # 把下次触发设到很久以前，模拟宕机期间错过很多次
        store.set_next_run(cron_id, 1)

        await manager.scan_once()

        assert len(calls) == 1
        refreshed = manager.read_detail(cron_id)
        assert refreshed is not None
        assert refreshed["next_run_at_ms"] is not None
        assert int(refreshed["next_run_at_ms"]) > 1_700_000_000_000

    @pytest.mark.asyncio
    async def test_one_shot_completes_and_stops(
        self, store: ScheduleStore
    ) -> None:
        """一次性任务触发后不再排程。"""

        async def submitter(**_: object) -> str:
            return "task-1"

        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(cron_name="n", prompt="p", stream_id="s", after="1s")
        cron_id = str(record["cron_id"])
        store.set_next_run(cron_id, 1)

        await manager.scan_once()

        refreshed = manager.read_detail(cron_id)
        assert refreshed is not None
        assert refreshed["state"] in ("completed", "expired")
        assert refreshed["next_run_at_ms"] is None

        # 不会再次触发
        assert await manager.scan_once() == 0

    @pytest.mark.asyncio
    async def test_thread_context_injected_on_second_run(
        self, store: ScheduleStore
    ) -> None:
        """固定线程模式下第二次触发会带上历史。"""
        contexts: list[str] = []

        async def submitter(*, context: str = "", **_: object) -> str:
            contexts.append(context)
            return "task-x"

        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(
            cron_name="n",
            prompt="p",
            stream_id="s",
            session_mode="session",
            schedule="*/5 * * * *",
        )
        cron_id = str(record["cron_id"])

        store.set_next_run(cron_id, 1)
        await manager.scan_once()
        assert contexts[0] == ""  # 首次没有历史

        store.set_next_run(cron_id, 1)
        await manager.scan_once()
        assert "此前运行" in contexts[1]
        assert "p" in contexts[1]

    @pytest.mark.asyncio
    async def test_stateless_mode_never_injects_context(
        self, store: ScheduleStore
    ) -> None:
        """无状态模式不注入历史。"""
        contexts: list[str] = []

        async def submitter(*, context: str = "", **_: object) -> str:
            contexts.append(context)
            return "task-x"

        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="*/5 * * * *"
        )
        cron_id = str(record["cron_id"])

        for _ in range(3):
            store.set_next_run(cron_id, 1)
            await manager.scan_once()

        assert contexts == ["", "", ""]

    @pytest.mark.asyncio
    async def test_scan_survives_store_error(
        self, store: ScheduleStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """存储异常不会中断扫描循环。"""

        def boom() -> tuple[()]:
            raise RuntimeError("db locked")

        monkeypatch.setattr(store, "claim_due", boom)
        manager = ScheduleManager(store)
        assert await manager.scan_once() == 0

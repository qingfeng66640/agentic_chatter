"""定时任务覆盖补齐测试。

针对主测试未覆盖到的分支：会话策略归一化、一次性任务过期判定、
提交回调缺失、不可达表达式、认领恢复、以及四个工具的错误分支。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - 测试环境准备
    sys.path.insert(0, str(ROOT))

from agentic_chatter.task_runtime.schedule import (  # noqa: E402
    ScheduleError,
    ScheduleManager,
    ScheduleStore,
)
from agentic_chatter.task_runtime.schedule import manager as manager_module  # noqa: E402
from agentic_chatter.tooling import cron_tools  # noqa: E402
from agentic_chatter.tooling.cron_tools import (  # noqa: E402
    CreateCronTool,
    DeleteCronTool,
    ReadCronTool,
    UpdateCronTool,
)

# 保存原始函数以便测试后恢复
_ORIGINAL_GET_MANAGER = cron_tools._get_manager


def _make(tool_cls: type, stream_id: str = "stream-1") -> object:
    """构造工具实例并绑定当前聊天流。

    Args:
        tool_cls: 工具类。
        stream_id: 伪装的当前聊天流。

    Returns:
        object: 已绑定流 ID 的工具实例。
    """
    tool = tool_cls.__new__(tool_cls)  # noqa: SLF001 - 绕过框架构造，仅测 execute 逻辑
    tool.get_current_stream_id = lambda: stream_id  # type: ignore[method-assign]
    return tool


# ==================== 会话策略归一化 ====================


class TestSessionMode:
    """会话策略归一化。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("new", "new"),
            ("", "new"),
            ("NEW", "new"),
            ("session", "session"),
            ("sessionId", "session"),
            ("session_id", "session"),
            ("fixed", "session"),
        ],
    )
    def test_normalize_valid(self, raw: str, expected: str) -> None:
        """合法策略被归一化。"""
        assert manager_module._normalize_session_mode(raw) == expected  # noqa: SLF001

    def test_normalize_invalid(self) -> None:
        """未知策略被拒绝。"""
        with pytest.raises(ScheduleError):
            manager_module._normalize_session_mode("weekly")  # noqa: SLF001

    def test_create_rejects_unknown_session_mode(self, tmp_path: Path) -> None:
        """创建时未知策略被拒绝。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        with pytest.raises(ScheduleError):
            manager.create(
                cron_name="n",
                prompt="p",
                stream_id="s",
                session_mode="weekly",
                schedule="0 9 * * *",
            )


# ==================== 一次性任务过期判定 ====================


class TestOneShotExpiry:
    """一次性任务的过期判定。"""

    def test_absolute_in_future_not_expired(self) -> None:
        """未来的绝对时间未过期。"""
        future = int(datetime.now(timezone.utc).timestamp() * 1000) + 600_000
        assert manager_module._is_expired({"absolute_at_ms": future}) is False  # noqa: SLF001

    def test_absolute_in_past_expired(self) -> None:
        """过去的绝对时间已过期。"""
        past = int(datetime.now(timezone.utc).timestamp() * 1000) - 600_000
        assert manager_module._is_expired({"absolute_at_ms": past}) is True  # noqa: SLF001

    def test_zero_duration_expired(self) -> None:
        """零延迟视为已过期。"""
        assert manager_module._is_expired({"duration_seconds": 0.0}) is True  # noqa: SLF001

    def test_positive_duration_not_expired(self) -> None:
        """正延迟未过期。"""
        assert manager_module._is_expired({"duration_seconds": 60.0}) is False  # noqa: SLF001


# ==================== 调度记录还原 ====================


class TestParseAgain:
    """从存储记录还原调度定义。"""

    def test_cron_kind(self) -> None:
        """cron 类型被还原。"""
        parsed = manager_module._parse_again(  # noqa: SLF001
            {"schedule_kind": "cron", "cron_expression": "0 9 * * *", "timezone_name": "UTC"}
        )
        assert parsed.kind == "cron"
        assert parsed.cron_expression == "0 9 * * *"

    def test_after_kind(self) -> None:
        """after 类型被还原。"""
        parsed = manager_module._parse_again(  # noqa: SLF001
            {"schedule_kind": "after", "duration_seconds": 60.0}
        )
        assert parsed.kind == "after"
        assert parsed.duration_seconds == 60.0

    def test_at_kind(self) -> None:
        """at 类型被还原。"""
        parsed = manager_module._parse_again(  # noqa: SLF001
            {"schedule_kind": "at", "absolute_at_ms": 1700000000000}
        )
        assert parsed.kind == "at"
        assert parsed.absolute_at_ms == 1700000000000


# ==================== 管理器边界 ====================


class TestManagerEdges:
    """管理器边界分支。"""

    def test_update_missing_returns_none(self, tmp_path: Path) -> None:
        """更新不存在的任务返回 None。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        assert manager.update("不存在", prompt="p") is None

    def test_update_rejects_empty_name(self, tmp_path: Path) -> None:
        """更新时标题为空被拒绝。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        with pytest.raises(ScheduleError):
            manager.update(str(record["cron_id"]), cron_name="   ")

    def test_update_rejects_empty_prompt(self, tmp_path: Path) -> None:
        """更新时指令为空被拒绝。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        with pytest.raises(ScheduleError):
            manager.update(str(record["cron_id"]), prompt="   ")

    def test_update_preserves_after_kind(self, tmp_path: Path) -> None:
        """改指令时保留 after 类型。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(cron_name="n", prompt="p", stream_id="s", after="5m")
        updated = manager.update(str(record["cron_id"]), prompt="新指令")
        assert updated is not None
        assert updated["schedule_kind"] == "after"
        assert updated["next_run_at_ms"] is not None

    def test_update_preserves_at_kind(self, tmp_path: Path) -> None:
        """改指令时保留 at 类型。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        record = manager.create(
            cron_name="n",
            prompt="p",
            stream_id="s",
            at="2099-01-01T00:00:00+08:00",
        )
        updated = manager.update(str(record["cron_id"]), prompt="新指令")
        assert updated is not None
        assert updated["schedule_kind"] == "at"

    def test_resolve_empty_reference(self, tmp_path: Path) -> None:
        """空引用解析为 None。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        assert manager.resolve_id("") is None
        assert manager.resolve_id("   ") is None

    def test_recover_stale_claims(self, tmp_path: Path) -> None:
        """恢复中断遗留的认领。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        store.set_next_run(str(record["cron_id"]), 1)
        store.claim_due(9999)
        assert manager.recover_stale_claims() == 1

    def test_recover_stale_claims_survives_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """恢复失败不影响启动。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)

        def boom() -> int:
            raise RuntimeError("db locked")

        monkeypatch.setattr(store, "reset_stale_claims", boom)
        assert manager.recover_stale_claims() == 0

    @pytest.mark.asyncio
    async def test_dispatch_without_submitter_marks_failed(
        self, tmp_path: Path
    ) -> None:
        """缺少提交回调时记为失败而不是崩溃。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store, submitter=None)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="*/5 * * * *"
        )
        store.set_next_run(str(record["cron_id"]), 1)
        await manager.scan_once()

        runs = store.list_runs(str(record["cron_id"]))
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        assert "缺少提交回调" in runs[0]["error"]

    @pytest.mark.asyncio
    async def test_unreachable_expression_completes_task(
        self, tmp_path: Path
    ) -> None:
        """永不相遇的表达式把任务置为完成且不再排程。"""

        async def submitter(**_: object) -> str:
            return "t1"

        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 30 2 *"
        )
        store.set_next_run(str(record["cron_id"]), 1)
        await manager.scan_once()

        refreshed = manager.read_detail(str(record["cron_id"]))
        assert refreshed is not None
        assert refreshed["state"] == "completed"
        assert refreshed["next_run_at_ms"] is None

    @pytest.mark.asyncio
    async def test_sync_submitter_supported(self, tmp_path: Path) -> None:
        """同步提交回调也能被正确调用。"""
        calls: list[str] = []

        def submitter(**_: object) -> str:
            calls.append("x")
            return "task-sync"

        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store, submitter=submitter)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="*/5 * * * *"
        )
        store.set_next_run(str(record["cron_id"]), 1)
        await manager.scan_once()

        assert calls == ["x"]
        runs = store.list_runs(str(record["cron_id"]))
        assert runs[0]["task_id"] == "task-sync"

    def test_thread_context_exposes_history(self, tmp_path: Path) -> None:
        """对外暴露固定线程历史。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        store.append_thread_entry("c1", "第一次的结果")
        context = manager.thread_context("c1")
        assert "第一次的结果" in context

    def test_thread_context_empty(self, tmp_path: Path) -> None:
        """无历史时上下文为空。"""
        manager = ScheduleManager(ScheduleStore(tmp_path / "db"))
        assert manager.thread_context("c1") == ""


# ==================== 工具错误分支 ====================


class TestToolErrorBranches:
    """工具的错误分支。"""

    @pytest.mark.asyncio
    async def test_create_without_stream(self, tmp_path: Path) -> None:
        """无法确定聊天流时拒绝创建。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            tool = _make(CreateCronTool, stream_id="")
            ok, message = await tool.execute(
                cron_name="n", prompt="p", schedule="0 9 * * *"
            )
            assert ok is False
            assert "聊天流" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_read_unknown_view(self, tmp_path: Path) -> None:
        """未知视图被拒绝。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            tool = _make(ReadCronTool)
            ok, message = await tool.execute(view="unknown")
            assert ok is False
            assert "未知" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_read_runs_empty(self, tmp_path: Path) -> None:
        """无运行记录时给出提示。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            record = manager.create(
                cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            tool = _make(ReadCronTool)
            ok, message = await tool.execute(
                view="runs", cron_ref=str(record["cron_id"])
            )
            assert ok is True
            assert "还没有运行记录" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_read_ambiguous_ref(self, tmp_path: Path) -> None:
        """前缀歧义时报错。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            manager.create(
                cron_name="日报A", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            manager.create(
                cron_name="日报B", prompt="p", stream_id="s", schedule="0 10 * * *"
            )
            tool = _make(ReadCronTool)
            ok, message = await tool.execute(view="detail", cron_ref="日报")
            assert ok is False
            assert "匹配到多个" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_update_ambiguous_ref(self, tmp_path: Path) -> None:
        """更新时前缀歧义报错。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            manager.create(
                cron_name="日报A", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            manager.create(
                cron_name="日报B", prompt="p", stream_id="s", schedule="0 10 * * *"
            )
            tool = _make(UpdateCronTool)
            ok, message = await tool.execute(cron_ref="日报", prompt="x")
            assert ok is False
            assert "匹配到多个" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_delete_ambiguous_ref(self, tmp_path: Path) -> None:
        """删除时前缀歧义报错。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            manager.create(
                cron_name="日报A", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            manager.create(
                cron_name="日报B", prompt="p", stream_id="s", schedule="0 10 * * *"
            )
            tool = _make(DeleteCronTool)
            ok, message = await tool.execute(cron_ref="日报")
            assert ok is False
            assert "匹配到多个" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_all_tools_report_disabled(self) -> None:
        """功能未启用时四个工具都明确报错。"""
        cron_tools._get_manager = lambda: None  # type: ignore[assignment]
        try:
            for tool, kwargs in (
                (CreateCronTool, {"cron_name": "n", "prompt": "p"}),
                (ReadCronTool, {}),
                (UpdateCronTool, {"cron_ref": "x"}),
                (DeleteCronTool, {"cron_ref": "x"}),
            ):
                ok, message = await _make(tool).execute(**kwargs)
                assert ok is False
                assert "未启用" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_update_missing_task(self, tmp_path: Path) -> None:
        """更新不存在的任务时报错。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            tool = _make(UpdateCronTool)
            ok, message = await tool.execute(cron_ref="查无此项", prompt="x")
            assert ok is False
            assert "没有找到" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER

    @pytest.mark.asyncio
    async def test_update_after_task_removed(self, tmp_path: Path) -> None:
        """任务在解析后消失时报错。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        cron_tools._get_manager = lambda: manager  # type: ignore[assignment]
        try:
            record = manager.create(
                cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
            )
            tool = _make(UpdateCronTool)

            original_update = manager.update

            def update_then_delete(*args: object, **kwargs: object) -> None:
                original_update(*args, **kwargs)  # type: ignore[arg-type]
                manager.delete(str(record["cron_id"]))

            manager.update = update_then_delete  # type: ignore[method-assign]
            ok, message = await tool.execute(
                cron_ref=str(record["cron_id"]), prompt="x"
            )
            assert ok is False
            assert "任务不存在" in message
        finally:
            cron_tools._get_manager = _ORIGINAL_GET_MANAGER


# ==================== 描述辅助函数 ====================


class TestDescribeHelpers:
    """任务描述渲染。"""

    def test_describe_after(self) -> None:
        """after 类型渲染为分钟数。"""
        text = cron_tools._describe_schedule(  # noqa: SLF001
            {"schedule_kind": "after", "duration_seconds": 1800}
        )
        assert "30 分钟" in text

    def test_describe_at(self) -> None:
        """at 类型渲染为时间。"""
        text = cron_tools._describe_schedule(  # noqa: SLF001
            {"schedule_kind": "at", "absolute_at_ms": 1700000000000}
        )
        assert "一次性" in text

    def test_format_time_none(self) -> None:
        """空时间戳渲染为「无」。"""
        assert cron_tools._format_time(None) == "无"  # noqa: SLF001
        assert cron_tools._format_time(0) == "无"  # noqa: SLF001

    def test_format_detail_of_disabled(self, tmp_path: Path) -> None:
        """暂停任务在详情中标注。"""
        store = ScheduleStore(tmp_path / "db")
        manager = ScheduleManager(store)
        record = manager.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        manager.update(str(record["cron_id"]), enabled=False)
        detail = cron_tools._format_detail(manager.read_detail(str(record["cron_id"])))  # noqa: SLF001
        assert "已暂停" in detail

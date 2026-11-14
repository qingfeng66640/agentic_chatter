"""定时任务 LLM 工具测试。

覆盖四个工具的约束：参数互斥、会话策略固定、引用解析、暂停恢复与
删除保留历史。未启用定时任务时应明确拒绝而不是静默成功。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - 测试环境准备
    sys.path.insert(0, str(ROOT))

from ..task_runtime.schedule import (  # noqa: E402
    ScheduleManager,
    ScheduleStore,
    init_schedule_manager,
    shutdown_schedule_manager,
)
from ..tooling.cron_tools import (  # noqa: E402
    CreateCronTool,
    DeleteCronTool,
    ReadCronTool,
    UpdateCronTool,
)


@pytest.fixture
def enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ScheduleManager]:
    """构造一个已启用的全局管理器。"""
    manager = init_schedule_manager(ScheduleStore(tmp_path / "db"), submitter=None)
    from ..tooling import cron_tools

    monkeypatch.setattr(cron_tools, "_get_manager", lambda: manager)
    yield manager
    shutdown_schedule_manager()


def _make(tool_cls: type, stream_id: str = "stream-1") -> Any:
    """构造工具实例并绑定当前聊天流。

    Args:
        tool_cls: 工具类。
        stream_id: 伪装的当前聊天流。

    Returns:
        Any: 已绑定流 ID 的工具实例。
    """
    tool = tool_cls.__new__(tool_cls)  # noqa: SLF001 - 绕过框架构造，仅测 execute 逻辑
    tool.get_current_stream_id = lambda: stream_id  # type: ignore[method-assign]
    return tool


@pytest.fixture
def tools() -> None:
    """保留占位：工具实例在各自测试内构造。"""
    return None


@pytest.mark.asyncio
class TestCreateTool:
    """验证定时任务创建工具的参数约束和会话策略。"""

    async def test_creates_cron(self, enabled: ScheduleManager, tools: None) -> None:
        """创建循环任务成功。"""
        tool = _make(CreateCronTool)
        ok, message = await tool.execute(
            cron_name="工作日晨报",
            prompt="整理今日待办",
            schedule="0 9 * * 1-5",
            timezone_name="Asia/Shanghai",
        )
        assert ok is True
        assert "工作日晨报" in message
        assert len(enabled.read_list()) == 1

    async def test_rejects_multiple_triggers(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """同时给多个触发参数时报错。"""
        tool = _make(CreateCronTool)
        ok, message = await tool.execute(
            cron_name="n", prompt="p", schedule="0 9 * * *", after="30m"
        )
        assert ok is False
        assert "只能提供一个" in message
        assert enabled.read_list() == ()

    async def test_rejects_no_trigger(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """不给触发参数时报错。"""
        tool = _make(CreateCronTool)
        ok, _ = await tool.execute(cron_name="n", prompt="p")
        assert ok is False

    async def test_rejects_invalid_cron(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """非法 cron 被拒绝且不落库。"""
        tool = _make(CreateCronTool)
        ok, message = await tool.execute(
            cron_name="n", prompt="p", schedule="不是表达式"
        )
        assert ok is False
        assert enabled.read_list() == ()

    async def test_rejects_me_session(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """me 不是合法会话目标。"""
        tool = _make(CreateCronTool)
        ok, message = await tool.execute(
            cron_name="n",
            prompt="p",
            schedule="0 9 * * *",
            session_mode="session",
            session_target="me",
        )
        assert ok is False
        assert "me" in message
        assert enabled.read_list() == ()

    async def test_accepts_named_session_target(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """session 模式可指定具名历史标识。"""
        tool = _make(CreateCronTool)
        ok, _ = await tool.execute(
            cron_name="日报",
            prompt="p",
            schedule="0 9 * * *",
            session_mode="session",
            session_target="日报主线",
        )
        assert ok is True
        record = enabled.read_list()[0]
        assert record["session_mode"] == "session"
        assert record["session_target"] == "日报主线"

    async def test_reports_when_manager_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """功能未启用时明确报错。"""
        from ..tooling import cron_tools

        monkeypatch.setattr(cron_tools, "_get_manager", lambda: None)
        tool = _make(CreateCronTool)
        ok, message = await tool.execute(
            cron_name="n", prompt="p", schedule="0 9 * * *"
        )
        assert ok is False
        assert "未启用" in message


@pytest.mark.asyncio
class TestReadTool:
    """验证任务列表、详情与运行历史的查询结果。"""

    async def test_list_empty(self, enabled: ScheduleManager, tools: None) -> None:
        """没有任务时给出提示。"""
        tool = _make(ReadCronTool)
        ok, message = await tool.execute(view="list")
        assert ok is True
        assert "没有定时任务" in message

    async def test_list_and_detail(self, enabled: ScheduleManager, tools: None) -> None:
        """列表与详情都能读到。"""
        enabled.create(
            cron_name="巡检", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(ReadCronTool)
        ok, listing = await tool.execute(view="list")
        assert ok is True and "巡检" in listing

        ok, detail = await tool.execute(view="detail", cron_ref="巡检")
        assert ok is True and "0 9 * * *" in detail

    async def test_runs_view(self, enabled: ScheduleManager, tools: None) -> None:
        """运行历史视图可用。"""
        record = enabled.create(
            cron_name="巡检", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(ReadCronTool)
        ok, message = await tool.execute(view="runs", cron_ref=str(record["cron_id"]))
        assert ok is True

    async def test_detail_requires_ref(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """detail 缺少引用时提示。"""
        tool = _make(ReadCronTool)
        ok, message = await tool.execute(view="detail")
        assert ok is False
        assert "cron_ref" in message

    async def test_unknown_ref(self, enabled: ScheduleManager, tools: None) -> None:
        """引用不存在时报错。"""
        tool = _make(ReadCronTool)
        ok, _ = await tool.execute(view="detail", cron_ref="不存在")
        assert ok is False


@pytest.mark.asyncio
class TestUpdateTool:
    """验证定时规则更新、暂停和恢复的行为约束。"""

    async def test_update_prompt(self, enabled: ScheduleManager, tools: None) -> None:
        """修改指令成功且保留调度。"""
        record = enabled.create(
            cron_name="n", prompt="旧指令", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(UpdateCronTool)
        ok, message = await tool.execute(
            cron_ref=str(record["cron_id"]), prompt="新指令"
        )
        assert ok is True
        assert "新指令" in message
        assert "0 9 * * *" in message

    async def test_pause_and_resume(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """暂停后不再有待执行时刻，恢复后重算。"""
        record = enabled.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(UpdateCronTool)
        ok, _ = await tool.execute(cron_ref=str(record["cron_id"]), enabled="false")
        assert ok is True
        paused = enabled.read_detail(str(record["cron_id"]))
        assert paused is not None
        assert paused["next_run_at_ms"] is None

        ok, _ = await tool.execute(cron_ref=str(record["cron_id"]), enabled="true")
        assert ok is True
        resumed = enabled.read_detail(str(record["cron_id"]))
        assert resumed is not None
        assert resumed["next_run_at_ms"] is not None

    async def test_rejects_bad_enabled(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """非法 enabled 值被拒绝。"""
        record = enabled.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(UpdateCronTool)
        ok, message = await tool.execute(
            cron_ref=str(record["cron_id"]), enabled="maybe"
        )
        assert ok is False
        assert "true/false" in message

    async def test_cannot_change_session_mode(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """工具不暴露会话策略修改。"""
        record = enabled.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(UpdateCronTool)
        ok, _ = await tool.execute(cron_ref=str(record["cron_id"]), prompt="只改指令")
        assert ok is True
        refreshed = enabled.read_detail(str(record["cron_id"]))
        assert refreshed is not None
        assert refreshed["session_mode"] == "new"

    async def test_rejects_multiple_triggers(
        self, enabled: ScheduleManager, tools: None
    ) -> None:
        """同时给多个触发参数时报错。"""
        record = enabled.create(
            cron_name="n", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        tool = _make(UpdateCronTool)
        ok, message = await tool.execute(
            cron_ref=str(record["cron_id"]), schedule="0 10 * * *", after="1h"
        )
        assert ok is False
        assert "只能提供一个" in message


@pytest.mark.asyncio
class TestDeleteTool:
    """验证删除任务时保留历史及处理无效任务引用。"""

    async def test_delete_removes(self, enabled: ScheduleManager, tools: None) -> None:
        """删除后任务消失但历史保留。"""
        record = enabled.create(
            cron_name="巡检", prompt="p", stream_id="s", schedule="0 9 * * *"
        )
        enabled._store.claim_due(9_999_999_999_999)  # noqa: SLF001
        tool = _make(DeleteCronTool)
        ok, message = await tool.execute(cron_ref=str(record["cron_id"]))
        assert ok is True
        assert "已删除" in message
        assert enabled.read_list() == ()
        assert len(enabled.read_runs(str(record["cron_id"]))) >= 1

    async def test_delete_unknown(self, enabled: ScheduleManager, tools: None) -> None:
        """删除不存在的任务时报错。"""
        tool = _make(DeleteCronTool)
        ok, _ = await tool.execute(cron_ref="查无此项")
        assert ok is False


def test_manager_construction_without_submitter(tmp_path: Path) -> None:
    """管理器可在无提交回调时构造。"""
    manager = ScheduleManager(ScheduleStore(tmp_path / "d"))
    assert manager.is_busy() is False

"""定时任务服务层测试。

覆盖服务启停、数据库目录解析、以及触发时把任务提交到复杂任务运行时的
完整链路。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - 测试环境准备
    sys.path.insert(0, str(ROOT))

from agentic_chatter import schedule_service  # noqa: E402


class _FakeConfig:
    """最小化的插件配置替身。"""

    def __init__(self, tasks_dir: str = "", schedule_dir: str = "", enabled: bool = True):
        """构造配置。

        Args:
            tasks_dir: 任务检查点目录。
            schedule_dir: 定时任务数据库目录。
            enabled: 是否启用定时任务。
        """
        self.tasks = type("Tasks", (), {"checkpoint_directory": tasks_dir})()
        self.schedule = type(
            "Schedule", (), {"database_directory": schedule_dir, "enabled": enabled}
        )()


def test_database_directory_defaults_to_checkpoints() -> None:
    """未配置时复用任务检查点目录。"""
    directory = schedule_service.build_schedule_directory(_FakeConfig(tasks_dir="data/x"))
    assert directory is not None
    assert directory.endswith("schedules")
    assert "data/x" in directory


def test_database_directory_explicit_wins() -> None:
    """显式配置直接作为数据库目录，不再追加子目录。"""
    directory = schedule_service.build_schedule_directory(
        _FakeConfig(tasks_dir="data/x", schedule_dir="data/cron")
    )
    assert directory == "data/cron"


def test_database_directory_disabled() -> None:
    """关闭功能时不返回目录。"""
    assert (
        schedule_service.build_schedule_directory(
            _FakeConfig(tasks_dir="data/x", enabled=False)
        )
        is None
    )


def test_database_directory_missing_config() -> None:
    """缺少 schedule 配置段时不返回目录。"""
    assert schedule_service.build_schedule_directory(None) is None


def test_database_directory_empty_everywhere() -> None:
    """两处都为空时不启用。"""
    assert schedule_service.build_schedule_directory(_FakeConfig()) is None


class _FakeStore:
    """记录调用顺序的存储替身。"""

    def __init__(self, directory: str) -> None:
        """构造替身。

        Args:
            directory: 目录参数。
        """
        self.directory = directory
        self.reset_called = 0

    def reset_stale_claims(self, now_ms: int | None = None) -> int:
        """记录恢复调用。

        Args:
            now_ms: 当前时间戳。

        Returns:
            int: 恢复数量。
        """
        self.reset_called += 1
        return 0


class _FakeScheduler:
    """记录注册与注销的调度器替身。"""

    def __init__(self) -> None:
        """构造替身。"""
        self.created: list[dict[str, Any]] = []
        self.removed: list[str] = []

    async def create_schedule(self, **kwargs: Any) -> str:
        """记录创建的调度。

        Args:
            kwargs: 调度参数。

        Returns:
            str: 调度 ID。
        """
        self.created.append(kwargs)
        return "schedule-1"

    async def remove_schedule_by_name(self, task_name: str) -> bool:
        """记录注销的调度。

        Args:
            task_name: 调度名。

        Returns:
            bool: 是否注销成功。
        """
        self.removed.append(task_name)
        return True


@pytest.mark.asyncio
class TestServiceLifecycle:
    """服务启停。"""

    async def test_start_registers_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """启动后注册扫描调度并完成认领恢复。"""
        store = _FakeStore(str(tmp_path))
        scheduler = _FakeScheduler()
        manager_instance = _FakeManager()

        monkeypatch.setattr(
            schedule_service, "ScheduleStore", lambda directory: store
        )
        monkeypatch.setattr(
            schedule_service,
            "init_schedule_manager",
            lambda s, submitter=None: manager_instance,
        )
        monkeypatch.setattr(
            schedule_service, "get_unified_scheduler", lambda: scheduler
        )

        plugin = type("P", (), {"config": _FakeConfig(tasks_dir=str(tmp_path))})()
        await schedule_service.start_schedule_service(plugin)

        assert len(scheduler.created) == 1
        assert scheduler.created[0]["task_name"] == schedule_service.SCHEDULER_TASK_NAME
        assert scheduler.created[0]["is_recurring"] is True
        # 认领恢复由管理器发起
        assert manager_instance.recovered == 1

    async def test_start_disabled_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """功能关闭时不注册任何调度。"""
        scheduler = _FakeScheduler()
        monkeypatch.setattr(
            schedule_service, "get_unified_scheduler", lambda: scheduler
        )
        plugin = type("P", (), {"config": _FakeConfig(enabled=False)})()
        await schedule_service.start_schedule_service(plugin)
        assert scheduler.created == []

    async def test_start_survives_store_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """数据库初始化失败时不阻断插件加载。"""

        def boom(directory: str) -> None:
            raise RuntimeError("无法创建数据库")

        monkeypatch.setattr(schedule_service, "ScheduleStore", boom)
        plugin = type("P", (), {"config": _FakeConfig(tasks_dir=str(tmp_path))})()
        await schedule_service.start_schedule_service(plugin)

    async def test_start_survives_scheduler_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """调度注册失败时释放管理器并保持插件可用。"""
        store = _FakeStore(str(tmp_path))
        released: list[bool] = []

        class _BoomScheduler:
            async def create_schedule(self, **kwargs: Any) -> str:
                raise RuntimeError("内核调度器不可用")

        monkeypatch.setattr(
            schedule_service, "ScheduleStore", lambda directory: store
        )
        monkeypatch.setattr(
            schedule_service,
            "init_schedule_manager",
            lambda s, submitter=None: _FakeManager(),
        )
        monkeypatch.setattr(
            schedule_service,
            "shutdown_schedule_manager",
            lambda: released.append(True),
        )
        monkeypatch.setattr(
            schedule_service, "get_unified_scheduler", lambda: _BoomScheduler()
        )

        plugin = type("P", (), {"config": _FakeConfig(tasks_dir=str(tmp_path))})()
        await schedule_service.start_schedule_service(plugin)
        assert released == [True]

    async def test_stop_unregisters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """停止时注销扫描调度。"""
        scheduler = _FakeScheduler()
        monkeypatch.setattr(
            schedule_service, "get_unified_scheduler", lambda: scheduler
        )
        monkeypatch.setattr(
            schedule_service, "get_schedule_manager", lambda: _FakeManager()
        )
        released: list[bool] = []
        monkeypatch.setattr(
            schedule_service,
            "shutdown_schedule_manager",
            lambda: released.append(True),
        )
        await schedule_service.stop_schedule_service()
        assert scheduler.removed == [schedule_service.SCHEDULER_TASK_NAME]
        assert released == [True]
    async def test_stop_without_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """未启动时停止是安全的空操作。"""
        monkeypatch.setattr(
            schedule_service, "get_schedule_manager", lambda: None
        )
        released: list[bool] = []
        monkeypatch.setattr(
            schedule_service,
            "shutdown_schedule_manager",
            lambda: released.append(True),
        )
        await schedule_service.stop_schedule_service()
        assert released == [True]

    async def test_stop_survives_removal_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """注销失败不阻断卸载。"""

        class _BoomScheduler:
            async def remove_schedule_by_name(self, task_name: str) -> bool:
                raise RuntimeError("注销失败")

        monkeypatch.setattr(
            schedule_service, "get_unified_scheduler", lambda: _BoomScheduler()
        )
        monkeypatch.setattr(
            schedule_service, "get_schedule_manager", lambda: _FakeManager()
        )
        released: list[bool] = []
        monkeypatch.setattr(
            schedule_service,
            "shutdown_schedule_manager",
            lambda: released.append(True),
        )
        await schedule_service.stop_schedule_service()
        assert released == [True]


class _FakeManager:
    """管理器替身。"""

    def __init__(self) -> None:
        """构造替身。"""
        self.recovered = 0

    def recover_stale_claims(self) -> int:
        """记录恢复调用。

        Returns:
            int: 恢复数量。
        """
        self.recovered += 1
        return 0

    async def scan_once(self) -> int:
        """扫描回调替身。

        Returns:
            int: 本次提交数量。
        """
        return 0


class TestSubmitter:
    """触发提交链路。"""

    def test_submitter_reports_capacity_and_creates_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """触发时在目标流创建复杂任务并注入固定线程上下文。"""
        plugin = type("P", (), {"config": _FakeConfig(tasks_dir=str(tmp_path))})()
        submit = schedule_service._make_submitter(plugin)  # noqa: SLF001

        created: list[dict[str, Any]] = []

        class _Runtime:
            def __init__(self, task_id: str) -> None:
                self.state = type("S", (), {"task_id": task_id})()

            def start(self) -> None:
                """对齐 TaskRuntime.start 的空实现。"""

        class _RuntimeManager:
            def get_active_tasks(self, stream_id: str) -> list[Any]:
                return []

            def create(self, stream_id: str, goal: str, **kwargs: Any) -> _Runtime:
                created.append({"stream_id": stream_id, "goal": goal})
                return _Runtime("task-42")

        started: list[Any] = []

        monkeypatch.setattr(
            "agentic_chatter.task_runtime.runtime.get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )
        monkeypatch.setattr(
            "agentic_chatter.task_runtime.runtime.get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )
        monkeypatch.setattr(
            "agentic_chatter.task_runtime.coordinator.start_task",
            lambda *args: started.append(args),
        )
        monkeypatch.setattr(
            "agentic_chatter.chatter.AgenticChatter",
            lambda stream_id, plugin_arg: object(),
        )

        import asyncio

        result = asyncio.run(
            submit(
                stream_id="s1",
                prompt="做点什么",
                context="历史记录",
                cron_id="c1",
                record={"cron_name": "巡检"},
            )
        )
        assert result == "task-42"
        assert created[0]["stream_id"] == "s1"
        assert len(started) == 1

    def test_submitter_skips_when_cap_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """活动任务达上限时跳过触发。"""
        plugin = type("P", (), {"config": _FakeConfig(tasks_dir=str(tmp_path))})()
        submit = schedule_service._make_submitter(plugin)  # noqa: SLF001

        class _RuntimeManager:
            def get_active_tasks(self, stream_id: str) -> list[Any]:
                return [object(), object()]

        monkeypatch.setattr(
            "agentic_chatter.task_runtime.runtime.get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )
        monkeypatch.setattr(
            "agentic_chatter.task_runtime.runtime.get_task_runtime_manager",
            lambda: _RuntimeManager(),
        )

        import asyncio

        result = asyncio.run(
            submit(
                stream_id="s1",
                prompt="p",
                context="",
                cron_id="c1",
                record={},
            )
        )
        assert result == ""

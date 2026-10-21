"""定时任务的 SQLite 持久化。

数据库文件放在插件配置的 ``tasks.checkpoint_directory`` 同级目录，
与任务检查点并列。共三张表：

- ``schedules``：任务定义，含 ``generation`` 世代计数与 ``next_run_at_ms``。
- ``schedule_runs``：单次运行记录，含认领时间戳用于单次认领。
- ``schedule_threads``：固定线程模式下历次运行结果的裁剪历史。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# 任务状态：active 仍在排程，其余三态不再有待执行时刻
STATE_ACTIVE = "active"
STATE_COMPLETED = "completed"
STATE_EXPIRED = "expired"
STATE_CANCELLED = "cancelled"

# 运行状态：pending -> delivered / failed
RUN_PENDING = "pending"
RUN_DELIVERED = "delivered"
RUN_FAILED = "failed"

# 认领后写入的远端哨兵值：表示「已被认领、真实下次时刻待重算」，
# 使同一触发点不会被随后的扫描重复认领。该值远大于任何真实时间戳。
CLAIMED_SENTINEL_MS = 4_102_444_800_000  # 2100-01-01T00:00:00Z

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    cron_id            TEXT PRIMARY KEY,
    cron_name          TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    stream_id          TEXT NOT NULL,
    session_mode       TEXT NOT NULL,
    session_target     TEXT NOT NULL DEFAULT '',
    schedule_kind      TEXT NOT NULL,
    cron_expression    TEXT NOT NULL DEFAULT '',
    duration_seconds   REAL NOT NULL DEFAULT 0,
    absolute_at_ms     INTEGER NOT NULL DEFAULT 0,
    timezone_name      TEXT NOT NULL DEFAULT '',
    enabled            INTEGER NOT NULL DEFAULT 1,
    generation         INTEGER NOT NULL DEFAULT 0,
    state              TEXT NOT NULL DEFAULT 'active',
    next_run_at_ms     INTEGER,
    run_count          INTEGER NOT NULL DEFAULT 0,
    created_at_ms      INTEGER NOT NULL,
    updated_at_ms      INTEGER NOT NULL,
    deleted_at_ms      INTEGER
);

CREATE TABLE IF NOT EXISTS schedule_runs (
    run_id             TEXT PRIMARY KEY,
    cron_id            TEXT NOT NULL,
    trigger_source     TEXT NOT NULL DEFAULT 'scheduled',
    status             TEXT NOT NULL DEFAULT 'pending',
    scheduled_at_ms    INTEGER NOT NULL,
    created_at_ms      INTEGER NOT NULL,
    claimed_at_ms      INTEGER,
    delivered_at_ms    INTEGER,
    failed_at_ms       INTEGER,
    error_code         TEXT NOT NULL DEFAULT '',
    error              TEXT NOT NULL DEFAULT '',
    task_id            TEXT
);

CREATE TABLE IF NOT EXISTS schedule_threads (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cron_id            TEXT NOT NULL,
    task_id            TEXT NOT NULL DEFAULT '',
    created_at_ms      INTEGER NOT NULL,
    summary            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_runs_cron ON schedule_runs (cron_id, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_threads_cron ON schedule_threads (cron_id, id DESC);
"""


def _now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    import time

    return int(time.time() * 1000)


class ScheduleStore:
    """定时任务的 SQLite 存储。"""

    def __init__(self, directory: str | Path, filename: str = "schedules.sqlite3") -> None:
        """初始化存储。

        Args:
            directory: 数据库所在目录，留空则构造失败。
            filename: 数据库文件名。
        """
        root = Path(str(directory)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / filename
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """打开一个受锁保护的连接。"""
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=10.0)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA journal_mode=WAL")
                yield connection
                connection.commit()
            finally:
                connection.close()

    # ==================== 任务定义 ====================

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        """写入一条任务定义。"""
        now = _now_ms()
        payload = dict(record)
        payload.setdefault("enabled", 1)
        payload.setdefault("generation", 0)
        payload.setdefault("state", STATE_ACTIVE)
        payload.setdefault("run_count", 0)
        payload.setdefault("session_target", "")
        payload.setdefault("cron_expression", "")
        payload.setdefault("duration_seconds", 0.0)
        payload.setdefault("absolute_at_ms", 0)
        payload.setdefault("timezone_name", "")
        payload["created_at_ms"] = now
        payload["updated_at_ms"] = now
        payload["enabled"] = 1 if payload["enabled"] else 0
        columns = ",".join(payload)
        placeholders = ",".join("?" for _ in payload)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO schedules ({columns}) VALUES ({placeholders})",  # noqa: S608 - 列名来自固定白名单
                tuple(payload.values()),
            )
        return self.get(str(payload["cron_id"])) or {}

    def get(self, cron_id: str) -> dict[str, Any] | None:
        """按 ID 读取任务定义。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM schedules WHERE cron_id = ? AND deleted_at_ms IS NULL",
                (cron_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_by_name(self, cron_name: str) -> dict[str, Any] | None:
        """按名称读取任务定义。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM schedules WHERE cron_name = ? AND deleted_at_ms IS NULL",
                (cron_name,),
            ).fetchone()
        return dict(row) if row else None

    def list_all(self, *, state: str | None = None) -> tuple[dict[str, Any], ...]:
        """列出未删除的任务定义。"""
        query = "SELECT * FROM schedules WHERE deleted_at_ms IS NULL"
        params: tuple[Any, ...] = ()
        if state:
            query += " AND state = ?"
            params = (state,)
        query += " ORDER BY created_at_ms ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(dict(row) for row in rows)

    def update(self, cron_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        """更新任务字段；自动递增 updated_at。"""
        if not changes:
            return self.get(cron_id)
        changes = dict(changes)
        changes["updated_at_ms"] = _now_ms()
        if "enabled" in changes:
            changes["enabled"] = 1 if changes["enabled"] else 0
        assignments = ",".join(f"{key} = ?" for key in changes)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE schedules SET {assignments} WHERE cron_id = ?",  # noqa: S608 - 列名来自固定白名单
                (*changes.values(), cron_id),
            )
        return self.get(cron_id)

    def bump_generation(self, cron_id: str) -> int:
        """递增世代计数，返回新值。

        用于检测调度器是否持有过期时间表。
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE schedules SET generation = generation + 1, updated_at_ms = ? "
                "WHERE cron_id = ?",
                (_now_ms(), cron_id),
            )
            row = connection.execute(
                "SELECT generation FROM schedules WHERE cron_id = ?", (cron_id,)
            ).fetchone()
        return int(row["generation"]) if row else 0

    def soft_delete(self, cron_id: str) -> bool:
        """软删除任务定义，保留运行历史。"""
        now = _now_ms()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE schedules SET deleted_at_ms = ?, state = ?, next_run_at_ms = NULL "
                "WHERE cron_id = ? AND deleted_at_ms IS NULL",
                (now, STATE_CANCELLED, cron_id),
            )
        return cursor.rowcount > 0

    # ==================== 到期扫描与认领 ====================

    def claim_due(self, now_ms: int, *, limit: int = 16) -> tuple[dict[str, Any], ...]:
        """认领到期任务。

        仅返回 ``enabled`` 且 ``state='active'`` 且 ``next_run_at_ms`` 不为
        NULL 的任务。认领动作会立刻把 ``next_run_at_ms`` 推到一个远端哨兵
        值（``CLAIMED_SENTINEL_MS``），使同一触发点不会被后续扫描重复认领；
        调度器在真正提交后会按新表达式重算真实的下次触发时刻。

        Args:
            now_ms: 当前时间戳。
            limit: 单次最多认领数量。

        Returns:
            tuple: 本次认领到的任务定义。
        """
        claimed: list[dict[str, Any]] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM schedules WHERE deleted_at_ms IS NULL "
                "AND enabled = 1 AND state = ? AND next_run_at_ms IS NOT NULL "
                "AND next_run_at_ms <= ? ORDER BY next_run_at_ms ASC LIMIT ?",
                (STATE_ACTIVE, now_ms, int(limit)),
            ).fetchall()
            for row in rows:
                record = dict(row)
                if not self._insert_run(connection, record, now_ms):
                    continue
                claimed.append(record)
        return tuple(claimed)

    def _insert_run(
        self, connection: sqlite3.Connection, record: dict[str, Any], now_ms: int
    ) -> bool:
        """为一次触发写入运行记录并推进待执行时刻。

        Returns:
            bool: 认领成功返回 True；已被其他执行者抢先时返回 False。
        """
        import uuid

        cron_id = str(record["cron_id"])
        # 条件更新即「单次认领」：只有仍处于同一触发时刻的行才会被推进，
        # 并发的第二次扫描会因条件不匹配而拿到 0 行，从而不会重复触发。
        claimed = connection.execute(
            "UPDATE schedules SET next_run_at_ms = ?, updated_at_ms = ? "
            "WHERE cron_id = ? AND next_run_at_ms IS NOT NULL AND next_run_at_ms <= ?",
            (CLAIMED_SENTINEL_MS, now_ms, cron_id, now_ms),
        )
        if claimed.rowcount == 0:  # pragma: no cover - 并发抢占兜底
            return False

        run_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO schedule_runs (run_id, cron_id, trigger_source, status, "
            "scheduled_at_ms, created_at_ms, claimed_at_ms) VALUES (?, ?, 'scheduled', ?, ?, ?, ?)",
            (
                run_id,
                cron_id,
                RUN_PENDING,
                int(record["next_run_at_ms"] or now_ms),
                now_ms,
                now_ms,
            ),
        )
        connection.execute(
            "UPDATE schedules SET run_count = run_count + 1, updated_at_ms = ? WHERE cron_id = ?",
            (now_ms, cron_id),
        )
        return True

    def mark_run(self, run_cron_id: str, *, status: str, error: str = "", task_id: str = "") -> None:
        """更新运行结果。

        Args:
            run_cron_id: 运行所属任务 ID（用于定位其待认领记录）。
            status: ``delivered`` 或 ``failed``。
            error: 失败原因。
            task_id: 产生的任务 ID。
        """
        now = _now_ms()
        stamp_column = "delivered_at_ms" if status == RUN_DELIVERED else "failed_at_ms"
        with self._connect() as connection:
            # SQLite 的 UPDATE 不支持 ORDER BY/LIMIT，先取最新一条待认领
            # 运行的 ID，再按主键精确更新，保证并发下不会串改。
            row = connection.execute(
                "SELECT run_id FROM schedule_runs WHERE cron_id = ? AND status = ? "
                "ORDER BY created_at_ms DESC LIMIT 1",
                (run_cron_id, RUN_PENDING),
            ).fetchone()
            if row is None:
                return
            connection.execute(
                f"UPDATE schedule_runs SET status = ?, {stamp_column} = ?, "  # noqa: S608 - 列名为固定常量
                "error = ?, task_id = ? WHERE run_id = ?",
                (status, now, error, task_id, row["run_id"]),
            )

    def set_next_run(self, cron_id: str, next_run_at_ms: int | None) -> None:
        """写入下一次触发时刻。

        Args:
            cron_id: 任务 ID。
            next_run_at_ms: 下一次时刻；``None`` 表示不再排程。
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE schedules SET next_run_at_ms = ?, updated_at_ms = ? WHERE cron_id = ?",
                (next_run_at_ms, _now_ms(), cron_id),
            )

    def finish_state(self, cron_id: str, state: str) -> None:
        """把任务置为终态并清空待执行时刻。"""
        with self._connect() as connection:
            connection.execute(
                "UPDATE schedules SET state = ?, next_run_at_ms = NULL, updated_at_ms = ? "
                "WHERE cron_id = ?",
                (state, _now_ms(), cron_id),
            )

    def reset_stale_claims(self, now_ms: int | None = None) -> int:
        """恢复因进程中断而滞留在哨兵值的任务。

        若进程在「认领成功」与「重算下次触发时刻」之间退出，该任务的
        ``next_run_at_ms`` 会停在远端哨兵上而永不触发。启动时把它们
        拉回当前时刻，使其重新参与排程；由于是按当前时间而非原触发点
        重排，符合 catch-up 不回放的约定。

        Args:
            now_ms: 当前时间戳，缺省取系统时间。

        Returns:
            int: 被恢复的任务数量。
        """
        moment = int(now_ms if now_ms is not None else _now_ms())
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE schedules SET next_run_at_ms = ?, updated_at_ms = ? "
                "WHERE next_run_at_ms = ? AND state = ? AND enabled = 1 "
                "AND deleted_at_ms IS NULL",
                (moment, _now_ms(), CLAIMED_SENTINEL_MS, STATE_ACTIVE),
            )
        return cursor.rowcount

    # ==================== 运行历史 ====================

    def list_runs(self, cron_id: str, *, limit: int = 20) -> tuple[dict[str, Any], ...]:
        """列出某任务的运行历史。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM schedule_runs WHERE cron_id = ? "
                "ORDER BY created_at_ms DESC LIMIT ?",
                (cron_id, int(limit)),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    # ==================== 固定线程历史 ====================

    def append_thread_entry(self, cron_id: str, summary: str, *, max_entries: int = 10) -> None:
        """追加一条固定线程历史，并按条数裁剪。"""
        now = _now_ms()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO schedule_threads (cron_id, created_at_ms, summary) VALUES (?, ?, ?)",
                (cron_id, now, summary),
            )
            connection.execute(
                "DELETE FROM schedule_threads WHERE cron_id = ? AND id NOT IN ("
                "SELECT id FROM schedule_threads WHERE cron_id = ? ORDER BY id DESC LIMIT ?)",
                (cron_id, cron_id, int(max_entries)),
            )

    def list_thread_entries(self, cron_id: str, *, limit: int = 10) -> tuple[dict[str, Any], ...]:
        """读取固定线程的历史（时间正序，便于注入上下文）。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM schedule_threads WHERE cron_id = ? ORDER BY id DESC LIMIT ?",
                (cron_id, int(limit)),
            ).fetchall()
        return tuple(dict(row) for row in reversed(rows))

    def drop_thread_entries(self, cron_id: str) -> None:
        """删除某任务的全部线程历史。"""
        with self._connect() as connection:
            connection.execute("DELETE FROM schedule_threads WHERE cron_id = ?", (cron_id,))


def describe_run(run: dict[str, Any]) -> str:
    """把运行记录渲染为可读文本。"""
    from datetime import datetime, timezone

    moment = run.get("scheduled_at_ms") or run.get("created_at_ms") or 0
    stamp = datetime.fromtimestamp(int(moment) / 1000, tz=timezone.utc).astimezone()
    status = run.get("status", "")
    text = f"[{stamp:%Y-%m-%d %H:%M:%S}] {status}"
    if run.get("task_id"):
        text += f" task={run['task_id']}"
    if run.get("error"):
        text += f" error={run['error']}"
    return text


def _unused_serialize(value: Any) -> str:  # pragma: no cover - 占位以保持模块导出稳定
    """序列化辅助函数。"""
    return json.dumps(value, ensure_ascii=False)

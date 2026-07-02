"""检查点持久化 — SQLite WAL 模式,任务每轮落盘,崩溃后可恢复。

并发防护:每个任务附带租约(lease)。执行前必须先获取租约,
两个进程同时 run/resume 同一 task_id 时,后到者拿不到租约,
抛 LeaseConflict 而不是产生双写。租约带 TTL,持有者崩溃后自动过期。

生产替换建议:Postgres / DynamoDB,接口保持不变即可。
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from typing import Any

from .types import Message, Role, TaskState, TaskStatus, ToolCall, ToolResult, Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id          TEXT PRIMARY KEY,
    status           TEXT NOT NULL,
    state_json       TEXT NOT NULL,
    updated_at       REAL NOT NULL,
    lease_owner      TEXT,
    lease_expires    REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status ON checkpoints(status);
"""

_MIGRATION_COLUMNS = (
    "lease_owner TEXT",
    "lease_expires REAL",
    "cancel_requested INTEGER NOT NULL DEFAULT 0",
)


class LeaseConflict(Exception):
    """任务租约被其他执行者持有,拒绝并发执行。"""


def _state_to_json(state: TaskState) -> str:
    return json.dumps(dataclasses.asdict(state), ensure_ascii=False)


def _state_from_json(raw: str) -> TaskState:
    d: dict[str, Any] = json.loads(raw)
    messages = []
    for m in d.get("messages", []):
        messages.append(Message(
            role=Role(m["role"]),
            content=m.get("content", ""),
            tool_calls=[ToolCall(**tc) for tc in m.get("tool_calls", [])],
            tool_results=[ToolResult(**tr) for tr in m.get("tool_results", [])],
            thinking_blocks=m.get("thinking_blocks", []),
        ))
    return TaskState(
        task_id=d["task_id"],
        goal=d["goal"],
        status=TaskStatus(d["status"]),
        messages=messages,
        iterations=d["iterations"],
        usage=Usage(**d["usage"]),  # 旧检查点缺少缓存字段时默认 0
        cost_usd=d["cost_usd"],
        error=d.get("error"),
        pending_approval_ids=d.get("pending_approval_ids", []),
        output_schema=d.get("output_schema"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


class CheckpointStore:
    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # 兼容旧库:补充后加的列(已存在时静默跳过)
        for col in _MIGRATION_COLUMNS:
            try:
                self._conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass

    def save(self, state: TaskState) -> None:
        state.updated_at = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints(task_id,status,state_json,updated_at) "
                "VALUES(?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "status=excluded.status, state_json=excluded.state_json, "
                "updated_at=excluded.updated_at",
                (state.task_id, state.status.value, _state_to_json(state), state.updated_at),
            )

    def load(self, task_id: str) -> TaskState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM checkpoints WHERE task_id=?", (task_id,)
            ).fetchone()
        return _state_from_json(row[0]) if row else None

    def list_resumable(self, statuses: tuple[TaskStatus, ...] = (
            TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL)) -> list[str]:
        """列出处于给定状态的任务(默认 running + 等待审批)。"""
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT task_id FROM checkpoints WHERE status IN ({marks})",
                [s.value for s in statuses],
            ).fetchall()
        return [r[0] for r in rows]

    # ---------------- 取消 ----------------

    def request_cancel(self, task_id: str) -> bool:
        """标记任务请求取消;执行循环在下一轮开始前生效。跨进程可用。"""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE checkpoints SET cancel_requested=1 WHERE task_id=?",
                (task_id,))
            return cur.rowcount > 0

    def cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM checkpoints WHERE task_id=?",
                (task_id,)).fetchone()
        return bool(row and row[0])

    # ---------------- 租约 ----------------

    def acquire_lease(self, task_id: str, owner: str, ttl_s: float) -> bool:
        """获取/续期任务租约。空闲、已过期或本就属于 owner 时成功。"""
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE checkpoints SET lease_owner=?, lease_expires=? "
                "WHERE task_id=? AND (lease_owner IS NULL OR lease_owner=? "
                "OR lease_expires IS NULL OR lease_expires<?)",
                (owner, now + ttl_s, task_id, owner, now),
            )
            return cur.rowcount > 0

    def release_lease(self, task_id: str, owner: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE checkpoints SET lease_owner=NULL, lease_expires=NULL "
                "WHERE task_id=? AND lease_owner=?",
                (task_id, owner),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

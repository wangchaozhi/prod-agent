"""Demo business tools backed by SQLite.

These tools are intentionally small but shaped like real integrations:
READ tools query orders, WRITE tools create internal records, and EXECUTE
tools model external side effects that should pass through approval.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from .tools import ToolRegistry
from .types import Permission


_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id      TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    phone         TEXT NOT NULL,
    status        TEXT NOT NULL,
    total         REAL NOT NULL,
    address       TEXT NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL,
    sku      TEXT NOT NULL,
    name     TEXT NOT NULL,
    qty      INTEGER NOT NULL,
    price    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL,
    issue        TEXT NOT NULL,
    priority     TEXT NOT NULL,
    status       TEXT NOT NULL,
    tool_call_id TEXT UNIQUE,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    message_id   TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL,
    channel      TEXT NOT NULL,
    content      TEXT NOT NULL,
    tool_call_id TEXT UNIQUE,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id    TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL,
    amount       REAL NOT NULL,
    reason       TEXT NOT NULL,
    status       TEXT NOT NULL,
    tool_call_id TEXT UNIQUE,
    created_at   REAL NOT NULL
);
"""


class DemoBusinessStore:
    def __init__(self, db_path: str = "demo_business.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._seed()

    def _seed(self) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()
            if row and row["n"]:
                return
            now = time.time()
            orders = [
                ("A1001", "Li Lei", "13800000001", "paid", 299.0,
                 "Shanghai Pudong Demo Road 1", now),
                ("A1002", "Han Mei", "13800000002", "shipped", 129.5,
                 "Beijing Haidian Demo Street 8", now),
                ("B2001", "Acme Corp", "010-88880000", "delivered", 879.0,
                 "Shenzhen Nanshan Demo Park", now),
            ]
            items = [
                ("A1001", "SKU-KEYBOARD", "Mechanical Keyboard", 1, 199.0),
                ("A1001", "SKU-CABLE", "USB-C Cable", 2, 50.0),
                ("A1002", "SKU-MOUSE", "Wireless Mouse", 1, 129.5),
                ("B2001", "SKU-MONITOR", "27 inch Monitor", 1, 879.0),
            ]
            self._conn.executemany(
                "INSERT INTO orders VALUES(?,?,?,?,?,?,?)",
                orders,
            )
            self._conn.executemany(
                "INSERT INTO order_items VALUES(?,?,?,?,?)",
                items,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _order_exists(self, order_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM orders WHERE order_id=?", (order_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"order {order_id} not found")

    @staticmethod
    def _dump(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    def query_order(self, order_id: str) -> str:
        with self._lock:
            order = self._conn.execute(
                "SELECT * FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if order is None:
                raise ValueError(f"order {order_id} not found")
            items = self._conn.execute(
                "SELECT sku,name,qty,price FROM order_items WHERE order_id=?",
                (order_id,),
            ).fetchall()
        data = dict(order)
        data["items"] = [dict(i) for i in items]
        return self._dump({"ok": True, "order": data})

    def search_orders(self, keyword: str = "") -> str:
        keyword = (keyword or "").strip()
        with self._lock:
            if keyword:
                rows = self._conn.execute(
                    "SELECT * FROM orders WHERE order_id LIKE ? "
                    "OR customer_name LIKE ? OR phone LIKE ? "
                    "ORDER BY updated_at DESC LIMIT 20",
                    (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM orders ORDER BY updated_at DESC LIMIT 20"
                ).fetchall()
        return self._dump({"ok": True, "orders": [dict(r) for r in rows]})

    def update_order_status(self, order_id: str, status: str) -> str:
        with self._lock, self._conn:
            self._order_exists(order_id)
            self._conn.execute(
                "UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
                (status, time.time(), order_id),
            )
        return self._dump({"ok": True, "order_id": order_id, "status": status})

    def create_ticket(self, order_id: str, issue: str, priority: str = "normal",
                      tool_call_id: str | None = None) -> str:
        with self._lock, self._conn:
            self._order_exists(order_id)
            if tool_call_id:
                existing = self._conn.execute(
                    "SELECT * FROM tickets WHERE tool_call_id=?", (tool_call_id,)
                ).fetchone()
                if existing:
                    return self._dump({"ok": True, "ticket": dict(existing),
                                       "idempotent": True})
            ticket_id = "T-" + uuid.uuid4().hex[:10]
            row = (ticket_id, order_id, issue, priority, "open",
                   tool_call_id, time.time())
            self._conn.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?)", row)
        return self._dump({
            "ok": True,
            "ticket": {
                "ticket_id": ticket_id,
                "order_id": order_id,
                "issue": issue,
                "priority": priority,
                "status": "open",
            },
        })

    def send_customer_message(self, order_id: str, channel: str, content: str,
                              tool_call_id: str | None = None) -> str:
        with self._lock, self._conn:
            self._order_exists(order_id)
            if tool_call_id:
                existing = self._conn.execute(
                    "SELECT * FROM notifications WHERE tool_call_id=?",
                    (tool_call_id,),
                ).fetchone()
                if existing:
                    return self._dump({"ok": True, "message": dict(existing),
                                       "idempotent": True})
            message_id = "M-" + uuid.uuid4().hex[:10]
            row = (message_id, order_id, channel, content, tool_call_id, time.time())
            self._conn.execute("INSERT INTO notifications VALUES(?,?,?,?,?,?)", row)
        return self._dump({
            "ok": True,
            "message": {
                "message_id": message_id,
                "order_id": order_id,
                "channel": channel,
                "content": content,
            },
        })

    def refund_order(self, order_id: str, amount: float, reason: str,
                     tool_call_id: str | None = None) -> str:
        if amount <= 0:
            raise ValueError("refund amount must be positive")
        with self._lock, self._conn:
            self._order_exists(order_id)
            order = self._conn.execute(
                "SELECT total FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if order is None:
                raise ValueError(f"order {order_id} not found")
            if amount > float(order["total"]):
                raise ValueError("refund amount cannot exceed order total")
            if tool_call_id:
                existing = self._conn.execute(
                    "SELECT * FROM refunds WHERE tool_call_id=?", (tool_call_id,)
                ).fetchone()
                if existing:
                    return self._dump({"ok": True, "refund": dict(existing),
                                       "idempotent": True})
            refund_id = "R-" + uuid.uuid4().hex[:10]
            row = (refund_id, order_id, amount, reason, "submitted",
                   tool_call_id, time.time())
            self._conn.execute("INSERT INTO refunds VALUES(?,?,?,?,?,?,?)", row)
            if amount == float(order["total"]):
                self._conn.execute(
                    "UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
                    ("refund_submitted", time.time(), order_id),
                )
        return self._dump({
            "ok": True,
            "refund": {
                "refund_id": refund_id,
                "order_id": order_id,
                "amount": amount,
                "reason": reason,
                "status": "submitted",
            },
        })


def build_demo_tools(db_path: str = "demo_business.db",
                     approval_hook=None,
                     require_approval_for_execute: bool = True) -> ToolRegistry:
    store = DemoBusinessStore(db_path)
    tools = ToolRegistry(
        require_approval_for_execute=require_approval_for_execute,
        approval_hook=approval_hook,
    )

    @tools.tool(
        name="query_order",
        description="Look up one demo order by order_id.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        permission=Permission.READ,
    )
    def query_order(order_id: str) -> str:
        return store.query_order(order_id)

    @tools.tool(
        name="search_orders",
        description="Search demo orders by order id, customer name, or phone.",
        input_schema={
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
        },
        permission=Permission.READ,
    )
    def search_orders(keyword: str = "") -> str:
        return store.search_orders(keyword)

    @tools.tool(
        name="update_order_status",
        description="Update a demo order status.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["order_id", "status"],
        },
        permission=Permission.WRITE,
    )
    def update_order_status(order_id: str, status: str) -> str:
        return store.update_order_status(order_id, status)

    @tools.tool(
        name="create_ticket",
        description="Create an internal support ticket for a demo order.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "issue": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["order_id", "issue"],
        },
        permission=Permission.WRITE,
        pass_call_id=True,
    )
    def create_ticket(order_id: str, issue: str, priority: str = "normal",
                      tool_call_id: str | None = None) -> str:
        return store.create_ticket(order_id, issue, priority, tool_call_id)

    @tools.tool(
        name="send_customer_message",
        description="Send a demo customer message. This models an external side effect.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "channel": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["order_id", "channel", "content"],
        },
        permission=Permission.EXECUTE,
        pass_call_id=True,
    )
    def send_customer_message(order_id: str, channel: str, content: str,
                              tool_call_id: str | None = None) -> str:
        return store.send_customer_message(order_id, channel, content, tool_call_id)

    @tools.tool(
        name="refund_order",
        description="Submit a demo refund. This models a high-risk external action.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount", "reason"],
        },
        permission=Permission.EXECUTE,
        pass_call_id=True,
    )
    def refund_order(order_id: str, amount: float, reason: str,
                     tool_call_id: str | None = None) -> str:
        return store.refund_order(order_id, amount, reason, tool_call_id)

    return tools

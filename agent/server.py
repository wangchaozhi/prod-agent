"""Small HTTP API and web UI for prod-agent.

The server uses only the Python standard library. It is meant as a practical
product shell around the runtime: submit tasks, inspect checkpoints, cancel
work, and approve EXECUTE tool calls.
"""
from __future__ import annotations

import argparse
import dataclasses
import hmac
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .config import AgentConfig
from .core import Agent
from .demo_tools import build_demo_tools
from .llm import AnthropicProvider, LLMProvider, OnText
from .llm_openai import OpenAICompatProvider
from .tools import ToolRegistry
from .types import LLMResponse, Message, Role, TaskState, TaskStatus, ToolCall, Usage

MAX_JSON_BODY_BYTES = 1_000_000
MAX_GOAL_CHARS = 20_000


class APIError(Exception):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


class AuthConfig:
    def __init__(self, api_token: str = "", approval_token: str = "") -> None:
        self.api_token = api_token
        self.approval_token = approval_token

    @property
    def enabled(self) -> bool:
        return bool(self.api_token or self.approval_token)

    def role_for(self, header: str | None) -> str | None:
        if not self.enabled:
            return "approve"
        if not header:
            raise APIError(HTTPStatus.UNAUTHORIZED, "unauthorized",
                           "missing Bearer token")
        prefix = "Bearer "
        if not header.startswith(prefix):
            raise APIError(HTTPStatus.UNAUTHORIZED, "unauthorized",
                           "authorization must use Bearer token")
        token = header[len(prefix):]
        if self.approval_token and hmac.compare_digest(token, self.approval_token):
            return "approve"
        if self.api_token and hmac.compare_digest(token, self.api_token):
            return "submit"
        raise APIError(HTTPStatus.FORBIDDEN, "forbidden", "invalid token")


TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.BUDGET_EXCEEDED,
    TaskStatus.CANCELLED,
}


class RuleBasedDemoProvider(LLMProvider):
    """A deterministic provider for local demos without an API key."""

    def complete(self, *, model: str, system: str, messages: list[Message],
                 tools: list[dict[str, Any]], max_tokens: int, timeout_s: float,
                 caching: bool = False, on_text: OnText | None = None,
                 thinking: str | None = None, effort: str | None = None,
                 output_schema: dict[str, Any] | None = None) -> LLMResponse:
        del model, system, tools, max_tokens, timeout_s, caching, thinking, effort
        if messages and messages[-1].tool_results:
            text = self._summarize_tool_results(messages[-1])
            if output_schema:
                text = json.dumps({"result": text}, ensure_ascii=False)
            if on_text:
                on_text(text)
            return LLMResponse(Message(role=Role.ASSISTANT, content=text),
                               Usage(input_tokens=120, output_tokens=40),
                               "end_turn")

        goal = self._latest_user_text(messages)
        call = self._choose_tool(goal)
        if call is None:
            text = (
                "Demo provider is ready. Try: 'query order A1001', "
                "'create ticket for A1001', 'message customer for A1001', "
                "or 'refund A1001 10'."
            )
            if on_text:
                on_text(text)
            return LLMResponse(Message(role=Role.ASSISTANT, content=text),
                               Usage(input_tokens=80, output_tokens=30),
                               "end_turn")
        return LLMResponse(Message(role=Role.ASSISTANT, tool_calls=[call]),
                           Usage(input_tokens=100, output_tokens=20),
                           "tool_use")

    @staticmethod
    def _latest_user_text(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _extract_order_id(text: str) -> str:
        match = re.search(r"\b[A-Z]\d{4,}\b|\b[A-Z]{2,}-?\d{3,}\b", text.upper())
        return match.group(0).replace("-", "") if match else "A1001"

    @staticmethod
    def _extract_amount(text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else 10.0

    def _choose_tool(self, goal: str) -> ToolCall | None:
        text = goal.lower()
        order_id = self._extract_order_id(goal)
        call_id = "demo_" + uuid.uuid4().hex[:10]
        if any(k in text for k in ("refund", "tuikuan")) or "退款" in goal:
            return ToolCall(call_id, "refund_order", {
                "order_id": order_id,
                "amount": self._extract_amount(goal),
                "reason": "requested from demo task",
            })
        if any(k in text for k in ("message", "notify", "sms")) or "通知" in goal or "短信" in goal:
            return ToolCall(call_id, "send_customer_message", {
                "order_id": order_id,
                "channel": "sms",
                "content": f"Your order {order_id} is being handled.",
            })
        if any(k in text for k in ("ticket", "case", "complaint")) or "工单" in goal:
            return ToolCall(call_id, "create_ticket", {
                "order_id": order_id,
                "issue": "Demo support request",
                "priority": "normal",
            })
        if any(k in text for k in ("search", "list")):
            return ToolCall(call_id, "search_orders", {"keyword": ""})
        if any(k in text for k in ("order", "query", "查")) or "订单" in goal:
            return ToolCall(call_id, "query_order", {"order_id": order_id})
        return None

    @staticmethod
    def _summarize_tool_results(message: Message) -> str:
        chunks = []
        for result in message.tool_results:
            label = "error" if result.is_error else "ok"
            chunks.append(f"{result.tool_call_id} [{label}]: {result.content}")
        return "Tool results:\n" + "\n".join(chunks)


class TaskRunner:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()

    def submit(self, goal: str,
               output_schema: dict[str, Any] | None = None) -> TaskState:
        state = TaskState(goal=goal, status=TaskStatus.RUNNING,
                          output_schema=output_schema)
        state.messages.append(Message(role=Role.USER, content=goal))
        self.agent.store.save(state)
        self._spawn(state.task_id, self.agent.resume, state.task_id)
        return self.agent.store.load(state.task_id) or state

    def approve(self, task_id: str, approvals: dict[str, bool],
                approver: str = "", reason: str = "") -> TaskState:
        state = self.agent.store.load(task_id)
        if state is None:
            raise KeyError(f"task {task_id} not found")
        if state.status != TaskStatus.AWAITING_APPROVAL:
            raise APIError(HTTPStatus.CONFLICT, "task_not_awaiting_approval",
                           f"task is {state.status.value}, not awaiting approval")
        pending = set(state.pending_approval_ids)
        provided = set(approvals)
        missing = sorted(pending - provided)
        extra = sorted(provided - pending)
        if missing:
            raise APIError(HTTPStatus.BAD_REQUEST, "missing_approval",
                           f"missing approvals for: {missing}")
        if extra:
            raise APIError(HTTPStatus.BAD_REQUEST, "unknown_approval",
                           f"approval ids are not pending: {extra}")
        for call_id, approved in approvals.items():
            self.agent.store.record_approval(task_id, call_id, approved,
                                             approver=approver, reason=reason)
        state.status = TaskStatus.RUNNING
        self.agent.store.save(state)
        self._spawn(task_id, self.agent.resume, task_id, approvals=approvals)
        return state

    def cancel(self, task_id: str) -> bool:
        state = self.agent.store.load(task_id)
        if state is None:
            return False
        if state.status in TERMINAL_STATUSES:
            raise APIError(HTTPStatus.CONFLICT, "task_terminal",
                           f"task is already {state.status.value}")
        return self.agent.cancel(task_id)

    def _spawn(self, task_id: str, fn, *args, **kwargs) -> None:
        thread = threading.Thread(
            target=self._run_guarded,
            args=(task_id, fn, args, kwargs),
            daemon=True,
        )
        with self._lock:
            self._threads.add(thread)
        thread.start()

    def _run_guarded(self, task_id: str, fn, args, kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # Last-resort boundary for web worker threads.
            state = self.agent.store.load(task_id)
            if state is not None:
                state.status = TaskStatus.FAILED
                state.error = f"{exc!r}\n{traceback.format_exc(limit=8)}"
                self.agent.store.save(state)
        finally:
            current = threading.current_thread()
            with self._lock:
                self._threads.discard(current)


def state_to_dict(state: TaskState,
                  approval_audit: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    last_answer = ""
    for msg in reversed(state.messages):
        if msg.role == Role.ASSISTANT and msg.content:
            last_answer = msg.content
            break
    pending_calls = []
    if state.pending_approval_ids and state.messages:
        pending = set(state.pending_approval_ids)
        for call in state.messages[-1].tool_calls:
            if call.id in pending:
                pending_calls.append(tool_call_to_dict(call))
    return {
        "task_id": state.task_id,
        "goal": state.goal,
        "status": state.status.value,
        "iterations": state.iterations,
        "usage": {
            "input_tokens": state.usage.input_tokens,
            "output_tokens": state.usage.output_tokens,
            "cache_write_tokens": state.usage.cache_write_tokens,
            "cache_read_tokens": state.usage.cache_read_tokens,
        },
        "cost_usd": state.cost_usd,
        "error": state.error,
        "pending_approval_ids": state.pending_approval_ids,
        "pending_calls": pending_calls,
        "answer": last_answer,
        "messages": [message_to_dict(m) for m in state.messages],
        "approval_audit": approval_audit or [],
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [tool_call_to_dict(c) for c in message.tool_calls],
        "tool_results": [
            {
                "tool_call_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in message.tool_results
        ],
    }


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _metric_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if not re.match(r"[a-zA-Z_:]", safe):
        safe = "_" + safe
    return "prod_agent_" + safe


def metrics_to_prometheus(snapshot: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, value in sorted(snapshot.get("counters", {}).items()):
        prom = _metric_name(name)
        lines.append(f"# TYPE {prom} counter")
        lines.append(f"{prom} {float(value)}")
    for name, hist in sorted(snapshot.get("histograms", {}).items()):
        prom = _metric_name(name)
        lines.append(f"# TYPE {prom}_count gauge")
        lines.append(f"{prom}_count {int(hist.get('count', 0))}")
        lines.append(f"# TYPE {prom}_avg gauge")
        lines.append(f"{prom}_avg {float(hist.get('avg', 0.0))}")
        lines.append(f"# TYPE {prom}_min gauge")
        lines.append(f"{prom}_min {float(hist.get('min', 0.0))}")
        lines.append(f"# TYPE {prom}_max gauge")
        lines.append(f"{prom}_max {float(hist.get('max', 0.0))}")
    return "\n".join(lines) + ("\n" if lines else "")


def make_handler(runner: TaskRunner, auth: AuthConfig | None = None):
    auth = auth or AuthConfig()

    class AgentRequestHandler(BaseHTTPRequestHandler):
        server_version = "prod-agent-http/0.1"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def do_OPTIONS(self) -> None:
            self._send_json({"ok": True})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
            try:
                self._authorize(parts)
                if not parts:
                    self._send_html(INDEX_HTML)
                elif parts == ["api", "health"]:
                    self._send_json({"ok": True, "time": time.time()})
                elif parts == ["api", "tools"]:
                    self._send_json({"tools": runner.agent.tools.describe()})
                elif parts == ["api", "metrics"]:
                    query = parse_qs(parsed.query)
                    if query.get("format", [""])[0] == "prom":
                        self._send_text(metrics_to_prometheus(
                            runner.agent.metrics.snapshot()))
                    else:
                        self._send_json(runner.agent.metrics.snapshot())
                elif parts == ["api", "tasks"]:
                    query = parse_qs(parsed.query)
                    limit = int(query.get("limit", ["50"])[0])
                    states = runner.agent.store.list_tasks(limit=limit)
                    self._send_json({"tasks": [state_to_dict(s) for s in states]})
                elif len(parts) == 3 and parts[:2] == ["api", "tasks"]:
                    state = runner.agent.store.load(parts[2])
                    if state is None:
                        self._send_error(HTTPStatus.NOT_FOUND, "not_found",
                                         "task not found")
                    else:
                        self._send_json(state_to_dict(
                            state, runner.agent.store.list_approvals(state.task_id)))
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not_found", "not found")
            except APIError as exc:
                self._send_error(exc.status, exc.code, exc.message)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                                 "internal_error", repr(exc))

        def do_POST(self) -> None:
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")
                     if p]
            try:
                role = self._authorize(parts)
                body = self._read_json()
                if parts == ["api", "tasks"]:
                    goal = str(body.get("goal", "")).strip()
                    if not goal:
                        self._send_error(HTTPStatus.BAD_REQUEST, "missing_goal",
                                         "goal is required")
                        return
                    if len(goal) > MAX_GOAL_CHARS:
                        self._send_error(HTTPStatus.BAD_REQUEST, "goal_too_long",
                                         f"goal exceeds {MAX_GOAL_CHARS} chars")
                        return
                    output_schema = body.get("output_schema")
                    if output_schema is not None and not isinstance(output_schema, dict):
                        self._send_error(HTTPStatus.BAD_REQUEST,
                                         "bad_output_schema",
                                         "output_schema must be an object")
                        return
                    state = runner.submit(goal, output_schema=output_schema)
                    self._send_json(state_to_dict(state), HTTPStatus.ACCEPTED)
                elif len(parts) == 4 and parts[:2] == ["api", "tasks"] \
                        and parts[3] == "approvals":
                    approvals = body.get("approvals", body)
                    if not isinstance(approvals, dict):
                        self._send_error(HTTPStatus.BAD_REQUEST,
                                         "bad_approvals",
                                         "approvals must be an object")
                        return
                    bad_values = [k for k, v in approvals.items()
                                  if not isinstance(v, bool)]
                    if bad_values:
                        self._send_error(HTTPStatus.BAD_REQUEST,
                                         "bad_approval_value",
                                         f"approval values must be bool: {bad_values}")
                        return
                    if role != "approve":
                        self._send_error(HTTPStatus.FORBIDDEN, "forbidden",
                                         "approval token required")
                        return
                    approver = str(body.get("approver") or role or "anonymous")
                    reason = str(body.get("reason") or "")
                    clean = {str(k): v for k, v in approvals.items()}
                    state = runner.approve(parts[2], clean, approver=approver,
                                           reason=reason)
                    self._send_json(state_to_dict(state), HTTPStatus.ACCEPTED)
                elif len(parts) == 4 and parts[:2] == ["api", "tasks"] \
                        and parts[3] == "cancel":
                    ok = runner.cancel(parts[2])
                    if not ok:
                        self._send_error(HTTPStatus.NOT_FOUND, "not_found",
                                         "task not found")
                    else:
                        self._send_json({"ok": True})
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not_found", "not found")
            except APIError as exc:
                self._send_error(exc.status, exc.code, exc.message)
            except KeyError as exc:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found", str(exc))
            except json.JSONDecodeError:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_json", "invalid JSON")
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                                 "internal_error", repr(exc))

        def _authorize(self, parts: list[str]) -> str | None:
            if parts and parts[0] == "api":
                return auth.role_for(self.headers.get("Authorization"))
            return None

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_JSON_BODY_BYTES:
                raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               "body_too_large",
                               f"JSON body exceeds {MAX_JSON_BODY_BYTES} bytes")
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise APIError(HTTPStatus.BAD_REQUEST, "bad_request",
                               "JSON body must be an object")
            return data

        def _send_json(self, payload: dict[str, Any],
                       status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers",
                             "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_text(self, text: str,
                       status: HTTPStatus = HTTPStatus.OK,
                       content_type: str = "text/plain; version=0.0.4; charset=utf-8"
                       ) -> None:
            raw = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
            self._send_json({"error": {"code": code, "message": message}}, status)

    return AgentRequestHandler


def build_agent(provider_name: str, *, model: str | None = None,
                base_url: str | None = None,
                checkpoint_db: str | None = None,
                business_db: str = "demo_business.db",
                demo_tools: bool = True) -> Agent:
    tools = build_demo_tools(business_db) if demo_tools else ToolRegistry()
    if provider_name == "demo":
        provider: LLMProvider = RuleBasedDemoProvider()
        model = model or "demo-rule-based"
    elif provider_name == "anthropic":
        provider = AnthropicProvider(AgentConfig.api_key())
    elif provider_name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        provider = OpenAICompatProvider(key, base_url=base_url)
    else:
        raise ValueError(f"unknown provider: {provider_name}")
    base_config = AgentConfig.from_env()
    config = dataclasses.replace(
        base_config,
        model=model or base_config.model,
        checkpoint_db=checkpoint_db or base_config.checkpoint_db,
        approval_mode="pause",
    )
    return Agent(provider, tools, config)


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 8000,
          auth: AuthConfig | None = None) -> None:
    runner = TaskRunner(agent)
    httpd = ThreadingHTTPServer((host, port), make_handler(runner, auth=auth))
    print(f"prod-agent server listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--provider", choices=["demo", "anthropic", "openai"],
                        default=os.environ.get("PROD_AGENT_PROVIDER", "demo"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--db")
    parser.add_argument("--business-db", default=os.environ.get(
        "PROD_AGENT_BUSINESS_DB", "demo_business.db"))
    parser.add_argument("--no-demo-tools", action="store_true")
    parser.add_argument("--api-token", default=os.environ.get("PROD_AGENT_API_TOKEN", ""),
                        help="Bearer token for regular API access")
    parser.add_argument("--approval-token",
                        default=os.environ.get("PROD_AGENT_APPROVAL_TOKEN", ""),
                        help="Bearer token allowed to approve EXECUTE calls")
    parser.add_argument("--purge-days", type=float,
                        default=float(os.environ.get("PROD_AGENT_PURGE_DAYS", "0")),
                        help="Purge terminal checkpoints older than N days at startup")
    args = parser.parse_args(argv)

    agent = build_agent(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        checkpoint_db=args.db,
        business_db=args.business_db,
        demo_tools=not args.no_demo_tools,
    )
    if args.purge_days > 0:
        purged = agent.store.purge_terminal(args.purge_days)
        if purged:
            print(f"purged {purged} old terminal task(s)")
    serve(agent, args.host, args.port,
          AuthConfig(args.api_token, args.approval_token or args.api_token))
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>prod-agent</title>
  <style>
    :root { color-scheme: light; --bg:#f7f8fb; --panel:#fff; --line:#d9deea;
      --text:#182033; --muted:#667085; --accent:#2563eb; --bad:#b42318;
      --wait:#b54708; --ok:#067647; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
      background:var(--bg); color:var(--text); }
    header { padding:18px 24px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0; font-size:20px; }
    main { display:grid; grid-template-columns:360px 1fr; gap:16px; padding:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    .panel { padding:14px; }
    textarea { width:100%; min-height:110px; resize:vertical; padding:10px;
      border:1px solid var(--line); border-radius:6px; font:inherit; }
    button { border:0; background:var(--accent); color:white; border-radius:6px;
      padding:8px 12px; cursor:pointer; font-weight:600; }
    button.secondary { background:#eef2ff; color:#1d4ed8; }
    button.danger { background:var(--bad); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .muted { color:var(--muted); }
    .task { padding:10px 12px; border-top:1px solid var(--line); cursor:pointer; }
    .task:hover { background:#f2f5fb; }
    .task.active { background:#eaf1ff; }
    .status { display:inline-block; padding:2px 7px; border-radius:999px;
      background:#eef2f6; color:#344054; font-size:12px; }
    .status.completed { background:#dcfae6; color:var(--ok); }
    .status.failed,.status.budget_exceeded,.status.cancelled {
      background:#fee4e2; color:var(--bad); }
    .status.awaiting_approval { background:#fef0c7; color:var(--wait); }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc;
      border:1px solid var(--line); border-radius:6px; padding:10px; max-height:360px;
      overflow:auto; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .tool { border-top:1px solid var(--line); padding:10px 0; }
    @media (max-width: 850px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div class="row" style="justify-content:space-between">
      <div><h1>prod-agent</h1><div class="muted">Tasks, tools, approvals</div></div>
      <label class="muted">API token
        <input id="token" type="password" placeholder="optional Bearer token"
          style="margin-left:8px;padding:7px;border:1px solid var(--line);border-radius:6px">
      </label>
    </div>
  </header>
  <main>
    <section>
      <div class="panel">
        <h2>New task</h2>
        <textarea id="goal">query order A1001</textarea>
        <div class="row" style="margin-top:10px">
          <button id="submit">Run</button>
          <button class="secondary" id="refresh">Refresh</button>
        </div>
      </div>
      <div id="tasks"></div>
    </section>
    <section class="panel">
      <div class="row" style="justify-content:space-between">
        <h2>Task detail</h2>
        <button class="danger" id="cancel" disabled>Cancel</button>
      </div>
      <div id="detail" class="muted">Select or run a task.</div>
      <h2>Tools</h2>
      <div id="tools" class="muted">Loading...</div>
    </section>
  </main>
  <script>
    let selected = null;
    const tokenInput = document.querySelector('#token');
    tokenInput.value = localStorage.getItem('prodAgentToken') || '';
    tokenInput.oninput = () => localStorage.setItem('prodAgentToken', tokenInput.value);

    async function api(path, options) {
      const token = tokenInput.value.trim();
      const opts = options || {};
      opts.headers = Object.assign({}, opts.headers || {});
      if (token) opts.headers.Authorization = 'Bearer ' + token;
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok) {
        const err = data.error && data.error.message ? data.error.message : data.error;
        throw new Error(err || res.statusText);
      }
      return data;
    }

    function statusBadge(s) {
      return `<span class="status ${s}">${s}</span>`;
    }

    async function loadTasks() {
      const data = await api('/api/tasks?limit=50');
      const box = document.querySelector('#tasks');
      box.innerHTML = data.tasks.map(t => `
        <div class="task ${selected === t.task_id ? 'active' : ''}" data-id="${t.task_id}">
          <div class="row" style="justify-content:space-between">
            <strong>${t.task_id}</strong>${statusBadge(t.status)}
          </div>
          <div class="muted">${escapeHtml(t.goal).slice(0, 120)}</div>
        </div>`).join('');
      box.querySelectorAll('.task').forEach(el => {
        el.onclick = () => { selected = el.dataset.id; loadDetail(); loadTasks(); };
      });
      if (!selected && data.tasks[0]) selected = data.tasks[0].task_id;
      if (selected) loadDetail();
    }

    async function loadDetail() {
      if (!selected) return;
      const t = await api('/api/tasks/' + selected);
      document.querySelector('#cancel').disabled =
        !['running', 'awaiting_approval'].includes(t.status);
      const approvals = t.pending_calls.length ? `
        <h3>Approvals</h3>
        ${t.pending_calls.map(c => `
          <div class="tool">
            <strong>${c.name}</strong> <code>${c.id}</code>
            <pre>${escapeHtml(JSON.stringify(c.arguments, null, 2))}</pre>
            <button onclick="approve('${c.id}', true)">Approve</button>
            <button class="danger" onclick="approve('${c.id}', false)">Deny</button>
          </div>`).join('')}` : '';
      document.querySelector('#detail').innerHTML = `
        <div class="grid">
          <div><strong>ID</strong><br>${t.task_id}</div>
          <div><strong>Status</strong><br>${statusBadge(t.status)}</div>
          <div><strong>Cost</strong><br>$${t.cost_usd}</div>
          <div><strong>Iterations</strong><br>${t.iterations}</div>
        </div>
        <h3>Goal</h3><pre>${escapeHtml(t.goal)}</pre>
        ${t.error ? `<h3>Error</h3><pre>${escapeHtml(t.error)}</pre>` : ''}
        ${approvals}
        <h3>Answer</h3><pre>${escapeHtml(t.answer || '')}</pre>
        <h3>Messages</h3><pre>${escapeHtml(JSON.stringify(t.messages, null, 2))}</pre>`;
    }

    async function loadTools() {
      const data = await api('/api/tools');
      document.querySelector('#tools').innerHTML = data.tools.map(t => `
        <div class="tool">
          <strong>${t.name}</strong> ${statusBadge(t.permission)}
          <div class="muted">${escapeHtml(t.description)}</div>
        </div>`).join('');
    }

    async function approve(id, value) {
      await api('/api/tasks/' + selected + '/approvals', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({approvals:{[id]: value}, approver:'web-ui'})
      });
      setTimeout(loadTasks, 500);
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({
        '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
      }[c]));
    }

    document.querySelector('#submit').onclick = async () => {
      const goal = document.querySelector('#goal').value.trim();
      if (!goal) return;
      const t = await api('/api/tasks', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({goal})
      });
      selected = t.task_id;
      setTimeout(loadTasks, 300);
    };
    document.querySelector('#refresh').onclick = loadTasks;
    document.querySelector('#cancel').onclick = async () => {
      if (!selected) return;
      await api('/api/tasks/' + selected + '/cancel', {method:'POST'});
      loadTasks();
    };

    loadTools();
    loadTasks();
    setInterval(loadTasks, 2500);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

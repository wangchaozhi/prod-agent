"""测试 — 用 MockProvider 验证主循环、重试、熔断、预算、权限、审批、租约、压缩、检查点。
运行: python -m pytest tests/ -v  (或 python tests/test_agent.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import (Agent, AgentConfig, BudgetConfig, ContextConfig,
                   ContextCompactor, LeaseConflict, MockProvider, Permission,
                   RateLimitConfig, RetryConfig, TaskState, TaskStatus,
                   ToolRegistry, ToolSpec, Worker, build_demo_tools,
                   estimate_tokens)
from agent.budget import BudgetTracker
from agent.config import CircuitBreakerConfig
from agent.context import COMPACTED_PLACEHOLDER, SUMMARY_PREFIX
from agent.observability import Metrics
from agent.types import (LLMResponse, Message, Role, ToolCall, ToolResult,
                         Usage)


def _db_path() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="prod_agent_test_"), "ckpt.db")


def _cfg(approval_mode: str = "sync", breaker_threshold: int = 10,
         **budget_kw) -> AgentConfig:
    return AgentConfig(
        retry=RetryConfig(max_attempts=3, base_delay_s=0.01, max_delay_s=0.02,
                          timeout_s=5),
        breaker=CircuitBreakerConfig(failure_threshold=breaker_threshold,
                                     recovery_time_s=60),
        budget=BudgetConfig(**({"max_iterations": 10, "max_total_tokens": 100000,
                                "max_cost_usd": 5.0} | budget_kw)),
        checkpoint_db=_db_path(),
        approval_mode=approval_mode,
    )


def _resp(text="", tool_calls=None, stop="end_turn", tokens=(100, 50)):
    msg = Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls or [])
    return LLMResponse(msg, Usage(*tokens), stop)


def _tools() -> ToolRegistry:
    reg = ToolRegistry(require_approval_for_execute=True, approval_hook=None)

    @reg.tool(name="echo", description="echo",
              input_schema={"type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"]})
    def echo(text: str) -> str:
        return f"echo:{text}"

    @reg.tool(name="deploy", description="dangerous",
              input_schema={"type": "object", "properties": {}},
              permission=Permission.EXECUTE)
    def deploy() -> str:
        return "deployed"

    return reg


def test_happy_path_with_tool():
    provider = MockProvider([
        _resp(tool_calls=[ToolCall("t1", "echo", {"text": "hi"})], stop="tool_use"),
        _resp(text="done", stop="end_turn"),
    ])
    state = Agent(provider, _tools(), _cfg()).run("test goal")
    assert state.status == TaskStatus.COMPLETED
    assert state.messages[-1].content == "done"
    # 工具结果已回填
    assert any(m.tool_results and "echo:hi" in m.tool_results[0].content
               for m in state.messages)
    print("PASS test_happy_path_with_tool")


def test_retry_on_transient_error():
    provider = MockProvider([_resp(text="ok")], fail_first_n=2)  # 前2次失败
    state = Agent(provider, _tools(), _cfg()).run("test")
    assert state.status == TaskStatus.COMPLETED
    assert provider.calls == 3  # 2 失败 + 1 成功
    print("PASS test_retry_on_transient_error")


def test_circuit_breaker_opens_and_task_persisted():
    provider = MockProvider([], fail_forever=True)
    cfg = _cfg(breaker_threshold=2)
    agent = Agent(provider, _tools(), cfg)
    state = agent.run("test")
    assert state.status == TaskStatus.FAILED
    assert "熔断" in (state.error or "")
    assert provider.calls == 2  # 2 次失败后熔断,第 3 次尝试被拦截
    # CircuitOpen 不再击穿主循环,失败状态已落盘
    saved = agent.store.load(state.task_id)
    assert saved is not None and saved.status == TaskStatus.FAILED
    print("PASS test_circuit_breaker_opens_and_task_persisted")


def test_max_tokens_truncation_marks_failed():
    provider = MockProvider([_resp(text="partial...", stop="max_tokens")])
    state = Agent(provider, _tools(), _cfg()).run("test")
    assert state.status == TaskStatus.FAILED
    assert "max_tokens" in (state.error or "")
    print("PASS test_max_tokens_truncation_marks_failed")


def test_budget_iteration_limit():
    # 模型永远要求调工具 → 应被迭代上限拦下
    loop = [_resp(tool_calls=[ToolCall(f"t{i}", "echo", {"text": "x"})],
                  stop="tool_use") for i in range(10)]
    state = Agent(MockProvider(loop), _tools(),
                  _cfg(max_iterations=3)).run("test")
    assert state.status == TaskStatus.BUDGET_EXCEEDED
    assert "迭代次数" in (state.error or "")
    print("PASS test_budget_iteration_limit")


def test_budget_cost_limit():
    loop = [_resp(tool_calls=[ToolCall(f"t{i}", "echo", {"text": "x"})],
                  stop="tool_use", tokens=(200000, 100000)) for i in range(5)]
    state = Agent(MockProvider(loop), _tools(),
                  _cfg(max_cost_usd=1.0, max_total_tokens=10_000_000)).run("test")
    assert state.status == TaskStatus.BUDGET_EXCEEDED
    assert "成本" in (state.error or "")
    print("PASS test_budget_cost_limit")


def test_cache_aware_cost_accounting():
    bt = BudgetTracker(BudgetConfig(
        price_per_mtok_input=5.0, price_per_mtok_output=25.0,
        price_per_mtok_cache_write=6.25, price_per_mtok_cache_read=0.5))
    bt.record(Usage(input_tokens=1_000_000, output_tokens=2_000_000,
                    cache_write_tokens=1_000_000, cache_read_tokens=1_000_000))
    assert abs(bt.cost_usd - (5.0 + 50.0 + 6.25 + 0.5)) < 1e-9
    assert bt.total_tokens == 5_000_000
    print("PASS test_cache_aware_cost_accounting")


def test_execute_permission_denied_without_approval():
    provider = MockProvider([
        _resp(tool_calls=[ToolCall("t1", "deploy", {})], stop="tool_use"),
        _resp(text="ok", stop="end_turn"),
    ])
    state = Agent(provider, _tools(), _cfg()).run("test")  # sync 模式 + 无 hook → 拒绝
    denied = [m for m in state.messages
              if m.tool_results and m.tool_results[0].is_error]
    assert denied and "未获人工批准" in denied[0].tool_results[0].content
    print("PASS test_execute_permission_denied")


def test_pause_approval_then_approve_resume():
    cfg = _cfg(approval_mode="pause")
    a1 = Agent(MockProvider([
        _resp(tool_calls=[ToolCall("t1", "deploy", {})], stop="tool_use"),
    ]), _tools(), cfg)
    s1 = a1.run("deploy something")
    assert s1.status == TaskStatus.AWAITING_APPROVAL
    assert s1.pending_approval_ids == ["t1"]
    # 挂起状态已落盘,工具尚未执行
    assert not any(m.tool_results for m in s1.messages)

    # 缺少审批决定 → 报错且状态不被破坏
    a2 = Agent(MockProvider([_resp(text="done")]), _tools(), cfg)
    try:
        a2.resume(s1.task_id)
        raise AssertionError("应当因缺少 approvals 而报错")
    except ValueError as e:
        assert "approvals" in str(e)
    assert a2.store.load(s1.task_id).status == TaskStatus.AWAITING_APPROVAL

    # 批准后续跑
    s2 = a2.resume(s1.task_id, approvals={"t1": True})
    assert s2.status == TaskStatus.COMPLETED
    assert any(m.tool_results and m.tool_results[0].content == "deployed"
               for m in s2.messages)
    print("PASS test_pause_approval_then_approve_resume")


def test_pause_approval_then_deny_resume():
    cfg = _cfg(approval_mode="pause")
    a1 = Agent(MockProvider([
        _resp(tool_calls=[ToolCall("t1", "deploy", {})], stop="tool_use"),
    ]), _tools(), cfg)
    s1 = a1.run("deploy something")
    assert s1.status == TaskStatus.AWAITING_APPROVAL

    a2 = Agent(MockProvider([_resp(text="ok, not deploying")]), _tools(), cfg)
    s2 = a2.resume(s1.task_id, approvals={"t1": False})
    assert s2.status == TaskStatus.COMPLETED
    denied = [m for m in s2.messages
              if m.tool_results and m.tool_results[0].is_error]
    assert denied and "拒绝" in denied[0].tool_results[0].content
    print("PASS test_pause_approval_then_deny_resume")


def test_lease_conflict_blocks_concurrent_resume():
    cfg = _cfg()
    agent = Agent(MockProvider([_resp(text="ok")]), _tools(), cfg)
    s = agent.run("test")
    saved = agent.store.load(s.task_id)
    saved.status = TaskStatus.RUNNING
    agent.store.save(saved)
    # 另一个执行者持有租约
    assert agent.store.acquire_lease(s.task_id, "other-owner", 300)
    try:
        agent.resume(s.task_id)
        raise AssertionError("应当因租约冲突而报错")
    except LeaseConflict:
        pass
    # 释放后可正常恢复
    agent.store.release_lease(s.task_id, "other-owner")
    agent2 = Agent(MockProvider([_resp(text="resumed")]), _tools(), cfg)
    s2 = agent2.resume(s.task_id)
    assert s2.status == TaskStatus.COMPLETED
    print("PASS test_lease_conflict_blocks_concurrent_resume")


def test_unknown_tool_and_bad_args():
    reg = _tools()
    r1 = reg.execute(ToolCall("a", "nonexistent", {}))
    assert r1.is_error and "不存在" in r1.content
    r2 = reg.execute(ToolCall("b", "echo", {}))  # 缺必填参数
    assert r2.is_error and "缺少必填参数" in r2.content
    r3 = reg.execute(ToolCall("c", "echo", {"text": 123}))  # 类型错误
    assert r3.is_error and "类型错误" in r3.content
    print("PASS test_unknown_tool_and_bad_args")


def test_tool_result_truncation():
    reg = ToolRegistry()
    reg.register(ToolSpec("big", "returns huge output",
                          {"type": "object", "properties": {}},
                          lambda: "x" * 100_000, max_result_chars=50))
    r = reg.execute(ToolCall("a", "big", {}))
    assert not r.is_error and "截断" in r.content
    assert len(r.content) < 200
    print("PASS test_tool_result_truncation")


def test_demo_business_tools_idempotency_and_approval():
    reg = build_demo_tools(_db_path())
    q = reg.execute(ToolCall("q1", "query_order", {"order_id": "A1001"}))
    assert not q.is_error and "A1001" in q.content

    call = ToolCall("ticket-call-1", "create_ticket", {
        "order_id": "A1001",
        "issue": "package damaged",
        "priority": "high",
    })
    first = reg.execute(call)
    second = reg.execute(call)
    assert not first.is_error and not second.is_error
    assert json.loads(second.content)["idempotent"] is True

    denied = reg.execute(ToolCall("m1", "send_customer_message", {
        "order_id": "A1001",
        "channel": "sms",
        "content": "hello",
    }), approved=False)
    assert denied.is_error and "拒绝" in denied.content

    sent = reg.execute(ToolCall("m1", "send_customer_message", {
        "order_id": "A1001",
        "channel": "sms",
        "content": "hello",
    }), approved=True)
    assert not sent.is_error and json.loads(sent.content)["ok"] is True
    print("PASS test_demo_business_tools_idempotency_and_approval")


def test_rule_based_demo_provider_approval_flow():
    from agent.server import RuleBasedDemoProvider

    cfg = _cfg(approval_mode="pause")
    agent = Agent(RuleBasedDemoProvider(), build_demo_tools(_db_path()), cfg)
    s1 = agent.run("message customer for order A1001")
    assert s1.status == TaskStatus.AWAITING_APPROVAL
    assert s1.pending_approval_ids

    s2 = agent.resume(s1.task_id, approvals={s1.pending_approval_ids[0]: True})
    assert s2.status == TaskStatus.COMPLETED
    assert "Tool results" in s2.messages[-1].content
    print("PASS test_rule_based_demo_provider_approval_flow")


def test_checkpoint_list_tasks_purge_and_tool_describe():
    cfg = _cfg()
    agent = Agent(MockProvider([]), _tools(), cfg)

    old_done = TaskState(goal="old", status=TaskStatus.COMPLETED)
    old_done.updated_at = time.time() - 10 * 86400
    running = TaskState(goal="running", status=TaskStatus.RUNNING)
    recent_done = TaskState(goal="recent", status=TaskStatus.COMPLETED)
    agent.store.save(old_done)
    agent.store.save(running)
    agent.store.save(recent_done)
    # save() refreshes updated_at, so age one row directly for retention coverage.
    with agent.store._lock, agent.store._conn:
        agent.store._conn.execute(
            "UPDATE checkpoints SET updated_at=? WHERE task_id=?",
            (time.time() - 10 * 86400, old_done.task_id),
        )

    tasks = agent.store.list_tasks(limit=2)
    assert len(tasks) == 2
    assert tasks[0].updated_at >= tasks[1].updated_at
    assert all(t.status == TaskStatus.COMPLETED
               for t in agent.store.list_tasks(statuses=(TaskStatus.COMPLETED,)))

    assert agent.store.purge_terminal(older_than_days=1) == 1
    assert agent.store.load(old_done.task_id) is None
    assert agent.store.load(running.task_id) is not None

    described = {t["name"]: t for t in _tools().describe()}
    assert described["deploy"]["permission"] == Permission.EXECUTE.value
    assert described["echo"]["max_result_chars"] == 20_000
    assert described["echo"]["pass_call_id"] is False
    print("PASS test_checkpoint_list_tasks_purge_and_tool_describe")


def test_http_api_auth_approval_audit_and_errors():
    from agent.server import AuthConfig, TaskRunner, build_agent, make_handler

    agent = build_agent(
        "demo",
        checkpoint_db=_db_path(),
        business_db=_db_path(),
    )
    runner = TaskRunner(agent)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(runner, auth=AuthConfig("submit-token", "approve-token")),
    )
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def req(path, body=None, token="submit-token", raw=None):
        data = raw
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def wait_status(task_id, statuses):
        for _ in range(50):
            _, state = req("/api/tasks/" + task_id)
            if state["status"] in statuses:
                return state
            time.sleep(0.05)
        raise AssertionError(f"task {task_id} did not reach {statuses}")

    try:
        status, err = req("/api/health", token=None)
        assert status == 401 and err["error"]["code"] == "unauthorized"
        status, err = req("/api/health", token="bad")
        assert status == 403 and err["error"]["code"] == "forbidden"

        status, err = req("/api/tasks", {"goal": ""})
        assert status == 400 and err["error"]["code"] == "missing_goal"
        status, err = req("/api/tasks", raw=b"{", token="submit-token")
        assert status == 400 and err["error"]["code"] == "invalid_json"
        status, err = req("/api/tasks/nope")
        assert status == 404 and err["error"]["code"] == "not_found"

        status, created = req("/api/tasks", {"goal": "query order A1001"})
        assert status == 202
        done = wait_status(created["task_id"], {"completed"})
        assert "A1001" in done["answer"]

        assert req("/api/tasks")[1]["tasks"]
        assert req("/api/tools")[1]["tools"]
        assert "counters" in req("/api/metrics")[1]
        status, err = req("/api/tasks/nope/cancel", {})
        assert status == 404 and err["error"]["code"] == "not_found"
        status, err = req(f"/api/tasks/{done['task_id']}/cancel", {})
        assert status == 409 and err["error"]["code"] == "task_terminal"
    except AssertionError:
        raise
    finally:
        httpd.shutdown()
        httpd.server_close()

    # Text endpoint check is separate because the helper expects JSON.
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(TaskRunner(agent), auth=AuthConfig("submit-token", "approve-token")),
    )
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/metrics?format=prom",
            headers={"Authorization": "Bearer submit-token"},
        )
        with urllib.request.urlopen(request, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        assert "prod_agent_" in text
    finally:
        httpd.shutdown()
        httpd.server_close()

    # Approval path with audit.
    agent = build_agent("demo", checkpoint_db=_db_path(), business_db=_db_path())
    runner = TaskRunner(agent)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(runner, auth=AuthConfig("submit-token", "approve-token")),
    )
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        status, created = req("/api/tasks", {"goal": "message customer for order A1001"})
        assert status == 202
        pending = wait_status(created["task_id"], {"awaiting_approval"})
        call_id = pending["pending_approval_ids"][0]
        status, err = req(
            f"/api/tasks/{created['task_id']}/approvals",
            {"approvals": {call_id: True}},
            token="submit-token",
        )
        assert status == 403
        status, err = req(
            f"/api/tasks/{created['task_id']}/approvals",
            {"approvals": {"wrong": True}},
            token="approve-token",
        )
        assert status == 400
        status, _ = req(
            f"/api/tasks/{created['task_id']}/approvals",
            {"approvals": {call_id: True}, "approver": "alice", "reason": "ok"},
            token="approve-token",
        )
        assert status == 202
        done = wait_status(created["task_id"], {"completed"})
        assert done["approval_audit"][0]["approver"] == "alice"
        status, err = req(
            f"/api/tasks/{created['task_id']}/approvals",
            {"approvals": {call_id: True}},
            token="approve-token",
        )
        assert status == 409
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("PASS test_http_api_auth_approval_audit_and_errors")


def test_task_runner_exception_marks_failed():
    from agent.server import TaskRunner, RuleBasedDemoProvider

    agent = Agent(RuleBasedDemoProvider(), build_demo_tools(_db_path()), _cfg())
    state = TaskState(goal="boom", status=TaskStatus.RUNNING)
    agent.store.save(state)
    runner = TaskRunner(agent)

    def fail():
        raise RuntimeError("background boom")

    runner._spawn(state.task_id, fail)
    for _ in range(50):
        saved = agent.store.load(state.task_id)
        if saved and saved.status == TaskStatus.FAILED:
            break
        time.sleep(0.02)
    saved = agent.store.load(state.task_id)
    assert saved is not None and saved.status == TaskStatus.FAILED
    assert "background boom" in (saved.error or "")
    print("PASS test_task_runner_exception_marks_failed")


def test_context_compaction():
    msgs = [Message(role=Role.USER, content="task goal")]
    for i in range(10):
        msgs.append(Message(role=Role.ASSISTANT, content="thinking " * 50,
                            tool_calls=[ToolCall(f"t{i}", "echo", {"text": "x"})]))
        msgs.append(Message(role=Role.USER,
                            tool_results=[ToolResult(f"t{i}", "R" * 2000)]))
    cc = ContextCompactor(ContextConfig(max_context_tokens=500,
                                        keep_recent_messages=4))
    before = estimate_tokens(msgs)
    assert cc.compact_if_needed(msgs)
    assert estimate_tokens(msgs) < before
    assert msgs[0].content == "task goal"                       # 目标受保护
    assert msgs[2].tool_results[0].content == COMPACTED_PLACEHOLDER  # 旧结果被压缩
    assert msgs[-1].tool_results[0].content == "R" * 2000       # 最近消息未动
    # 未超阈值时不压缩
    cc2 = ContextCompactor(ContextConfig(max_context_tokens=10_000_000))
    assert not cc2.compact_if_needed(msgs)
    print("PASS test_context_compaction")


def test_metrics_bounded_histogram():
    m = Metrics()
    for v in (1.0, 2.0, 3.0):
        m.observe("dur", v)
    snap = m.snapshot()["histograms"]["dur"]
    assert snap == {"count": 3, "avg": 2.0, "min": 1.0, "max": 3.0}
    print("PASS test_metrics_bounded_histogram")


def test_agent_config_from_env():
    keys = {
        "PROD_AGENT_MODEL": "env-model",
        "PROD_AGENT_MAX_TOKENS_PER_CALL": "123",
        "PROD_AGENT_BUDGET_MAX_COST_USD": "0.5",
        "PROD_AGENT_CONTEXT_KEEP_RECENT_MESSAGES": "3",
        "PROD_AGENT_RATE_MAX_CALLS_PER_MINUTE": "60",
        "PROD_AGENT_RATE_MAX_CONCURRENT_CALLS": "2",
        "PROD_AGENT_CHECKPOINT_DB": "env.db",
        "PROD_AGENT_ENABLE_PROMPT_CACHING": "false",
    }
    old = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        cfg = AgentConfig.from_env()
        assert cfg.model == "env-model"
        assert cfg.max_tokens_per_call == 123
        assert cfg.budget.max_cost_usd == 0.5
        assert cfg.context.keep_recent_messages == 3
        assert cfg.rate_limit == RateLimitConfig(60.0, 2)
        assert cfg.checkpoint_db == "env.db"
        assert cfg.enable_prompt_caching is False
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("PASS test_agent_config_from_env")


def test_streaming_retry_does_not_duplicate_failed_attempt_text():
    from agent.llm import LLMProvider, TransientLLMError

    class FlakyStreamingProvider(LLMProvider):
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            on_text = kwargs.get("on_text")
            if self.calls == 1:
                if on_text:
                    on_text("partial")
                raise TransientLLMError("stream broke")
            if on_text:
                on_text("final")
            return _resp(text="final", stop="end_turn")

    chunks = []
    provider = FlakyStreamingProvider()
    state = Agent(provider, _tools(), _cfg()).run("stream", on_text=chunks.append)
    assert state.status == TaskStatus.COMPLETED
    assert chunks == ["final"]
    assert provider.calls == 2
    print("PASS test_streaming_retry_does_not_duplicate_failed_attempt_text")


class _SpyProvider(MockProvider):
    """记录每次 complete 收到的 kwargs,用于验证参数透传。"""

    def __init__(self, script):
        super().__init__(script)
        self.seen: list[dict] = []

    def complete(self, **kw):
        self.seen.append(kw)
        return super().complete(**kw)


def test_refusal_marks_failed():
    provider = MockProvider([_resp(text="", stop="refusal")])
    state = Agent(provider, _tools(), _cfg()).run("test")
    assert state.status == TaskStatus.FAILED
    assert "refusal" in (state.error or "")
    print("PASS test_refusal_marks_failed")


def test_pause_turn_continues():
    provider = MockProvider([
        _resp(text="server tool working...", stop="pause_turn"),
        _resp(text="done", stop="end_turn"),
    ])
    state = Agent(provider, _tools(), _cfg()).run("test")
    assert state.status == TaskStatus.COMPLETED
    assert provider.calls == 2  # pause_turn 后重发历史续跑
    assert state.messages[-1].content == "done"
    print("PASS test_pause_turn_continues")


def test_cancel_task():
    cfg = _cfg()
    agent = Agent(MockProvider([]), _tools(), cfg)  # 空脚本:被调用即报错
    st = TaskState(goal="g", status=TaskStatus.RUNNING)
    st.messages.append(Message(role=Role.USER, content="g"))
    agent.store.save(st)
    assert agent.cancel(st.task_id)
    s = agent.resume(st.task_id)
    assert s.status == TaskStatus.CANCELLED  # LLM 从未被调用
    assert "取消" in (s.error or "")
    print("PASS test_cancel_task")


def test_wall_time_budget():
    provider = MockProvider([_resp(text="ok")])
    state = Agent(provider, _tools(),
                  _cfg(max_wall_time_s=1e-9)).run("test")
    assert state.status == TaskStatus.BUDGET_EXCEEDED
    assert "墙钟" in (state.error or "")
    assert provider.calls == 0
    print("PASS test_wall_time_budget")


def test_parallel_read_tools_keep_order():
    provider = MockProvider([
        _resp(tool_calls=[ToolCall(f"t{i}", "echo", {"text": str(i)})
                          for i in range(4)], stop="tool_use"),
        _resp(text="done"),
    ])
    state = Agent(provider, _tools(), _cfg()).run("test")
    assert state.status == TaskStatus.COMPLETED
    batch = next(m for m in state.messages if m.tool_results)
    assert [r.tool_call_id for r in batch.tool_results] == ["t0", "t1", "t2", "t3"]
    assert [r.content for r in batch.tool_results] == [f"echo:{i}" for i in range(4)]
    print("PASS test_parallel_read_tools_keep_order")


def test_thinking_blocks_roundtrip():
    from agent.llm import _to_anthropic_messages
    tb = {"type": "thinking", "thinking": "let me think", "signature": "sig123"}
    m = Message(role=Role.ASSISTANT, content="answer",
                thinking_blocks=[tb],
                tool_calls=[ToolCall("t1", "echo", {"x": 1})])
    out = _to_anthropic_messages([Message(role=Role.USER, content="q"), m],
                                 caching=True)
    c = out[1]["content"]
    assert c[0] == tb  # thinking 块原样领先回传
    assert c[1]["type"] == "text" and c[2]["type"] == "tool_use"
    assert "cache_control" in c[-1]  # 断点落在 tool_use 上
    # 消息以 thinking 结尾时,缓存断点不落在 thinking 块上
    only_think = Message(role=Role.ASSISTANT, thinking_blocks=[dict(tb)])
    out2 = _to_anthropic_messages([only_think], caching=True)
    assert "cache_control" not in out2[0]["content"][-1]
    # 检查点序列化往返保留 thinking 块
    agent = Agent(MockProvider([]), _tools(), _cfg())
    st = TaskState(goal="g")
    st.messages = [Message(role=Role.USER, content="q"), m]
    agent.store.save(st)
    loaded = agent.store.load(st.task_id)
    assert loaded.messages[1].thinking_blocks == [tb]
    print("PASS test_thinking_blocks_roundtrip")


def test_summary_compaction_keeps_tool_pairing():
    msgs = [Message(role=Role.USER, content="task goal")]
    for i in range(10):
        msgs.append(Message(role=Role.ASSISTANT, content="thinking " * 30,
                            tool_calls=[ToolCall(f"t{i}", "echo", {"text": "x"})]))
        msgs.append(Message(role=Role.USER,
                            tool_results=[ToolResult(f"t{i}", "R" * 1000)]))
    cc = ContextCompactor(
        ContextConfig(max_context_tokens=300, keep_recent_messages=3),
        summarize_fn=lambda old: f"合并了 {len(old)} 条历史")
    assert cc.compact_if_needed(msgs)
    assert msgs[0].content == "task goal"
    assert msgs[1].content.startswith(SUMMARY_PREFIX) and "合并了" in msgs[1].content
    # 折叠边界对齐:保留区不以孤儿 tool_results 开头,配对完整
    assert not msgs[2].tool_results
    for idx, m in enumerate(msgs):
        if m.tool_results:
            prev = msgs[idx - 1]
            assert prev.tool_calls
            assert prev.tool_calls[0].id == m.tool_results[0].tool_call_id
    print("PASS test_summary_compaction_keeps_tool_pairing")


def test_worker_resumes_crashed_task():
    cfg = _cfg()
    agent = Agent(MockProvider([_resp(text="recovered")]), _tools(), cfg)
    # 模拟崩溃遗留:RUNNING 状态躺在库里,无人持有租约
    st = TaskState(goal="g", status=TaskStatus.RUNNING)
    st.messages.append(Message(role=Role.USER, content="g"))
    agent.store.save(st)
    w = Worker(agent, poll_interval_s=0.01)
    assert w.run_once() == 1
    assert agent.store.load(st.task_id).status == TaskStatus.COMPLETED
    assert w.run_once() == 0  # 没有遗留任务了
    print("PASS test_worker_resumes_crashed_task")


def test_structured_output_and_reasoning_passthrough():
    spy = _SpyProvider([_resp(text='{"ok": true}')])
    cfg = AgentConfig(
        thinking="adaptive", effort="high",
        retry=RetryConfig(max_attempts=2, base_delay_s=0.01, max_delay_s=0.02,
                          timeout_s=5),
        budget=BudgetConfig(max_iterations=5),
        checkpoint_db=_db_path(),
    )
    agent = Agent(spy, _tools(), cfg)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    state = agent.run("extract", output_schema=schema)
    assert state.status == TaskStatus.COMPLETED
    kw = spy.seen[0]
    assert kw["output_schema"] == schema
    assert kw["thinking"] == "adaptive" and kw["effort"] == "high"
    assert kw["caching"] is True
    # schema 随检查点持久化,resume 后依然生效
    assert agent.store.load(state.task_id).output_schema == schema
    print("PASS test_structured_output_and_reasoning_passthrough")


def test_openai_compat_conversion():
    from agent.llm_openai import (_map_finish_reason, _parse_arguments,
                                  _to_openai_messages, _to_openai_tools,
                                  _usage_from)
    msgs = [
        Message(role=Role.USER, content="goal"),
        Message(role=Role.ASSISTANT, content="calling",
                tool_calls=[ToolCall("t1", "echo", {"x": 1})]),
        Message(role=Role.USER, tool_results=[
            ToolResult("t1", "ok"),
            ToolResult("t2", "boom", is_error=True),
        ]),
    ]
    out = _to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "goal"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}
    assert out[4]["tool_call_id"] == "t2" and "error" in out[4]["content"]

    oa_tools = _to_openai_tools(
        [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}])
    assert oa_tools[0]["function"]["parameters"] == {"type": "object"}
    assert _to_openai_tools([]) is None

    assert _map_finish_reason("stop") == "end_turn"
    assert _map_finish_reason("tool_calls") == "tool_use"
    assert _map_finish_reason("length") == "max_tokens"
    assert _map_finish_reason(None) == "end_turn"

    assert _parse_arguments('{"a": 1}') == {"a": 1}
    assert _parse_arguments("not json") == {}
    assert _parse_arguments("") == {}

    u = _usage_from(NS(prompt_tokens=100, completion_tokens=50,
                       prompt_tokens_details=NS(cached_tokens=30)))
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (70, 30, 50)
    assert _usage_from(None).input_tokens == 0
    print("PASS test_openai_compat_conversion")


def test_checkpoint_resume():
    cfg = _cfg()
    # 第一段:跑到工具调用后 provider 脚本耗尽 → FAILED,但检查点已落盘
    p1 = MockProvider([
        _resp(tool_calls=[ToolCall("t1", "echo", {"text": "hi"})], stop="tool_use"),
    ])
    a1 = Agent(p1, _tools(), cfg)
    s1 = a1.run("resume test")
    assert s1.status == TaskStatus.FAILED
    # 第二段:用同一 db 恢复,新 provider 给出收尾响应
    saved = a1.store.load(s1.task_id)
    assert saved is not None and saved.iterations >= 1
    saved.status = TaskStatus.RUNNING
    a1.store.save(saved)
    a2 = Agent(MockProvider([_resp(text="resumed done")]), _tools(), cfg)
    s2 = a2.resume(s1.task_id)
    assert s2.status == TaskStatus.COMPLETED
    assert s2.messages[-1].content == "resumed done"
    print("PASS test_checkpoint_resume")


if __name__ == "__main__":
    test_happy_path_with_tool()
    test_retry_on_transient_error()
    test_circuit_breaker_opens_and_task_persisted()
    test_max_tokens_truncation_marks_failed()
    test_budget_iteration_limit()
    test_budget_cost_limit()
    test_cache_aware_cost_accounting()
    test_execute_permission_denied_without_approval()
    test_pause_approval_then_approve_resume()
    test_pause_approval_then_deny_resume()
    test_lease_conflict_blocks_concurrent_resume()
    test_unknown_tool_and_bad_args()
    test_tool_result_truncation()
    test_demo_business_tools_idempotency_and_approval()
    test_rule_based_demo_provider_approval_flow()
    test_checkpoint_list_tasks_purge_and_tool_describe()
    test_http_api_auth_approval_audit_and_errors()
    test_task_runner_exception_marks_failed()
    test_context_compaction()
    test_metrics_bounded_histogram()
    test_agent_config_from_env()
    test_streaming_retry_does_not_duplicate_failed_attempt_text()
    test_refusal_marks_failed()
    test_pause_turn_continues()
    test_cancel_task()
    test_wall_time_budget()
    test_parallel_read_tools_keep_order()
    test_thinking_blocks_roundtrip()
    test_summary_compaction_keeps_tool_pairing()
    test_worker_resumes_crashed_task()
    test_structured_output_and_reasoning_passthrough()
    test_openai_compat_conversion()
    test_checkpoint_resume()
    print("\n全部测试通过 [OK]")

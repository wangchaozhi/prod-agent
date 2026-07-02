"""Agent 主循环 — 把可靠性/可观测性/预算/安全四条线编排到一起。

每轮循环:
  1. 预算检查(超限 → BUDGET_EXCEEDED 优雅终止)+ 续期任务租约
  2. 上下文超阈值时压缩旧消息
  3. LLM 调用(带重试/熔断/超时/提示词缓存,可选流式回调)
  4. 若有 tool_use → 经权限/校验/审批后执行工具,结果回填
     (approval_mode="pause" 且无 approval_hook 时,EXECUTE 级调用
      落盘为 AWAITING_APPROVAL 并返回,等待外部审批后 resume 续跑)
  5. 落盘检查点
直到模型 end_turn 或预算耗尽。崩溃后可用 resume(task_id) 续跑。
"""
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .budget import BudgetExceeded, BudgetTracker
from .config import AgentConfig
from .context import ContextCompactor
from .llm import (CircuitOpen, LLMProvider, OnText, PermanentLLMError,
                  ResilientLLMClient, TransientLLMError)
from .memory import CheckpointStore, LeaseConflict
from .observability import Metrics, Tracer, get_logger
from .tools import ToolRegistry
from .types import (Message, Permission, Role, TaskState, TaskStatus, ToolCall,
                    ToolResult)


class Agent:
    def __init__(self, provider: LLMProvider, tools: ToolRegistry,
                 config: AgentConfig | None = None,
                 compactor: ContextCompactor | None = None) -> None:
        self.config = config or AgentConfig()
        self.logger = get_logger()
        self.metrics = Metrics()
        self.tracer = Tracer(self.logger, self.metrics)
        self.llm = ResilientLLMClient(provider, self.config, self.tracer, self.metrics)
        self.tools = tools
        self.store = CheckpointStore(self.config.checkpoint_db)
        # 可注入定制压缩器(如 LLM 摘要折叠、精确 token 计数)
        self.compactor = compactor or ContextCompactor(self.config.context)
        self._owner = uuid.uuid4().hex  # 租约持有者标识

    # ---------------- 公共 API ----------------

    def run(self, goal: str, *, output_schema: dict[str, Any] | None = None,
            on_text: OnText | None = None) -> TaskState:
        """output_schema: 传入 JSON Schema 时,最终回答被约束为该结构的 JSON。"""
        state = TaskState(goal=goal, status=TaskStatus.RUNNING,
                          output_schema=output_schema)
        state.messages.append(Message(role=Role.USER, content=goal))
        self.store.save(state)
        return self._run_leased(state, on_text=on_text)

    def cancel(self, task_id: str) -> bool:
        """请求取消任务(跨进程可用);执行循环在下一轮开始前以 CANCELLED 收尾。"""
        return self.store.request_cancel(task_id)

    def resume(self, task_id: str, *, approvals: dict[str, bool] | None = None,
               on_text: OnText | None = None) -> TaskState:
        """崩溃/中断/等待审批后从最后一个检查点续跑。

        approvals: 任务处于 AWAITING_APPROVAL 时必填,
                   {tool_call_id: True(批准)/False(拒绝)},须覆盖全部挂起调用。
        """
        state = self.store.load(task_id)
        if state is None:
            raise KeyError(f"task {task_id} 不存在")
        if state.status not in (TaskStatus.RUNNING, TaskStatus.PENDING,
                                TaskStatus.AWAITING_APPROVAL):
            return state  # 已终结的任务直接返回(幂等)
        state.status = TaskStatus.RUNNING
        return self._run_leased(state, approvals=approvals, on_text=on_text)

    async def arun(self, goal: str, *, on_text: OnText | None = None) -> TaskState:
        """异步封装(线程池执行,便于嵌入 Web 服务)。"""
        return await asyncio.to_thread(self.run, goal, on_text=on_text)

    async def aresume(self, task_id: str, *, approvals: dict[str, bool] | None = None,
                      on_text: OnText | None = None) -> TaskState:
        return await asyncio.to_thread(
            self.resume, task_id, approvals=approvals, on_text=on_text)

    # ---------------- 主循环 ----------------

    def _run_leased(self, state: TaskState, approvals: dict[str, bool] | None = None,
                    on_text: OnText | None = None) -> TaskState:
        if not self.store.acquire_lease(state.task_id, self._owner,
                                        self.config.lease_ttl_s):
            raise LeaseConflict(
                f"任务 {state.task_id} 的租约被其他执行者持有,拒绝并发执行")
        try:
            return self._loop(state, approvals=approvals, on_text=on_text)
        finally:
            self.store.release_lease(state.task_id, self._owner)

    def _loop(self, state: TaskState, approvals: dict[str, bool] | None = None,
              on_text: OnText | None = None) -> TaskState:
        budget = BudgetTracker(self.config.budget,
                               iterations=state.iterations, usage=state.usage)
        trace_id = state.task_id

        with self.tracer.span("task", trace_id, goal=state.goal[:200]):
            try:
                if state.pending_approval_ids:
                    self._resume_pending_tools(state, approvals, trace_id)

                while True:
                    if self.store.cancel_requested(state.task_id):
                        state.status = TaskStatus.CANCELLED
                        state.error = "收到外部取消请求"
                        self.store.save(state)
                        break
                    budget.check()
                    budget.iterations += 1
                    state.iterations = budget.iterations
                    self.store.acquire_lease(state.task_id, self._owner,
                                             self.config.lease_ttl_s)  # 续期
                    if self.compactor.compact_if_needed(state.messages):
                        self.metrics.inc("context.compactions")
                        self.logger.info("context.compacted",
                                         extra={"ctx": {"trace_id": trace_id}})

                    resp = self.llm.complete(
                        trace_id=trace_id,
                        system=self.config.system_prompt,
                        messages=state.messages,
                        tools=self.tools.specs_for_llm(),
                        on_text=on_text,
                        output_schema=state.output_schema,
                    )
                    budget.record(resp.usage)
                    state.usage = budget.usage
                    state.cost_usd = round(budget.cost_usd, 6)
                    state.messages.append(resp.message)

                    if resp.stop_reason == "pause_turn":
                        # 服务端工具暂停;历史已含本轮 assistant 消息,重发即续跑
                        # (预算护栏兜底,不会无限暂停)
                        self.store.save(state)
                        continue

                    if resp.stop_reason == "refusal":
                        state.status = TaskStatus.FAILED
                        state.error = "模型出于安全策略拒绝生成 (stop_reason=refusal)"
                        self.store.save(state)
                        break

                    if resp.stop_reason == "model_context_window_exceeded":
                        state.status = TaskStatus.FAILED
                        state.error = ("上下文窗口耗尽;可调低 ContextConfig."
                                       "max_context_tokens 让压缩更早介入")
                        self.store.save(state)
                        break

                    if resp.stop_reason == "max_tokens":
                        # 输出被截断不是正常完成,标记失败并留出补救线索
                        state.status = TaskStatus.FAILED
                        state.error = ("LLM 输出因 max_tokens 截断;"
                                       "可提高 max_tokens_per_call 后重试")
                        self.store.save(state)
                        break

                    if resp.stop_reason != "tool_use" or not resp.message.tool_calls:
                        if resp.stop_reason not in ("end_turn", "stop_sequence"):
                            self.logger.warning(
                                "unknown_stop_reason", extra={"ctx": {
                                    "trace_id": trace_id,
                                    "stop_reason": resp.stop_reason}})
                        state.status = TaskStatus.COMPLETED
                        self.store.save(state)
                        break

                    calls = resp.message.tool_calls
                    if self.config.approval_mode == "pause" and self.tools.approval_hook is None:
                        pending = [c.id for c in calls if self.tools.needs_approval(c)]
                        if pending:
                            # 批次内任一调用需审批 → 全部不执行,落盘等待,避免部分执行的幂等问题
                            state.status = TaskStatus.AWAITING_APPROVAL
                            state.pending_approval_ids = pending
                            self.store.save(state)
                            break

                    self._exec_tools(state, calls, trace_id)

            except BudgetExceeded as e:
                state.status = TaskStatus.BUDGET_EXCEEDED
                state.error = e.reason
                self.store.save(state)
            except CircuitOpen as e:
                state.status = TaskStatus.FAILED
                state.error = repr(e)
                self.store.save(state)
            except (TransientLLMError, PermanentLLMError) as e:
                state.status = TaskStatus.FAILED
                state.error = repr(e)
                self.store.save(state)

        return state

    # ---------------- 工具执行 ----------------

    def _exec_tools(self, state: TaskState, calls: list[ToolCall], trace_id: str,
                    approvals: dict[str, bool] | None = None) -> None:
        approvals = approvals or {}

        def run_one(call: ToolCall) -> ToolResult:
            with self.tracer.span("tool.exec", trace_id, tool=call.name) as ctx:
                r = self.tools.execute(call, approved=approvals.get(call.id))
                ctx["is_error"] = r.is_error
                self.metrics.inc("tool.errors" if r.is_error else "tool.calls")
                return r

        if self._parallel_safe(calls):
            with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
                results = list(pool.map(run_one, calls))  # map 保持结果顺序
        else:
            results = [run_one(c) for c in calls]
        state.messages.append(Message(role=Role.USER, tool_results=results))
        self.store.save(state)  # 每轮落盘 = 检查点

    def _parallel_safe(self, calls: list[ToolCall]) -> bool:
        """整批都是 READ 级工具才并行;WRITE/EXECUTE 保持串行,避免副作用竞态。"""
        if len(calls) < 2:
            return False
        specs = [self.tools.get(c.name) for c in calls]
        return all(s is not None and s.permission == Permission.READ for s in specs)

    def _resume_pending_tools(self, state: TaskState,
                              approvals: dict[str, bool] | None,
                              trace_id: str) -> None:
        """AWAITING_APPROVAL 恢复:按审批结果执行挂起的工具批次。"""
        approvals = approvals or {}
        missing = [i for i in state.pending_approval_ids if i not in approvals]
        if missing:
            raise ValueError(
                f"任务等待审批,resume 必须提供 approvals 决定,缺少: {missing}")
        last = state.messages[-1]
        state.pending_approval_ids = []
        self._exec_tools(state, last.tool_calls, trace_id, approvals=approvals)

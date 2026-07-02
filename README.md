# prod-agent — 生产级通用任务编排 Agent

零重型依赖(核心只用 Python 标准库,LLM 调用按需选装 `anthropic` / `openai` SDK)、可插拔 Provider、四大生产特性内建。

内建 Provider:`AnthropicProvider`(全部 Claude 模型,含提示词缓存断点)、
`OpenAICompatProvider`(OpenAI / DeepSeek / Qwen / vLLM 等一切 OpenAI 格式端点)、
`MockProvider`(测试)。换厂商记得同步改 `BudgetConfig` 四个单价。

## 架构

```
┌──────────────────────── Agent 主循环 (core.py) ─────────────────────────┐
│ 预算检查 → 租约续期 → 上下文压缩 → LLM调用 → 工具执行/审批挂起 → 检查点  │
└─────┬──────────┬──────────────┬─────────────┬──────────────┬────────────┘
      │          │              │             │              │
  BudgetTracker  ResilientLLM   Compactor     ToolRegistry   CheckpointStore
  (budget.py)    (llm.py)       (context.py)  (tools.py)     (memory.py)
  迭代/token/    重试+退避+超时  历史消息压缩   白名单+权限+    SQLite WAL
  成本三重护栏    +熔断+提示词    防撞上下文     校验+同步/异步   崩溃可恢复
                 缓存+流式       上限          审批            +任务租约
      └──────────┴──────────────┴─────────────┴──────────────┘
                   Tracer / Metrics / JSON 日志 (observability.py)
```

## 四大生产特性

**可靠性** — 仅重试瞬态错误(429/5xx/超时),指数退避 + 随机抖动;熔断器连续失败后快速失败、冷却后半开试探(熔断触发时任务以 FAILED 落盘,不会击穿主循环);stop_reason 全语义处理:`max_tokens` 截断/`refusal` 拒答/上下文耗尽 → FAILED(带补救提示),`pause_turn` 自动续跑;每轮落盘检查点,`agent.resume(task_id)` 从断点续跑;`agent.cancel(task_id)` 跨进程取消,任务以 CANCELLED 优雅收尾;任务租约防止多进程并发执行同一 task_id(`LeaseConflict`),租约带 TTL,持有者崩溃后自动过期;`Worker` 轮询认领崩溃遗留任务;工具异常被隔离为 error result,`pass_call_id=True` 可让工具拿到调用 id 实现崩溃重放去重。

**可观测性** — 每任务一个 trace_id、每次 LLM/工具调用一个 span;JSON 结构化日志可直接被 ELK/Loki 采集;`agent.metrics.snapshot()` 导出调用数、错误数、token 量(含缓存读写)、延迟分布(流式聚合,内存 O(1),长期运行不泄漏);日志自动脱敏 api_key/token 等字段。

**成本控制** — 迭代次数、总 token、美元成本、墙钟时长四重硬上限,任一触发即以 `BUDGET_EXCEEDED` 优雅终止;默认开启提示词缓存(system/tools/最新消息打 `cache_control` 断点),多轮循环中缓存读价约为输入价 0.1x;缓存读/写 token 独立计价入账;价格通过配置注入(模型价格会变,见 https://docs.claude.com 实际定价);工具输出超长自动截断;上下文超阈值自动压缩旧消息(占位符模式零成本;注入 `summarize_fn` 可升级为 LLM 摘要折叠,注入 `token_counter` 可接 API 精确计量);READ 级工具批次自动并行,降低延迟。

**安全** — 工具白名单注册,未注册即拒绝;READ/WRITE/EXECUTE 三级权限;EXECUTE 级支持两种审批模式(见下);调用前 JSON Schema 参数校验;API Key 只从环境变量读取,绝不落盘。

## 快速开始

```bash
pip install -e ".[anthropic]"     # 或最小化: pip install anthropic
export ANTHROPIC_API_KEY=sk-...
cd examples && python run.py
```

```python
from agent import Agent, AgentConfig, AnthropicProvider, ToolRegistry, Permission

tools = ToolRegistry(approval_hook=my_approval_fn)

@tools.tool(name="get_weather", description="查天气",
            input_schema={"type": "object",
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"]},
            permission=Permission.READ)
def get_weather(city: str) -> str: ...

agent = Agent(AnthropicProvider(AgentConfig.api_key()), tools)
state = agent.run("查北京天气并总结",
                  on_text=lambda t: print(t, end=""))   # 可选:流式输出
print(state.status, state.cost_usd, state.messages[-1].content)
```

嵌入异步服务时用 `await agent.arun(goal)` / `await agent.aresume(task_id)`(线程池封装)。

接 OpenAI 兼容端点(DeepSeek / Qwen / vLLM / OpenAI):

```python
from agent import Agent, AgentConfig, BudgetConfig, OpenAICompatProvider

provider = OpenAICompatProvider(api_key="sk-...", base_url="https://api.deepseek.com")
agent = Agent(provider, tools, AgentConfig(
    model="deepseek-chat",
    budget=BudgetConfig(price_per_mtok_input=0.27, price_per_mtok_output=1.1,
                        price_per_mtok_cache_read=0.07),  # 按厂商实际定价配置
))
```

## 人工审批的两种模式

| 模式 | 配置 | 行为 |
|---|---|---|
| 同步(默认) | `ToolRegistry(approval_hook=fn)` | EXECUTE 级调用时同步调用 hook,阻塞等待放行;无 hook 直接拒绝 |
| 异步挂起 | `AgentConfig(approval_mode="pause")` 且不传 hook | 任务落盘为 `AWAITING_APPROVAL` 并返回(整批工具都不执行,无部分执行的幂等问题);外部审批系统(工单/IM 机器人)决策后续跑 |

```python
state = agent.run("部署新版本")                 # → status == AWAITING_APPROVAL
ids = state.pending_approval_ids                # 挂起的 tool_call id
# ... 审批系统回调后 ...
state = agent.resume(state.task_id, approvals={ids[0]: True})   # False 则拒绝
```

## 进阶用法

```python
# 结构化输出:最终回答被约束为 JSON Schema(随检查点持久化)
state = agent.run("提取联系人信息", output_schema={
    "type": "object",
    "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
    "required": ["name", "email"], "additionalProperties": False,
})

# 新一代模型的推理配置(Sonnet 5 / Fable 5 默认开启 adaptive thinking,
# 框架会存储并回传 thinking 块,工具循环不会 400)
config = AgentConfig(model="claude-sonnet-5", thinking="adaptive", effort="high")

# 取消与超时
agent.cancel(task_id)                            # 跨进程,下一轮循环生效
AgentConfig(budget=BudgetConfig(max_wall_time_s=600))   # 单次运行 10 分钟上限

# Worker:轮询认领崩溃遗留的 RUNNING 任务(配租约,多进程安全)
from agent import Worker
Worker(agent, poll_interval_s=5).run_forever()
```

命令行(不注册工具的纯对话模式):

```bash
python -m agent "总结一下量子计算的现状"
python -m agent --provider openai --base-url https://api.deepseek.com --model deepseek-chat "..."
python -m agent --resume <task_id> --approvals '{"call_id": true}'
```

## 测试

```bash
python tests/test_agent.py    # 无需 API Key,使用 MockProvider
python -m pytest tests/ -q   # 或 pytest
```

覆盖:工具调用闭环、瞬态错误重试、熔断落盘、max_tokens/refusal/pause_turn 停止语义、取消、墙钟预算、迭代/成本预算拦截、缓存计费、EXECUTE 权限拒绝、异步审批挂起/批准/拒绝、租约冲突、Worker 认领、并行工具顺序、thinking 块往返、参数校验、工具输出截断、两种上下文压缩、结构化输出透传、指标聚合、检查点恢复。CI(GitHub Actions)在 Python 3.10–3.13 上跑 ruff + pytest。

## 上生产的替换点

| 组件 | 当前实现 | 生产建议 |
|---|---|---|
| 检查点 | SQLite WAL + 租约 + 取消标记 | Postgres / DynamoDB(接口不变) |
| 指标 | 进程内流式聚合 | prometheus_client |
| Trace | 自研轻量 span | OpenTelemetry(接口对齐,易迁移) |
| 审批 | input() 示例 / pause 模式 | 审批工单 / IM 机器人回调 + `resume(approvals=...)` |
| 参数校验 | jsonschema(装了即用)/ 轻量回退 | — |
| 上下文压缩 | 占位符 / 摘要折叠(可注入) | summarize_fn 接 LLM + token_counter 接 count_tokens |
| 任务分发 | Worker 轮询 + 租约 | 任务队列(Celery/SQS)+ 幂等 task_id |
| 异步 | arun/aresume(线程池封装) | 高并发场景换 AsyncAnthropic + 异步存储 |

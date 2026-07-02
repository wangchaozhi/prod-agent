# Changelog

## 0.3.0 (2026-07-03)

新增:

- **thinking 块存储与回传** — 适配 Sonnet 5 / Fable 5 等默认开启 adaptive thinking 的模型
  (同模型续跑要求原样回传,否则工具循环可能 400);`AgentConfig` 新增 `thinking` / `effort`
- **stop_reason 完整处理** — `refusal` → FAILED(此前会被误判为完成)、`pause_turn` → 自动重发续跑、
  `model_context_window_exceeded` → FAILED(带调参提示)、未知原因记 warning
- **任务取消** — `agent.cancel(task_id)` 跨进程生效,任务以新状态 `CANCELLED` 优雅收尾
- **墙钟预算** — `BudgetConfig.max_wall_time_s`,单次运行时长第四道护栏
- **Worker 模式** — `Worker(agent).run_forever()` 轮询认领崩溃遗留任务,配合租约多进程安全
- **READ 级工具批次并行执行** — 线程池执行,保持结果顺序;WRITE/EXECUTE 维持串行
- **结构化输出** — `agent.run(goal, output_schema=...)`,映射 Anthropic `output_config.format` /
  OpenAI `response_format`,schema 随检查点持久化,resume 后依然生效
- **上下文压缩升级** — 可注入 token 计数器(如 API count_tokens)与 LLM 摘要折叠函数,
  折叠边界自动对齐 tool_use/tool_result 配对
- **工具幂等** — `ToolSpec(pass_call_id=True)` 把 `tool_call_id` 传给工具函数,
  供崩溃重放窗口内去重
- **jsonschema 可选接入** — 装了即用完整校验(保留白名单参数检查),未装回退轻量自研
- **CLI** — `python -m agent "任务目标"`,支持 `--provider/--model/--base-url/--resume/--approvals`

修复:

- Windows GBK 控制台中文日志乱码(日志流包 UTF-8,不影响 print)
- 墙钟计时改用 `perf_counter`(Windows 上 `monotonic` 分辨率仅 ~16ms)

## 0.2.0 (2026-07-03)

- 修复:`CircuitOpen` 击穿主循环、`max_tokens` 截断被误判为完成、
  `AgentConfig.require_approval_for_execute` 死配置、Metrics 直方图无界增长
- 新增:提示词缓存(system/tools/最新消息断点)+ 缓存 token 独立计费、上下文压缩、
  异步审批(AWAITING_APPROVAL + `resume(approvals=...)`)、任务租约防并发双跑、
  流式输出 `on_text`、`arun`/`aresume`、`OpenAICompatProvider`
- 工程化:pyproject 打包、GitHub Actions CI(ruff + pytest);测试 7 → 17

## 0.1.0

- 初始版本:Agent 主循环、瞬态错误重试 + 熔断、迭代/token/成本三重预算护栏、
  工具白名单/三级权限/人工审批、SQLite WAL 检查点、JSON 日志 + trace + metrics

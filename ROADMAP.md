# prod-agent Roadmap

## 当前基线

0.4.0 是当前工作区基线,尚未提交/发布:

- HTTP API 与内置 Web 页面:提交任务、查看状态、取消、审批、工具列表、metrics
- Demo 业务工具:`query_order` / `search_orders` / `update_order_status` /
  `create_ticket` / `send_customer_message` / `refund_order`
- 无 API Key 可运行的 `demo` provider
- `CheckpointStore.list_tasks` / `purge_terminal` 与 `ToolRegistry.describe`
- 静态 Bearer token 鉴权、审批审计、Prometheus metrics 文本格式
- Dockerfile、Compose、systemd、nginx 示例、部署安全文档
- 自动化测试与 `ruff check .` 通过
- 手工 HTTP smoke:提交任务 + EXECUTE 审批闭环通过

下面清单只保留还没完成的工作。工作量:S = 半天内,M = 1~2 天,L = 3 天以上。
每项含验收标准,完成勾选即可。

---

## M0 · 收尾服务化第一版(最高优先级)

- [x] **0.1 HTTP 端点集成测试入库** `M`
  用 `ThreadingHTTPServer` 起临时服务(随机端口)+ `urllib` 实测:
  - `POST /api/tasks` 提交 → 轮询 `GET /api/tasks/{id}` 到终态
  - demo provider 审批闭环:提交 EXECUTE 任务 → AWAITING_APPROVAL →
    `POST .../approvals` 批准/拒绝 → 终态正确
  - `POST .../cancel`、`GET /api/tasks` 列表、`GET /api/tools`、`GET /api/metrics`
  - 错误路径:404(任务不存在)、400(缺 goal / 非法 JSON)、审批缺 id
  - 验收:pytest 全绿,覆盖以上每条路径至少一个断言

- [x] **0.2 新增底层方法单元测试** `S`
  - `CheckpointStore.list_tasks`:按 updated_at 倒序、limit 截断(1~500 夹取)、状态过滤
  - `ToolRegistry.describe`:字段完整性(permission/pass_call_id/max_result_chars)
  - 验收:两个方法各有独立测试函数

- [x] **0.3 server 并发与异常路径复查** `S`
  - 确认 HTTP server 用 ThreadingHTTPServer(并发请求不互相阻塞)
  - `TaskRunner._spawn` 后台线程崩溃时任务状态必须落盘(不能停留 RUNNING)
  - 同一 task_id 重复审批/取消的幂等行为(依赖租约,给出明确 409/错误响应)
  - 验收:代码走查 + 每个结论对应一个测试或注释说明

- [ ] **0.4 发布 0.4.0** `S`
  - CHANGELOG 增补 0.4.0(HTTP API、Web 页面、demo 工具、demo provider)
  - pyproject 版本号 bump;README 与实际端点核对
  - commit 后 CI 绿
  - 验收:GitHub Actions 通过,发布包版本与文档一致

## M1 · 发布准备(开源/分发前必做)

- [ ] **1.1 LICENSE 文件** `S` ⚠️ 需要你决定:MIT(宽松简短)或 Apache-2.0(带专利条款)
  - 根目录 LICENSE + pyproject `license` 字段 + README 徽章
- [ ] **1.2 PyPI 发布流水线** `S`
  - `.github/workflows/release.yml`:打 tag → build → `pypa/gh-action-pypi-publish`
  - 需要在仓库 Settings 配 `PYPI_API_TOKEN` secret(或 Trusted Publisher)
  - 验收:打 `v0.4.0` tag 后 `pip install prod-agent` 可用
- [ ] **1.3 英文版 README**(可选,面向开源社区)`M`

## M2 · 服务硬化(对公网/团队暴露前必做)

- [x] **2.1 API 鉴权** `M`
  - 最小实现:环境变量注入静态 Bearer token,所有 `/api/*` 校验
  - 进阶:双 token 区分「提交者」与「审批者」角色(RBAC 雏形)
  - 验收:无 token 401;错误 token 403;页面支持填 token
- [x] **2.2 审批审计** `M`
  - 新表 approval_audit:task_id、call_id、决定、审批人、理由、时间
  - `POST .../approvals` 接受可选 `approver`/`reason` 字段并落库
  - `GET /api/tasks/{id}` 返回审计记录
  - 验收:审批一次后审计可查,含全部字段
- [x] **2.3 输入校验与错误响应规范化** `S`
  - 统一错误结构 `{"error": {"code": ..., "message": ...}}`;goal 长度上限;
    approvals 值必须为 bool
- [x] **2.4 部署安全文档** `S`
  - 明确默认绑定 127.0.0.1;对外必须反代(nginx/caddy)+ TLS;示例配置

## M3 · 运维与可靠性

- [x] **3.1 检查点保留策略** `S`
  - `CheckpointStore.purge_terminal(older_than_days)`(只清终态任务)
  - Worker.run_forever 支持定期调用;CLI 加 `--purge-days`
  - 验收:测试构造过期终态任务被清、RUNNING 不被清
- [x] **3.2 全局限流** `M`
  - 进程级 token bucket(次/分钟 + 并发上限),Provider 调用前取令牌
  - 多 Worker 共享限额进阶:基于 SQLite/Redis 的分布式桶(可后置)
  - 验收:压测下调用速率不超过配置值
- [x] **3.3 流式重试重复输出修复** `S`
  - 现状:瞬态错误重试时 on_text 会把已流式输出的文本重吐一遍
  - 方案:重试的中间尝试抑制 on_text,仅最终成功尝试回调;或回调层去重
  - 验收:fail_first_n=1 + 流式的测试,回调文本无重复
- [x] **3.4 /api/metrics Prometheus 文本格式** `S`
  - snapshot → Prometheus exposition format(counter/summary),`?format=prom`
  - 验收:promtool check metrics 通过(或格式对拍测试)
- [ ] **3.5 OpenTelemetry 适配层**(可选)`M`
  - Tracer 接口对齐 OTel span;otel 装了即用,没装保持自研
- [x] **3.6 配置外部化** `M`
  - `AgentConfig.from_env()`(PROD_AGENT_MODEL 等)+ 可选 YAML 加载
  - server/CLI/Worker 全部走同一加载路径
  - 验收:不改代码,仅环境变量即可完成模型/预算/审批模式配置
- [x] **3.7 部署模板** `M`
  - Dockerfile(多阶段,非 root)、docker-compose.yml(server + worker)、
    systemd unit、`.env.example`
  - 验收:`docker compose up` 一键起 demo 模式

## M4 · 能力扩展(按需求排期)

- [ ] **4.1 多轮会话** `M`
  - `agent.send(task_id, text)`:对 COMPLETED/CANCELLED 任务追加用户消息续跑
    (复用租约与检查点;状态回 RUNNING)
  - HTTP:`POST /api/tasks/{id}/messages`;页面加输入框
  - 验收:同一 task 两轮问答,历史连续、成本累计
- [ ] **4.2 PostgresCheckpointStore** `M`
  - 与 SQLite 版同接口(save/load/lease/cancel/list);psycopg 可选依赖
  - 验收:现有全部测试参数化后对 Postgres 同样通过(CI 加 services: postgres)
- [ ] **4.3 任务队列适配** `L`
  - 投递:submit 写队列(SQS/Redis/Celery);消费:Worker 从队列取 task_id
    (替代轮询);幂等靠现有租约
- [ ] **4.4 record/replay 评估** `M`
  - RecordingProvider 包装真实 Provider,交互序列化落盘;
    ReplayProvider 从文件回放;`tests/evals/` 断言关键行为
  - 验收:录一条真实交互,离线回放测试稳定通过
- [ ] **4.5 真·异步管道** `L`
  - AsyncAgent + AsyncAnthropic/AsyncOpenAI + aiosqlite;
    等有真实高并发需求再启动(维护面翻倍)
- [ ] **4.6 子 Agent 编排** `L`
  - fan-out 委托、子任务预算分配、结果汇总;设计先行(独立设计文档)

## M5 · 真实业务接入(全部 ⚠️ 需要你提供信息)

- [ ] **5.1 真实订单/工单/CRM 工具** — 需要:数据库表结构或内部 API 文档、测试环境
  - 替换 demo_tools 各函数体;权限分级沿用 READ/WRITE/EXECUTE
- [ ] **5.2 真实消息通道** — 需要:通道选择(邮件/短信/企业微信/飞书/Slack)+ 凭据
  - `send_customer_message` 接真实 webhook/SDK;失败重试与投递回执
- [ ] **5.3 真实审批通道** — 需要:审批发生在哪(飞书审批/钉钉/内部工单)
  - IM 机器人卡片 → 回调 `POST /api/tasks/{id}/approvals`;联动 2.2 审计

---

## 依赖关系

```
M0 ──► M1(发布)──► 对外分发
 │
 ├──► M2(硬化)──► 对外/团队部署 ──► M5(业务接入)
 │
 └──► M3(运维)按需并行;M4 按需求单独排期
```

## 建议节奏

| 批次 | 内容 | 预估 |
|---|---|---|
| 第 1 批 | M0 全部 + 1.1 + 1.2 | 1~2 天 |
| 第 2 批 | M2 全部 + 3.1 + 3.3 | 2~3 天 |
| 第 3 批 | 3.2 + 3.4 + 3.6 + 3.7 | 2~3 天 |
| 之后 | M4 按需求、M5 等业务信息到位 | — |

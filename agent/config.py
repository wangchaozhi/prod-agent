"""配置 — 密钥永远只从环境变量读取,绝不落盘/入库/打日志。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter: float = 0.3          # 随机抖动比例,避免重试风暴
    timeout_s: float = 120.0     # 单次 LLM 调用超时


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5   # 连续失败 N 次后熔断
    recovery_time_s: float = 30.0


@dataclass(frozen=True)
class BudgetConfig:
    max_iterations: int = 20            # 主循环最大轮数
    max_total_tokens: int = 500_000     # 单任务 token 硬上限(含缓存读写)
    max_cost_usd: float = 5.0           # 单任务成本硬上限
    max_wall_time_s: float = 0.0        # 单次运行墙钟上限(0 = 不限;resume 后重新计时)
    # 价格随时变动,务必按 https://docs.claude.com 实际价格配置 (USD / 百万 token)
    # 默认为 claude-opus-4-8:输入 $5 / 输出 $25,缓存写 1.25x、缓存读 0.1x 输入价
    price_per_mtok_input: float = 5.0
    price_per_mtok_output: float = 25.0
    price_per_mtok_cache_write: float = 6.25
    price_per_mtok_cache_read: float = 0.50


@dataclass(frozen=True)
class ContextConfig:
    """上下文窗口管理 — 超过阈值时压缩旧消息,防止长任务撞上下文上限。"""
    max_context_tokens: int = 150_000   # 触发压缩的估算 token 阈值(按 ~4 字符/token 粗估)
    keep_recent_messages: int = 10      # 最近 N 条消息永不压缩(首条任务目标同样受保护)


@dataclass(frozen=True)
class AgentConfig:
    model: str = "claude-opus-4-8"
    max_tokens_per_call: int = 4096
    # 推理配置(Claude 4.6+ / Sonnet 5 / Fable 5 系列):
    #   thinking: None=不传该参数,按模型默认 | "adaptive" | "disabled"(Fable 5 不接受 disabled)
    #   effort:   None=不传 | "low" / "medium" / "high" / "max"(按模型支持情况)
    # OpenAI 兼容 Provider 会忽略这两项(各家推理参数不统一)。
    thinking: str | None = None
    effort: str | None = None
    system_prompt: str = "You are a reliable task-orchestration agent. Use tools when needed; be concise."
    retry: RetryConfig = field(default_factory=RetryConfig)
    breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    checkpoint_db: str = "agent_checkpoints.db"
    # EXECUTE 级工具审批策略(工具是否需要审批由 ToolRegistry 声明,这里只决定无 hook 时的行为):
    #   "sync"  — 无 approval_hook 时直接拒绝执行(默认,行为可预期)
    #   "pause" — 无 approval_hook 时任务落盘为 AWAITING_APPROVAL,
    #             由外部审批后 agent.resume(task_id, approvals={call_id: bool}) 续跑
    approval_mode: str = "sync"
    lease_ttl_s: float = 300.0           # 任务租约时长,防止多进程并发执行同一 task_id
    enable_prompt_caching: bool = True   # 对 system/tools/最新消息打缓存断点,多轮循环显著省钱

    @staticmethod
    def api_key() -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY 未设置 (密钥只能通过环境变量注入)")
        return key

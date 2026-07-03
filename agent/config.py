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
class RateLimitConfig:
    """进程级 Provider 调用限流。0 表示不限制。"""
    max_calls_per_minute: float = 0.0
    max_concurrent_calls: int = 0


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
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
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

    @staticmethod
    def from_env(prefix: str = "PROD_AGENT_") -> "AgentConfig":
        """Build config from environment variables.

        Examples:
          PROD_AGENT_MODEL=claude-sonnet-5
          PROD_AGENT_BUDGET_MAX_COST_USD=1
          PROD_AGENT_RATE_MAX_CALLS_PER_MINUTE=60
        """

        def get(name: str, default: str | None = None) -> str | None:
            return os.environ.get(prefix + name, default)

        def as_int(name: str, default: int) -> int:
            raw = get(name)
            return default if raw in (None, "") else int(raw)

        def as_float(name: str, default: float) -> float:
            raw = get(name)
            return default if raw in (None, "") else float(raw)

        def as_bool(name: str, default: bool) -> bool:
            raw = get(name)
            if raw in (None, ""):
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        base = AgentConfig()
        retry = RetryConfig(
            max_attempts=as_int("RETRY_MAX_ATTEMPTS", base.retry.max_attempts),
            base_delay_s=as_float("RETRY_BASE_DELAY_S", base.retry.base_delay_s),
            max_delay_s=as_float("RETRY_MAX_DELAY_S", base.retry.max_delay_s),
            jitter=as_float("RETRY_JITTER", base.retry.jitter),
            timeout_s=as_float("RETRY_TIMEOUT_S", base.retry.timeout_s),
        )
        breaker = CircuitBreakerConfig(
            failure_threshold=as_int("BREAKER_FAILURE_THRESHOLD",
                                     base.breaker.failure_threshold),
            recovery_time_s=as_float("BREAKER_RECOVERY_TIME_S",
                                     base.breaker.recovery_time_s),
        )
        budget = BudgetConfig(
            max_iterations=as_int("BUDGET_MAX_ITERATIONS",
                                  base.budget.max_iterations),
            max_total_tokens=as_int("BUDGET_MAX_TOTAL_TOKENS",
                                    base.budget.max_total_tokens),
            max_cost_usd=as_float("BUDGET_MAX_COST_USD",
                                  base.budget.max_cost_usd),
            max_wall_time_s=as_float("BUDGET_MAX_WALL_TIME_S",
                                     base.budget.max_wall_time_s),
            price_per_mtok_input=as_float("PRICE_PER_MTOK_INPUT",
                                          base.budget.price_per_mtok_input),
            price_per_mtok_output=as_float("PRICE_PER_MTOK_OUTPUT",
                                           base.budget.price_per_mtok_output),
            price_per_mtok_cache_write=as_float(
                "PRICE_PER_MTOK_CACHE_WRITE",
                base.budget.price_per_mtok_cache_write),
            price_per_mtok_cache_read=as_float(
                "PRICE_PER_MTOK_CACHE_READ",
                base.budget.price_per_mtok_cache_read),
        )
        context = ContextConfig(
            max_context_tokens=as_int("CONTEXT_MAX_TOKENS",
                                      base.context.max_context_tokens),
            keep_recent_messages=as_int("CONTEXT_KEEP_RECENT_MESSAGES",
                                        base.context.keep_recent_messages),
        )
        rate_limit = RateLimitConfig(
            max_calls_per_minute=as_float("RATE_MAX_CALLS_PER_MINUTE",
                                          base.rate_limit.max_calls_per_minute),
            max_concurrent_calls=as_int("RATE_MAX_CONCURRENT_CALLS",
                                        base.rate_limit.max_concurrent_calls),
        )
        return AgentConfig(
            model=get("MODEL", base.model) or base.model,
            max_tokens_per_call=as_int("MAX_TOKENS_PER_CALL",
                                       base.max_tokens_per_call),
            thinking=get("THINKING", base.thinking),
            effort=get("EFFORT", base.effort),
            system_prompt=get("SYSTEM_PROMPT", base.system_prompt) or base.system_prompt,
            retry=retry,
            breaker=breaker,
            budget=budget,
            context=context,
            rate_limit=rate_limit,
            checkpoint_db=get("CHECKPOINT_DB", base.checkpoint_db) or base.checkpoint_db,
            approval_mode=get("APPROVAL_MODE", base.approval_mode) or base.approval_mode,
            lease_ttl_s=as_float("LEASE_TTL_S", base.lease_ttl_s),
            enable_prompt_caching=as_bool("ENABLE_PROMPT_CACHING",
                                          base.enable_prompt_caching),
        )

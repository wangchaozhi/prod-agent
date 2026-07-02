"""成本控制 — 三重护栏:迭代上限、token 上限、成本(美元)上限。

任何一道护栏触发都会让任务以 BUDGET_EXCEEDED 优雅终止,
而不是无声地继续烧钱。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import BudgetConfig
from .types import Usage


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class BudgetTracker:
    config: BudgetConfig
    iterations: int = 0
    usage: Usage = None  # type: ignore[assignment]
    # perf_counter:Windows 上 monotonic 分辨率仅 ~16ms,perf_counter 是纳秒级
    started_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = Usage()

    @property
    def total_tokens(self) -> int:
        u = self.usage
        return (u.input_tokens + u.output_tokens
                + u.cache_write_tokens + u.cache_read_tokens)

    @property
    def cost_usd(self) -> float:
        c, u = self.config, self.usage
        return (
            u.input_tokens / 1_000_000 * c.price_per_mtok_input
            + u.output_tokens / 1_000_000 * c.price_per_mtok_output
            + u.cache_write_tokens / 1_000_000 * c.price_per_mtok_cache_write
            + u.cache_read_tokens / 1_000_000 * c.price_per_mtok_cache_read
        )

    def record(self, usage: Usage) -> None:
        self.usage += usage

    def check(self) -> None:
        """每轮循环开始前调用;超限抛 BudgetExceeded。"""
        if self.iterations >= self.config.max_iterations:
            raise BudgetExceeded(f"迭代次数达到上限 {self.config.max_iterations}")
        if self.total_tokens >= self.config.max_total_tokens:
            raise BudgetExceeded(
                f"token 用量 {self.total_tokens} 达到上限 {self.config.max_total_tokens}"
            )
        if self.cost_usd >= self.config.max_cost_usd:
            raise BudgetExceeded(
                f"成本 ${self.cost_usd:.4f} 达到上限 ${self.config.max_cost_usd}"
            )
        if self.config.max_wall_time_s > 0:
            elapsed = time.perf_counter() - self.started_at
            if elapsed >= self.config.max_wall_time_s:
                raise BudgetExceeded(
                    f"墙钟时间 {elapsed:.1f}s 达到上限 {self.config.max_wall_time_s}s"
                )

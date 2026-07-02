"""核心数据模型 — 零外部依赖,全部基于 dataclass。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Permission(str, Enum):
    """工具权限分级。EXECUTE 默认需要人工确认。"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    AWAITING_APPROVAL = "awaiting_approval"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # 模型的 thinking/redacted_thinking 原始块(含 signature)。
    # 开启 adaptive thinking 的模型要求同模型续跑时原样回传,否则工具循环可能 400。
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0  # 提示词缓存写入(计费约为输入价 1.25x)
    cache_read_tokens: int = 0   # 提示词缓存读取(计费约为输入价 0.1x)

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        return self


@dataclass
class LLMResponse:
    message: Message
    usage: Usage
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"


@dataclass
class TaskState:
    """任务状态 — 可序列化,用于检查点持久化。"""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    messages: list[Message] = field(default_factory=list)
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    error: str | None = None
    # AWAITING_APPROVAL 时挂起的 tool_call id 列表;resume 时须逐一给出批准/拒绝
    pending_approval_ids: list[str] = field(default_factory=list)
    # 结构化输出 JSON Schema(随任务持久化,resume 后依然生效)
    output_schema: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

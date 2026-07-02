"""LLM 客户端 — 可插拔 Provider + 重试/退避/超时/熔断 + 提示词缓存 + 流式输出。

可靠性策略:
- 仅重试瞬态错误(429/5xx/网络超时);4xx 业务错误立即失败
- 指数退避 + 随机抖动,避免重试风暴
- 熔断器:连续失败超阈值后快速失败,冷却期后半开试探
- 每次调用强制超时

成本策略:
- caching=True 时对 system / tools / 最新消息打 cache_control 断点,
  Agent 循环里 system+tools+历史每轮重发,缓存读价约为输入价 0.1x
"""
from __future__ import annotations

import abc
import random
import time
from typing import Any, Callable

from .config import AgentConfig, CircuitBreakerConfig, RetryConfig
from .observability import Metrics, Tracer
from .types import LLMResponse, Message, Role, ToolCall, Usage

OnText = Callable[[str], None]


class TransientLLMError(Exception):
    """可重试错误:限流、5xx、超时、网络抖动。"""


class PermanentLLMError(Exception):
    """不可重试错误:鉴权失败、请求非法等。"""


class CircuitOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, cfg: CircuitBreakerConfig) -> None:
        self.cfg = cfg
        self.failures = 0
        self.opened_at: float | None = None

    def before_call(self) -> None:
        if self.opened_at is None:
            return
        if time.monotonic() - self.opened_at < self.cfg.recovery_time_s:
            raise CircuitOpen(
                f"熔断器打开,{self.cfg.recovery_time_s}s 冷却中,快速失败"
            )
        # 冷却结束 → 半开,放一个请求试探
        self.opened_at = None
        self.failures = self.cfg.failure_threshold - 1

    def on_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.cfg.failure_threshold:
            self.opened_at = time.monotonic()


class LLMProvider(abc.ABC):
    """Provider 抽象 — 换厂商只需实现这个接口。

    caching:       是否在请求中打提示词缓存断点(不支持的 Provider 可忽略)
    on_text:       非 None 时应流式生成,每产出一段文本回调一次(不支持可忽略)
    thinking:      "adaptive"/"disabled"/None,推理开关(不支持可忽略)
    effort:        "low"/"medium"/"high"/"max"/None,推理力度(不支持可忽略)
    output_schema: 结构化输出 JSON Schema(不支持可忽略)
    """

    @abc.abstractmethod
    def complete(self, *, model: str, system: str, messages: list[Message],
                 tools: list[dict[str, Any]], max_tokens: int, timeout_s: float,
                 caching: bool = False, on_text: OnText | None = None,
                 thinking: str | None = None, effort: str | None = None,
                 output_schema: dict[str, Any] | None = None) -> LLMResponse: ...


_CACHE_CTL = {"cache_control": {"type": "ephemeral"}}


class AnthropicProvider(LLMProvider):
    """基于官方 anthropic SDK。pip install anthropic
    API 文档: https://docs.claude.com/en/api/overview
    """

    def __init__(self, api_key: str) -> None:
        import anthropic  # 延迟导入,未安装时其他 Provider 仍可用
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, model, system, messages, tools, max_tokens, timeout_s,
                 caching=False, on_text=None, thinking=None, effort=None,
                 output_schema=None):
        a = self._anthropic
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=self._system_param(system, caching),
            messages=_to_anthropic_messages(messages, caching),
            tools=self._tools_param(tools, caching),
            timeout=timeout_s,
        )
        if thinking:
            kwargs["thinking"] = {"type": thinking}
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if output_schema:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        if output_config:
            kwargs["output_config"] = output_config
        try:
            if on_text is None:
                resp = self._client.messages.create(**kwargs)
            else:
                with self._client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        on_text(text)
                    resp = stream.get_final_message()
        except (a.RateLimitError, a.InternalServerError,
                a.APIConnectionError, a.APITimeoutError) as e:
            raise TransientLLMError(repr(e)) from e
        except a.APIStatusError as e:
            raise PermanentLLMError(repr(e)) from e

        msg = Message(role=Role.ASSISTANT)
        for block in resp.content:
            if block.type == "text":
                msg.content += block.text
            elif block.type == "tool_use":
                msg.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
            elif block.type == "thinking":
                msg.thinking_blocks.append({
                    "type": "thinking", "thinking": block.thinking,
                    "signature": block.signature})
            elif block.type == "redacted_thinking":
                msg.thinking_blocks.append({
                    "type": "redacted_thinking", "data": block.data})
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )
        return LLMResponse(msg, usage, resp.stop_reason or "end_turn")

    def _system_param(self, system: str, caching: bool) -> Any:
        if not system:
            return self._anthropic.NOT_GIVEN
        if not caching:
            return system
        return [{"type": "text", "text": system, **_CACHE_CTL}]

    def _tools_param(self, tools: list[dict[str, Any]], caching: bool) -> Any:
        if not tools:
            return self._anthropic.NOT_GIVEN
        if not caching:
            return tools
        # 工具列表稳定不变,在最后一个工具上打断点即可缓存整个 tools 段
        return [*tools[:-1], {**tools[-1], **_CACHE_CTL}]


def _to_anthropic_messages(messages: list[Message],
                           caching: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.ASSISTANT:
            # thinking 块必须原样领先回传(同模型续跑的 API 要求)
            content: list[dict[str, Any]] = list(m.thinking_blocks)
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id,
                                "name": tc.name, "input": tc.arguments})
            out.append({"role": "assistant", "content": content})
        else:
            if m.tool_results:
                content = [
                    {"type": "tool_result", "tool_use_id": r.tool_call_id,
                     "content": r.content, "is_error": r.is_error}
                    for r in m.tool_results
                ]
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user",
                            "content": [{"type": "text", "text": m.content}]})
    if caching and out and out[-1]["content"]:
        # 对最新消息打增量断点:下一轮请求可直接命中到本轮为止的全部历史
        # (断点不能落在 thinking 块上)
        last_content = out[-1]["content"]
        if last_content[-1].get("type") not in ("thinking", "redacted_thinking"):
            last_content[-1] = {**last_content[-1], **_CACHE_CTL}
    return out


class MockProvider(LLMProvider):
    """测试用:按预设脚本回放响应,可注入失败次数验证重试逻辑。"""

    def __init__(self, script: list[LLMResponse], fail_first_n: int = 0,
                 fail_forever: bool = False) -> None:
        self.script = list(script)
        self.fail_first_n = fail_first_n
        self.fail_forever = fail_forever
        self.calls = 0

    def complete(self, **_: Any) -> LLMResponse:
        self.calls += 1
        if self.fail_forever or self.calls <= self.fail_first_n:
            raise TransientLLMError("mock transient failure")
        if not self.script:
            raise PermanentLLMError("mock script exhausted")
        return self.script.pop(0)


class ResilientLLMClient:
    """组合 Provider + 重试 + 熔断 + 缓存 + 可观测性。"""

    def __init__(self, provider: LLMProvider, config: AgentConfig,
                 tracer: Tracer, metrics: Metrics) -> None:
        self.provider = provider
        self.config = config
        self.tracer = tracer
        self.metrics = metrics
        self.breaker = CircuitBreaker(config.breaker)

    def complete(self, *, trace_id: str, system: str, messages: list[Message],
                 tools: list[dict[str, Any]], on_text: OnText | None = None,
                 output_schema: dict[str, Any] | None = None) -> LLMResponse:
        retry: RetryConfig = self.config.retry
        last_err: Exception | None = None

        for attempt in range(1, retry.max_attempts + 1):
            self.breaker.before_call()
            with self.tracer.span("llm.call", trace_id, attempt=attempt,
                                  model=self.config.model) as ctx:
                try:
                    resp = self.provider.complete(
                        model=self.config.model,
                        system=system,
                        messages=messages,
                        tools=tools,
                        max_tokens=self.config.max_tokens_per_call,
                        timeout_s=retry.timeout_s,
                        caching=self.config.enable_prompt_caching,
                        on_text=on_text,
                        thinking=self.config.thinking,
                        effort=self.config.effort,
                        output_schema=output_schema,
                    )
                except TransientLLMError as e:
                    last_err = e
                    self.breaker.on_failure()
                    self.metrics.inc("llm.transient_errors")
                    ctx["retryable"] = True
                    if attempt == retry.max_attempts:
                        break
                    delay = min(retry.base_delay_s * 2 ** (attempt - 1), retry.max_delay_s)
                    delay *= 1 + random.uniform(-retry.jitter, retry.jitter)
                    time.sleep(max(delay, 0.0))
                    continue
                # PermanentLLMError / CircuitOpen 由 span 记录后直接上抛
                self.breaker.on_success()
                self.metrics.inc("llm.calls")
                self.metrics.inc("tokens.input", resp.usage.input_tokens)
                self.metrics.inc("tokens.output", resp.usage.output_tokens)
                self.metrics.inc("tokens.cache_write", resp.usage.cache_write_tokens)
                self.metrics.inc("tokens.cache_read", resp.usage.cache_read_tokens)
                ctx["input_tokens"] = resp.usage.input_tokens
                ctx["output_tokens"] = resp.usage.output_tokens
                ctx["cache_read_tokens"] = resp.usage.cache_read_tokens
                ctx["stop_reason"] = resp.stop_reason
                return resp

        raise TransientLLMError(
            f"重试 {retry.max_attempts} 次后仍失败: {last_err!r}"
        )

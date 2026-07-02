"""OpenAI 兼容 Provider — 覆盖 OpenAI / DeepSeek / Qwen / vLLM 等 OpenAI 格式端点。

pip install openai

用法:
    provider = OpenAICompatProvider(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",   # 不传则用 OpenAI 官方端点
    )
    agent = Agent(provider, tools, AgentConfig(model="deepseek-chat"))

注意:
- 换厂商后 BudgetConfig 的四个单价必须按该厂商定价重配,否则成本护栏失真
- caching 参数被忽略:OpenAI 系接口的提示词缓存是服务端自动的,无断点概念;
  命中的缓存 token 记入 cache_read_tokens(把 price_per_mtok_cache_read
  配成厂商的"缓存命中"价即可正确计费)
- 语义映射:finish_reason stop→end_turn / tool_calls→tool_use / length→max_tokens
"""
from __future__ import annotations

import json
from typing import Any

from .llm import LLMProvider, PermanentLLMError, TransientLLMError
from .types import LLMResponse, Message, Role, ToolCall, Usage

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _map_finish_reason(reason: str | None) -> str:
    return _FINISH_REASON_MAP.get(reason or "stop", "end_turn")


def _parse_arguments(raw: str) -> dict[str, Any]:
    """模型偶发输出非法 JSON 参数;解析失败返回空 dict,
    由 ToolRegistry 参数校验产出 error result 回喂模型自纠。"""
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == Role.ASSISTANT:
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments,
                                                          ensure_ascii=False)}}
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.tool_results:
            # OpenAI 格式要求每个工具结果单独一条 role=tool 消息
            for r in m.tool_results:
                content = f"[tool error] {r.content}" if r.is_error else r.content
                out.append({"role": "tool", "tool_call_id": r.tool_call_id,
                            "content": content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"],
                      "parameters": t["input_schema"]}}
        for t in tools
    ]


def _usage_from(u: Any) -> Usage:
    if u is None:
        return Usage()
    prompt = getattr(u, "prompt_tokens", 0) or 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    return Usage(
        input_tokens=max(prompt - cached, 0),  # prompt_tokens 含缓存命中,拆开计价
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_tokens=cached,
    )


class OpenAICompatProvider(LLMProvider):
    """基于官方 openai SDK,base_url 指向任意 OpenAI 兼容端点。

    include_stream_usage: 流式请求时附带 stream_options={"include_usage": True}
      以拿到 token 统计(OpenAI/DeepSeek/Qwen/较新 vLLM 均支持);
      个别旧兼容端点会拒绝该参数,置 False 关闭(代价是流式模式下 usage 为 0)。
    """

    def __init__(self, api_key: str, base_url: str | None = None,
                 include_stream_usage: bool = True) -> None:
        import openai  # 延迟导入,未安装时其他 Provider 仍可用
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._include_stream_usage = include_stream_usage

    def complete(self, *, model, system, messages, tools, max_tokens, timeout_s,
                 caching=False, on_text=None, thinking=None, effort=None,
                 output_schema=None):
        # caching: OpenAI 系缓存为服务端自动,无需断点
        # thinking/effort: 各家推理参数不统一(reasoning_effort/enable_thinking...),忽略
        del caching, thinking, effort
        o = self._openai
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=_to_openai_messages(system, messages),
            timeout=timeout_s,
        )
        oa_tools = _to_openai_tools(tools)
        if oa_tools:
            kwargs["tools"] = oa_tools
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": output_schema},
            }
        try:
            if on_text is None:
                return self._from_response(
                    self._client.chat.completions.create(**kwargs))
            return self._stream(kwargs, on_text)
        except (o.RateLimitError, o.InternalServerError,
                o.APIConnectionError, o.APITimeoutError) as e:
            raise TransientLLMError(repr(e)) from e
        except o.APIStatusError as e:
            raise PermanentLLMError(repr(e)) from e

    def _from_response(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = Message(role=Role.ASSISTANT, content=choice.message.content or "")
        for tc in choice.message.tool_calls or []:
            msg.tool_calls.append(ToolCall(tc.id, tc.function.name,
                                           _parse_arguments(tc.function.arguments)))
        return LLMResponse(msg, _usage_from(resp.usage),
                           _map_finish_reason(choice.finish_reason))

    def _stream(self, kwargs: dict[str, Any], on_text: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_accum: dict[int, dict[str, str]] = {}  # index → {id, name, args}
        finish: str | None = None
        usage = Usage()
        if self._include_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        stream = self._client.chat.completions.create(**kwargs, stream=True)
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = _usage_from(chunk.usage)
            if not chunk.choices:
                continue  # 末尾的 usage-only chunk
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None and delta.content:
                text_parts.append(delta.content)
                on_text(delta.content)
            for tc in (delta.tool_calls or []) if delta is not None else []:
                slot = tool_accum.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
        msg = Message(role=Role.ASSISTANT, content="".join(text_parts))
        for i in sorted(tool_accum):
            s = tool_accum[i]
            msg.tool_calls.append(ToolCall(s["id"], s["name"], _parse_arguments(s["args"])))
        return LLMResponse(msg, usage, _map_finish_reason(finish))

"""工具系统 — 白名单注册 + 权限分级 + 参数校验 + 人工确认钩子。

安全模型:
- 只有显式注册的工具才可被调用(白名单)
- 每个工具声明权限级别 READ / WRITE / EXECUTE
- EXECUTE 级工具默认需要 approval_hook 放行(人在回路)
- 调用前按 JSON Schema 做轻量参数校验,拒绝畸形输入
- 工具异常被捕获并作为 error result 返回给模型,不会击穿主循环
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Permission, ToolCall, ToolResult

try:  # 装了 jsonschema 就用完整校验,否则回退到轻量自研
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover
    _jsonschema = None

ApprovalHook = Callable[[ToolCall, Permission], bool]
ToolFn = Callable[..., str]


class ToolDenied(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn
    permission: Permission = Permission.READ
    max_result_chars: int = 20_000  # 防止超大输出撑爆上下文/预算
    # True 时以 tool_call_id 关键字参数把调用 id 传给 fn。
    # 崩溃重放窗口内同一调用会携带同一 id,工具可据此实现幂等去重。
    pass_call_id: bool = False


def _validate(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """参数校验:白名单键检查 + jsonschema(装了的话)/ 轻量自研回退。"""
    props = schema.get("properties", {})
    for key in args:  # 未声明的参数一律拒绝(比 JSON Schema 默认更严格)
        if props and key not in props:
            return f"未知参数: {key}"
    if _jsonschema is not None:
        try:
            _jsonschema.validate(instance=args, schema=schema)
            return None
        except _jsonschema.ValidationError as e:
            if e.validator == "required":
                return f"缺少必填参数: {e.message}"
            if e.validator == "type":
                path = ".".join(str(p) for p in e.absolute_path) or "?"
                return f"参数 {path} 类型错误: {e.message}"
            return e.message
    # 轻量回退:必填项 + 基础类型
    for req in schema.get("required", []):
        if req not in args:
            return f"缺少必填参数: {req}"
    type_map = {"string": str, "integer": int, "number": (int, float),
                "boolean": bool, "array": list, "object": dict}
    for key, val in args.items():
        expected = type_map.get(props.get(key, {}).get("type", ""))
        if expected and not isinstance(val, expected):
            return f"参数 {key} 类型错误,期望 {props[key]['type']}"
    return None


@dataclass
class ToolRegistry:
    require_approval_for_execute: bool = True
    approval_hook: ApprovalHook | None = None
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._tools[spec.name] = spec

    def tool(self, *, name: str, description: str, input_schema: dict[str, Any],
             permission: Permission = Permission.READ,
             pass_call_id: bool = False) -> Callable[[ToolFn], ToolFn]:
        """装饰器注册方式。"""
        def deco(fn: ToolFn) -> ToolFn:
            self.register(ToolSpec(name, description, input_schema, fn, permission,
                                   pass_call_id=pass_call_id))
            return fn
        return deco

    def specs_for_llm(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def needs_approval(self, call: ToolCall) -> bool:
        """该调用是否需要人工审批(EXECUTE 级且开启了审批要求)。"""
        spec = self._tools.get(call.name)
        return (spec is not None
                and spec.permission == Permission.EXECUTE
                and self.require_approval_for_execute)

    def execute(self, call: ToolCall, approved: bool | None = None) -> ToolResult:
        """执行工具调用。

        approved: 外部(如异步审批流)已做出的审批决定。
          True  → 跳过 approval_hook 直接放行
          False → 直接返回拒绝结果
          None  → 走本地逻辑:有 hook 问 hook,无 hook 拒绝
        """
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(call.id, f"工具 {call.name} 不存在或未授权", is_error=True)

        err = _validate(call.arguments, spec.input_schema)
        if err:
            return ToolResult(call.id, f"参数校验失败: {err}", is_error=True)

        if self.needs_approval(call) and approved is not True:
            if approved is False:
                return ToolResult(
                    call.id, f"工具 {call.name} 已被人工拒绝执行", is_error=True)
            hook_ok = bool(self.approval_hook and self.approval_hook(call, spec.permission))
            if not hook_ok:
                return ToolResult(
                    call.id,
                    f"工具 {call.name} 为 EXECUTE 级,未获人工批准,已拒绝执行",
                    is_error=True,
                )

        try:
            if spec.pass_call_id:
                result = spec.fn(**call.arguments, tool_call_id=call.id)
            else:
                result = spec.fn(**call.arguments)
        except Exception as e:  # 工具异常不击穿主循环
            return ToolResult(call.id, f"工具执行异常: {e!r}", is_error=True)

        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if len(text) > spec.max_result_chars:
            text = text[: spec.max_result_chars] + "\n...[结果过长已截断]"
        return ToolResult(call.id, text)

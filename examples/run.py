"""示例:注册工具并运行一个任务(含流式输出与人工审批)。

运行前: export ANTHROPIC_API_KEY=sk-...
        pip install anthropic
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "..")  # 示例直接引用上级目录的包

from agent import Agent, AgentConfig, AnthropicProvider, Permission, ToolRegistry

tools = ToolRegistry(
    require_approval_for_execute=True,
    # 人在回路(同步审批):EXECUTE 级工具走这里确认。
    # 生产建议改用异步审批:AgentConfig(approval_mode="pause") + 不传 hook,
    # 任务会落盘为 AWAITING_APPROVAL,审批系统回调后
    # agent.resume(task_id, approvals={call_id: True}) 续跑。
    approval_hook=lambda call, perm: input(
        f"[审批] 允许执行 {call.name}({json.dumps(call.arguments, ensure_ascii=False)})? [y/N] "
    ).strip().lower() == "y",
)


@tools.tool(
    name="get_weather",
    description="查询城市当前天气",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"],
    },
    permission=Permission.READ,
)
def get_weather(city: str) -> str:
    # 演示用假数据;生产接真实 API
    return json.dumps({"city": city, "temp_c": 24, "condition": "sunny"})


@tools.tool(
    name="send_report",
    description="向团队发送报告(有外部副作用)",
    input_schema={
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
    permission=Permission.EXECUTE,  # 需人工批准
)
def send_report(content: str) -> str:
    return f"报告已发送 ({len(content)} 字符)"


if __name__ == "__main__":
    config = AgentConfig()
    agent = Agent(AnthropicProvider(AgentConfig.api_key()), tools, config)

    state = agent.run(
        "查一下北京和上海的天气,对比后给团队发一份简短报告",
        on_text=lambda t: print(t, end="", flush=True),  # 模型文本流式打印
    )

    print("\n========== 任务结束 ==========")
    print("status :", state.status.value)
    print("answer :", state.messages[-1].content)
    print("tokens :", state.usage.input_tokens + state.usage.output_tokens)
    print("cache  :", f"read={state.usage.cache_read_tokens}",
          f"write={state.usage.cache_write_tokens}")
    print("cost   : $", state.cost_usd)
    print("metrics:", json.dumps(agent.metrics.snapshot(), ensure_ascii=False, indent=2))

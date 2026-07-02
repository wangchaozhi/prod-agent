"""CLI 入口: python -m agent "任务目标" [选项]

示例:
    python -m agent "总结一下量子计算的现状"
    python -m agent --provider openai --base-url https://api.deepseek.com \\
        --model deepseek-chat "写一首诗"
    python -m agent --resume <task_id>
    python -m agent --resume <task_id> --approvals '{"toolu_xx": true}'

密钥从环境变量读取:ANTHROPIC_API_KEY / OPENAI_API_KEY。
CLI 模式不注册任何工具(纯对话);需要工具请用库方式集成。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import (Agent, AgentConfig, AnthropicProvider, OpenAICompatProvider,
               TaskStatus, ToolRegistry)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent", description="prod-agent CLI")
    ap.add_argument("goal", nargs="?", help="任务目标")
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    default="anthropic")
    ap.add_argument("--model", help="模型 id(默认取 AgentConfig 默认值)")
    ap.add_argument("--base-url", help="OpenAI 兼容端点地址")
    ap.add_argument("--db", default="agent_checkpoints.db", help="检查点数据库路径")
    ap.add_argument("--resume", metavar="TASK_ID", help="从检查点续跑")
    ap.add_argument("--approvals", help='审批决定 JSON,如 {"call_id": true}')
    ap.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = ap.parse_args(argv)

    if not args.goal and not args.resume:
        ap.error("需要给出任务目标,或 --resume TASK_ID")

    cfg_kw: dict = {"checkpoint_db": args.db}
    if args.model:
        cfg_kw["model"] = args.model
    config = AgentConfig(**cfg_kw)

    if args.provider == "anthropic":
        provider: AnthropicProvider | OpenAICompatProvider = (
            AnthropicProvider(AgentConfig.api_key()))
    else:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            sys.exit("OPENAI_API_KEY 未设置")
        provider = OpenAICompatProvider(key, base_url=args.base_url)

    agent = Agent(provider, ToolRegistry(), config)
    on_text = None if args.no_stream else (
        lambda t: print(t, end="", flush=True))

    if args.resume:
        approvals = json.loads(args.approvals) if args.approvals else None
        state = agent.resume(args.resume, approvals=approvals, on_text=on_text)
    else:
        state = agent.run(args.goal, on_text=on_text)

    print(f"\n[task_id] {state.task_id}")
    print(f"[status]  {state.status.value}")
    print(f"[cost]    ${state.cost_usd}")
    if state.error:
        print(f"[error]   {state.error}")
    return 0 if state.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())

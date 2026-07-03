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
import dataclasses
import json
import os
import sys

from . import (Agent, AgentConfig, AnthropicProvider, OpenAICompatProvider,
               TaskStatus, ToolRegistry)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from .server import main as server_main
        return server_main(argv[1:])
    if argv and argv[0] == "worker":
        return worker_main(argv[1:])

    ap = argparse.ArgumentParser(prog="agent", description="prod-agent CLI")
    ap.add_argument("goal", nargs="?", help="任务目标")
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    default="anthropic")
    ap.add_argument("--model", help="模型 id(默认取 AgentConfig 默认值)")
    ap.add_argument("--base-url", help="OpenAI 兼容端点地址")
    ap.add_argument("--db", help="检查点数据库路径")
    ap.add_argument("--resume", metavar="TASK_ID", help="从检查点续跑")
    ap.add_argument("--approvals", help='审批决定 JSON,如 {"call_id": true}')
    ap.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = ap.parse_args(argv)

    if not args.goal and not args.resume:
        ap.error("需要给出任务目标,或 --resume TASK_ID")

    base_config = AgentConfig.from_env()
    config = dataclasses.replace(
        base_config,
        checkpoint_db=args.db or base_config.checkpoint_db,
        model=args.model or base_config.model,
    )

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


def worker_main(argv: list[str] | None = None) -> int:
    from .server import build_agent
    from .worker import Worker

    ap = argparse.ArgumentParser(prog="agent worker", description="prod-agent worker")
    ap.add_argument("--provider", choices=["demo", "anthropic", "openai"],
                    default=os.environ.get("PROD_AGENT_PROVIDER", "demo"))
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--db")
    ap.add_argument("--business-db", default=os.environ.get(
        "PROD_AGENT_BUSINESS_DB", "demo_business.db"))
    ap.add_argument("--no-demo-tools", action="store_true")
    ap.add_argument("--poll-interval", type=float, default=float(os.environ.get(
        "PROD_AGENT_WORKER_POLL_INTERVAL_S", "5")))
    ap.add_argument("--purge-days", type=float, default=float(os.environ.get(
        "PROD_AGENT_PURGE_DAYS", "0")))
    args = ap.parse_args(argv)

    agent = build_agent(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        checkpoint_db=args.db,
        business_db=args.business_db,
        demo_tools=not args.no_demo_tools,
    )
    print("prod-agent worker started")
    Worker(
        agent,
        poll_interval_s=args.poll_interval,
        purge_older_than_days=args.purge_days,
    ).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

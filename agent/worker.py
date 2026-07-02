"""轻量 Worker — 轮询检查点库,认领崩溃遗留的 RUNNING 任务续跑。

配合任务租约,多个 Worker 进程可以安全并存:同一任务同一时刻只会被
一个执行者认领(其余拿不到租约自动跳过),正在被其他进程执行的任务
因租约未过期同样不会被抢走。

等待审批(AWAITING_APPROVAL)的任务不会被自动认领——它们需要人工
approvals,应由审批系统回调 agent.resume(task_id, approvals=...)。

用法:
    worker = Worker(agent, poll_interval_s=5)
    worker.run_forever()      # 阻塞;另一线程调用 worker.stop() 退出
    # 或嵌入自己的调度循环:
    n = worker.run_once()     # 认领一批,返回处理数
"""
from __future__ import annotations

import threading

from .core import Agent
from .memory import LeaseConflict
from .types import TaskStatus


class Worker:
    def __init__(self, agent: Agent, poll_interval_s: float = 5.0) -> None:
        self.agent = agent
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()

    def run_once(self) -> int:
        """认领并续跑一批遗留任务,返回本轮处理的任务数。"""
        handled = 0
        for task_id in self.agent.store.list_resumable(
                statuses=(TaskStatus.RUNNING,)):
            try:
                self.agent.resume(task_id)
                handled += 1
            except LeaseConflict:
                continue  # 其他执行者正在跑或刚认领
        return handled

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.poll_interval_s)

    def stop(self) -> None:
        self._stop.set()

"""上下文窗口管理 — 长任务的消息历史压缩。

state.messages 会随迭代无限增长,长任务迟早撞上下文上限
(届时 API 返回 400 → PermanentLLMError → 任务 FAILED)。

两种压缩模式(均保证 tool_use/tool_result 配对结构合法):

1. 占位符模式(默认,零成本):旧消息的工具结果正文原地替换为占位符,
   仍超阈值再把旧文本截断到前 200 字符。消息条数与配对结构完全不变。
2. 摘要折叠模式(传入 summarize_fn 启用):旧消息区间整体折叠为一条
   摘要消息,信息保真度更高。summarize_fn 可以接 LLM 做真正的摘要,
   折叠边界会自动对齐,保证保留区不会以孤儿 tool_result 开头。

token 计数默认用字符数 ~4:1 粗估(中文实际比率更低,阈值偏保守失真;
生产可注入 token_counter,接 API 的 count_tokens 精确计量)。
压缩后的消息随检查点落盘,保证 LLM 看到的内容与恢复后的状态一致。
"""
from __future__ import annotations

from typing import Callable

from .config import ContextConfig
from .types import Message, Role

COMPACTED_PLACEHOLDER = "[已压缩:历史工具结果已省略]"
SUMMARY_PREFIX = "[历史对话已压缩为摘要]"
_TEXT_KEEP_CHARS = 200

TokenCounter = Callable[[list[Message]], int]
SummarizeFn = Callable[[list[Message]], str]


def _message_chars(m: Message) -> int:
    n = len(m.content)
    for tc in m.tool_calls:
        n += len(tc.name) + len(str(tc.arguments))
    for tr in m.tool_results:
        n += len(tr.content)
    for tb in m.thinking_blocks:
        n += len(str(tb))
    return n


def estimate_tokens(messages: list[Message]) -> int:
    """粗略估算消息列表的 token 数(每条消息附加 8 token 结构开销)。"""
    return sum(_message_chars(m) // 4 + 8 for m in messages)


class ContextCompactor:
    def __init__(self, cfg: ContextConfig,
                 token_counter: TokenCounter | None = None,
                 summarize_fn: SummarizeFn | None = None) -> None:
        self.cfg = cfg
        self.token_counter = token_counter or estimate_tokens
        self.summarize_fn = summarize_fn

    def compact_if_needed(self, messages: list[Message]) -> bool:
        """超过阈值时原地压缩旧消息,返回是否发生了压缩。"""
        if self.token_counter(messages) <= self.cfg.max_context_tokens:
            return False
        end = max(1, len(messages) - self.cfg.keep_recent_messages)
        if self.summarize_fn is not None:
            return self._collapse_with_summary(messages, end)
        return self._placeholder_compact(messages, end)

    def _placeholder_compact(self, messages: list[Message], end: int) -> bool:
        old = messages[1:end]
        compacted = False
        for m in old:  # 第一遍:压缩工具结果(通常是最大的 token 来源)
            for tr in m.tool_results:
                if len(tr.content) > len(COMPACTED_PLACEHOLDER):
                    tr.content = COMPACTED_PLACEHOLDER
                    compacted = True
        if self.token_counter(messages) > self.cfg.max_context_tokens:
            for m in old:  # 第二遍:截断文本正文
                if len(m.content) > _TEXT_KEEP_CHARS:
                    m.content = m.content[:_TEXT_KEEP_CHARS] + "…[已截断]"
                    compacted = True
        return compacted

    def _collapse_with_summary(self, messages: list[Message], end: int) -> bool:
        # 保留区不能以 tool_results 开头(其配对的 tool_use 会被折叠掉),
        # 把边界前移,让 tool_use/tool_result 成对留在保留区
        while 1 < end < len(messages) and messages[end].tool_results:
            end -= 1
        if end <= 1:
            return False  # 无可折叠区间
        old = messages[1:end]
        summary = self.summarize_fn(old)  # type: ignore[misc]
        messages[1:end] = [Message(role=Role.USER,
                                   content=f"{SUMMARY_PREFIX}\n{summary}")]
        return True

"""可观测性 — JSON 结构化日志 + trace/span + 进程内指标。

设计要点:
- 每个任务一个 trace_id,每次 LLM 调用 / 工具调用一个 span,日志可按 trace_id 聚合
- 日志输出 JSON,可直接被 ELK / Loki / CloudWatch 采集
- 指标在进程内累计,通过 snapshot() 导出,可对接 Prometheus exporter
- 自动脱敏:不打印疑似密钥字段
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Iterator

_SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: ("***REDACTED***" if k.lower() in _SENSITIVE_KEYS else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(_redact(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        stream = sys.stdout
        try:
            # Windows GBK 控制台下中文日志会乱码,包一层 UTF-8(不影响 print)
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="backslashreplace",
                                      line_buffering=True)
        except (AttributeError, io.UnsupportedOperation, ValueError):
            pass  # stdout 被重定向/捕获(如 pytest)时保持原样
        h = logging.StreamHandler(stream)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class Metrics:
    """线程安全的进程内指标。生产环境可替换为 prometheus_client。

    直方图采用流式聚合(count/total/min/max),内存 O(1),
    长期运行的进程不会因样本累积而泄漏。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, dict[str, float]] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            h = self._histograms.setdefault(
                name, {"count": 0.0, "total": 0.0, "min": value, "max": value})
            h["count"] += 1
            h["total"] += value
            h["min"] = min(h["min"], value)
            h["max"] = max(h["max"], value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hist = {
                k: {
                    "count": int(h["count"]),
                    "avg": round(h["total"] / h["count"], 4),
                    "min": round(h["min"], 4),
                    "max": round(h["max"], 4),
                }
                for k, h in self._histograms.items()
            }
            return {"counters": dict(self._counters), "histograms": hist}


class Tracer:
    """轻量 trace/span。生产环境可替换为 OpenTelemetry,接口保持一致。"""

    def __init__(self, logger: logging.Logger, metrics: Metrics) -> None:
        self.logger = logger
        self.metrics = metrics

    @contextlib.contextmanager
    def span(self, name: str, trace_id: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        span_id = uuid.uuid4().hex[:8]
        ctx: dict[str, Any] = {"trace_id": trace_id, "span_id": span_id, "span": name, **attrs}
        start = time.monotonic()
        self.logger.info(f"{name}.start", extra={"ctx": ctx})
        try:
            yield ctx
        except Exception as e:
            ctx["error"] = repr(e)
            self.metrics.inc(f"{name}.errors")
            self.logger.error(f"{name}.error", extra={"ctx": ctx}, exc_info=True)
            raise
        finally:
            dur = time.monotonic() - start
            ctx["duration_s"] = round(dur, 3)
            self.metrics.observe(f"{name}.duration_s", dur)
            self.logger.info(f"{name}.end", extra={"ctx": ctx})

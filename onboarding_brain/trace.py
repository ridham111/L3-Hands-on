"""Structured tracing & logging."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("onboarding_brain")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, get_settings().log_level, logging.INFO))
    logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for k in ("trace_id", "event"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


logger = _configure_logger()


def new_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex[:16]}"


def fingerprint(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_event(event: str, trace_id: str) -> None:
    logger.info(event, extra={"event": event, "trace_id": trace_id})


@contextmanager
def timed() -> Iterator[dict[str, int]]:
    start = time.perf_counter()
    holder: dict[str, int] = {}
    try:
        yield holder
    finally:
        holder["ms"] = int((time.perf_counter() - start) * 1000)


def append_trace(entry: dict[str, Any]) -> None:
    path = Path(get_settings().trace_file)
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # never let tracing crash the agent
        logger.warning("trace_write_failed: %s", exc, extra={"event": "trace_write_failed"})

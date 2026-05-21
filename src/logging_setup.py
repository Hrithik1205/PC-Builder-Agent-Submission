"""Structured logging setup.

We use `structlog` to emit one JSON object per log record. Each agent run
gets its own jsonl file under `TRACE_DIR/<run_id>.jsonl`. Console output
stays human-readable so the developer can watch the agent reason in real
time, while the jsonl trace is the machine-readable record we render into
the agent run report.
"""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from src.config import get_settings


_current_run_id: Optional[str] = None
_current_trace_path: Optional[Path] = None


def new_run_id() -> str:
    """Generate a fresh, sortable run identifier."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def configure_logging(run_id: Optional[str] = None, level: int = logging.INFO) -> str:
    """Configure structlog + stdlib logging.

    Returns the run_id so callers can correlate logs with a specific run.
    Safe to call multiple times; only the first call wires up handlers,
    subsequent calls just rotate the run_id and jsonl sink.
    """
    global _current_run_id, _current_trace_path

    settings = get_settings()
    run_id = run_id or new_run_id()
    _current_run_id = run_id
    trace_path = settings.trace_dir / f"{run_id}.jsonl"
    _current_trace_path = trace_path

    # Reset any previously installed handlers so re-configuration is clean
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.setLevel(level)

    # JSON file sink (the trace)
    file_handler = logging.FileHandler(trace_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # Pretty console sink
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_run_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _route_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return run_id


def _add_run_id(_logger, _name, event_dict):
    if _current_run_id:
        event_dict.setdefault("run_id", _current_run_id)
    return event_dict


def _route_renderer(logger, name, event_dict):
    """Render JSON for the file handler, plain-ish for the console.

    structlog calls renderers once and we pass the same string to all
    stdlib handlers. To keep both human-readable console output and a
    machine-parseable jsonl file, we let the JSON form be the single
    output and accept that the console is JSON too (developers can pipe
    it through `jq` if they want pretty output). Simpler and faithful.
    """
    return structlog.processors.JSONRenderer()(logger, name, event_dict)


def get_logger(name: str = "pc_builder"):
    """Get a structlog bound logger for a module."""
    return structlog.get_logger(name)


def current_run_id() -> Optional[str]:
    return _current_run_id


def current_trace_path() -> Optional[Path]:
    return _current_trace_path

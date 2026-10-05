"""Structured logging shared by all dfo-ai services.

Chooses single-line JSON logs (Cloud Run) or human-readable text (local dev)
depending on environment, and routes uvicorn's own handlers through the same
formatter so a startup traceback lands as one Cloud Logging entry instead of
fragmenting across lines.

Selection is automatic: `K_SERVICE` (always set on Cloud Run) selects JSON.
Locally, set `LOG_FORMAT=json` to force JSON or `LOG_FORMAT=text` to force
text even on Cloud Run.
"""

from __future__ import annotations

import json
import logging
import os

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# https://cloud.google.com/logging/docs/reference/v2/rest/v2/LogEntry#LogSeverity
_LEVEL_TO_SEVERITY: dict[str, str] = {
    "CRITICAL": "CRITICAL",
    "FATAL": "CRITICAL",
    "ERROR": "ERROR",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
    "NOTSET": "DEFAULT",
}

_configured = False


class _CloudRunFormatter(logging.Formatter):
    """JSON formatter that keeps multi-line tracebacks in a single log entry.

    Cloud Logging fragments multi-line messages across separate entries
    unless the traceback is embedded in one `message` field; this formatter
    does that while still exposing `stack_trace` as a discrete field.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        stack_trace: str | None = None
        if record.exc_info and record.exc_info[0]:
            stack_trace = self.formatException(record.exc_info)
            message = f"{message}\n{stack_trace}"

        entry: dict[str, object] = {
            "severity": _LEVEL_TO_SEVERITY.get(record.levelname, "DEFAULT"),
            "message": message,
            "logger": record.name,
        }
        if stack_trace is not None:
            entry["stack_trace"] = stack_trace
        return json.dumps(entry)


def _use_json() -> bool:
    override = os.getenv("LOG_FORMAT", "").strip().lower()
    if override == "json":
        return True
    if override == "text":
        return False
    return bool(os.getenv("K_SERVICE"))


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure the root logger (and uvicorn's) for the current process.

    Idempotent -- safe to call multiple times, e.g. across lifespan reloads.
    """
    global _configured  # noqa: PLW0603 -- deliberate idempotent-init flag
    if _configured:
        return
    _configured = True

    handler = logging.StreamHandler()
    handler.setFormatter(_CloudRunFormatter() if _use_json() else logging.Formatter(_TEXT_FORMAT))

    root = logging.root
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers with propagate=False; clear them and
    # re-enable propagation so its records (including startup tracebacks)
    # flow through our formatter instead of uvicorn's multi-line default.
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # Known issue: noisy "Failed to detach context" errors from OTel's async
    # context propagation on Python 3.11+.
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

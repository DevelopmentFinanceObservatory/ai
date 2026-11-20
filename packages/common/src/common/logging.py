"""Structured logging shared by all dfo-ai services.

Chooses single-line JSON logs (Cloud Run) or human-readable text (local dev)
depending on environment, and routes uvicorn's own handlers through the same
formatter so a startup traceback lands as one Cloud Logging entry instead of
fragmenting across lines.

Selection is automatic: `K_SERVICE`, set in every Cloud Run service, or
`CLOUD_RUN_JOB`, set in every Cloud Run job execution, selects JSON. Locally, set `LOG_FORMAT=json` to force JSON or `LOG_FORMAT=text` to force
text even on Cloud Run.

A record logged with `extra={JSON_FIELDS: {...}}` carries those fields as
top-level members of its JSON entry, so Cloud Logging stores each as a
`jsonPayload` field beside `message`. Text output shows the message alone.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping

# The `extra` key whose mapping `_CloudRunFormatter` writes into the entry.
JSON_FIELDS = "json_fields"

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


def _str_keys(value: object) -> object:
    """`value` with every mapping key at any depth, in nested mappings and in
    lists and tuples, written as its `str`, which `json.dumps` needs of a key."""
    if isinstance(value, Mapping):
        return {str(k): _str_keys(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_str_keys(v) for v in value]
    return value


class _CloudRunFormatter(logging.Formatter):
    """JSON formatter that keeps multi-line tracebacks in a single log entry.

    Cloud Logging fragments multi-line messages across separate entries
    unless the traceback is embedded in one `message` field; this formatter
    does that while still exposing `stack_trace` as a discrete field. A
    record's `JSON_FIELDS` mapping is merged in under the formatter's own
    keys, which win a clash. A field key, at any depth, is written as its
    `str`, and a field value JSON has no form for (a datetime, a set) is
    written as its `str` too, so the record still logs. Fields that hold a
    cycle are left out, and the record logs without them.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        stack_trace: str | None = None
        if record.exc_info and record.exc_info[0]:
            stack_trace = self.formatException(record.exc_info)
            message = f"{message}\n{stack_trace}"

        own: dict[str, object] = {
            "severity": _LEVEL_TO_SEVERITY.get(record.levelname, "DEFAULT"),
            "message": message,
            "logger": record.name,
        }
        if stack_trace is not None:
            own["stack_trace"] = stack_trace
        fields = getattr(record, JSON_FIELDS, None)
        if isinstance(fields, Mapping):
            try:
                entry = {str(k): _str_keys(v) for k, v in fields.items()} | own
                return json.dumps(entry, default=str)
            except (RecursionError, ValueError):
                # A cycle makes `_str_keys` recurse without end, or `json.dumps`
                # refuse the circular reference.
                pass
        return json.dumps(own, default=str)


def _use_json() -> bool:
    override = os.getenv("LOG_FORMAT", "").strip().lower()
    if override == "json":
        return True
    if override == "text":
        return False
    return bool(os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"))


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

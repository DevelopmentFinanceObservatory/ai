import json
import logging
import sys

import common.logging as common_logging
import pytest
from common.logging import _CloudRunFormatter, configure_logging


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    """`configure_logging` is idempotent via a module-level flag; reset it
    between tests so each test observes its own call."""
    common_logging._configured = False
    yield
    common_logging._configured = False


def test_configure_logging_sets_root_handler(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    configure_logging()

    assert len(logging.root.handlers) == 1
    assert isinstance(logging.root.handlers[0].formatter, logging.Formatter)


def test_configure_logging_is_idempotent(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    configure_logging()
    first_handler = logging.root.handlers[0]
    configure_logging()

    assert logging.root.handlers[0] is first_handler


def test_json_format_selected_on_cloud_run(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "harness")
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    configure_logging()

    assert isinstance(logging.root.handlers[0].formatter, _CloudRunFormatter)


def test_log_format_override_forces_text_on_cloud_run(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "harness")
    monkeypatch.setenv("LOG_FORMAT", "text")

    configure_logging()

    assert not isinstance(logging.root.handlers[0].formatter, _CloudRunFormatter)


def test_cloud_run_formatter_embeds_traceback():
    formatter = _CloudRunFormatter()
    logger = logging.getLogger("test.cloud_run_formatter")

    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0, "failed", (), sys.exc_info()
        )

    payload = json.loads(formatter.format(record))

    assert payload["severity"] == "ERROR"
    assert "stack_trace" in payload
    assert "ValueError: boom" in payload["stack_trace"]

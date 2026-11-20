import json
import logging
import sys
from datetime import UTC, datetime

import common.logging as common_logging
import pytest
from common.logging import JSON_FIELDS, _CloudRunFormatter, configure_logging


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


def test_cloud_run_formatter_writes_json_fields_beside_its_own_keys():
    started = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
    formatter = _CloudRunFormatter()
    logger = logging.getLogger("test.cloud_run_formatter")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        "studio.turn",
        (),
        None,
        extra={
            JSON_FIELDS: {
                "event": "studio.turn",
                "drawn_specs": ["spec_a"],
                "message": "x",
                "started_at": started,
                "fetched_specs": {"spec_b"},
                ("spec_c", 1): "tuple key",
                "nested": {("spec_d", 2): [{("spec_e", 3): "deep"}]},
            }
        },
    )

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "studio.turn"
    assert payload["drawn_specs"] == ["spec_a"]
    assert payload["started_at"] == str(started)
    assert payload["fetched_specs"] == str({"spec_b"})
    assert payload[str(("spec_c", 1))] == "tuple key"
    assert payload["nested"] == {str(("spec_d", 2)): [{str(("spec_e", 3)): "deep"}]}
    assert payload["message"] == "studio.turn"
    assert payload["severity"] == "INFO"


def _cyclic_dict() -> dict[str, object]:
    value: dict[str, object] = {}
    value["self"] = value
    return value


def _cyclic_list() -> list[object]:
    value: list[object] = []
    value.append(value)
    return value


@pytest.mark.parametrize("cycle", [_cyclic_dict, _cyclic_list], ids=["dict", "list"])
def test_cloud_run_formatter_logs_a_record_whose_fields_hold_a_cycle(cycle):
    """Fields that refer to themselves are left out. The record still logs its
    own keys."""
    logger = logging.getLogger("test.cloud_run_formatter")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        "studio.turn",
        (),
        None,
        extra={JSON_FIELDS: {"event": "studio.turn", "loop": cycle()}},
    )

    payload = json.loads(_CloudRunFormatter().format(record))

    assert payload == {"severity": "INFO", "message": "studio.turn", "logger": logger.name}

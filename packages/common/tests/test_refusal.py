"""`refusal_code` reads the code from either refusal envelope."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from common import refusal as refusal_module
from common.refusal import internal_error_message, log_unexpected_error, refusal_code


@pytest.mark.parametrize(
    ("structured", "code"),
    [
        ({"code": "not_found", "retry_mode": "do_not_retry"}, "not_found"),
        ({"error": "write_refused", "retry": "do_not_retry"}, "write_refused"),
        ({"type": "Column", "children": []}, None),
        ({"code": ""}, None),
        ({}, None),
        (None, None),
    ],
    ids=["code", "presentation-error", "no-code", "empty-code", "empty", "none"],
)
def test_refusal_code_reads_code_or_presentations_error(
    structured: dict[str, Any] | None, code: str | None
) -> None:
    assert refusal_code(structured) == code


def test_an_unexpected_error_is_logged_with_the_trace_id_and_the_reply_text_has_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trace_id = "0af7651916cd43dd8448eb211c80319c"
    monkeypatch.setattr(refusal_module, "current_trace_id", lambda: trace_id)
    logger = logging.getLogger("test_refusal")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("secret")
        except RuntimeError:
            log_unexpected_error(logger, "a tool")

    [record] = [r for r in caplog.records if r.name == logger.name]
    assert f"trace_id={trace_id}" in record.getMessage()
    assert record.exc_info is not None
    reply = internal_error_message(subject="a tool")
    assert reply.startswith("a tool failed on the server")
    assert "trace" not in reply.lower()

"""`refusal_code` reads the code from the refusal envelope."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from common import refusal as refusal_module
from common.refusal import (
    detail_sentence,
    internal_error_message,
    log_unexpected_error,
    refusal_code,
)


@pytest.mark.parametrize(
    ("structured", "code"),
    [
        ({"code": "not_found", "retry_mode": "do_not_retry"}, "not_found"),
        ({"error": "write_refused", "retry": "do_not_retry"}, None),
        ({"type": "Column", "children": []}, None),
        ({"code": ""}, None),
        ({}, None),
        (None, None),
    ],
    ids=["code", "error-field", "no-code", "empty-code", "empty", "none"],
)
def test_refusal_code_reads_the_code_field(
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


@pytest.mark.parametrize(
    ("detail", "sentence"),
    [
        ("the frame was not found", "The frame was not found."),
        (
            "frame_ref names ops_1 without its content_version",
            "frame_ref names ops_1 without its content_version.",
        ),
        ("frame_ref.spec_id must be a valid id.", "frame_ref.spec_id must be a valid id."),
        ("  unknown role  ", "Unknown role."),
        ("Already a sentence!", "Already a sentence!"),
        ("", ""),
    ],
    ids=["fragment", "argument-name", "dotted-argument", "padded", "sentence", "empty"],
)
def test_detail_sentence_capitalises_all_but_an_identifier(detail: str, sentence: str) -> None:
    assert detail_sentence(detail) == sentence

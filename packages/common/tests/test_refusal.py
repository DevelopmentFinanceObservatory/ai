"""`refusal_code` reads the code from either refusal envelope."""

from __future__ import annotations

from typing import Any

import pytest
from common.refusal import refusal_code


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

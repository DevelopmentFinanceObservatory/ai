"""Tests for `harness.gemini_agent`, what a Pydantic AI agent on Gemini shares."""

from __future__ import annotations

import base64

import pytest
from harness.gemini_agent import without_unsigned_thoughts
from pydantic_ai.messages import ModelResponse, ModelResponsePart, TextPart, ThinkingPart

_SIGNATURE = {"thought_signature": base64.b64encode(b"sig").decode()}
_SIGNED = ThinkingPart("weighed the sources", provider_details=_SIGNATURE)
_UNSIGNED = ThinkingPart("thinking aloud")
_ANSWER = TextPart("Accra")
_SIGNED_ANSWER = TextPart("Accra", provider_details=_SIGNATURE)


@pytest.mark.parametrize(
    ("parts", "kept"),
    [
        pytest.param([_SIGNED, _UNSIGNED, _ANSWER], [_SIGNED, _ANSWER], id="signed-and-unsigned"),
        pytest.param([_UNSIGNED, _ANSWER], [_ANSWER], id="unsigned"),
        pytest.param([_UNSIGNED, _SIGNED_ANSWER], [_SIGNED_ANSWER], id="signed-answer"),
        pytest.param([_UNSIGNED], [_UNSIGNED], id="only-unsigned"),
        pytest.param([_SIGNED, _ANSWER], [_SIGNED, _ANSWER], id="none-to-drop"),
    ],
)
def test_the_thought_rule_keeps_the_signed_parts_and_the_answer(
    parts: list[ModelResponsePart], kept: list[ModelResponsePart]
) -> None:
    """Every unsigned thought summary is dropped, a signed thought kept, and a
    response made only of unsigned thoughts kept whole."""
    ruled = without_unsigned_thoughts(ModelResponse(parts=parts))

    assert ruled.parts == kept

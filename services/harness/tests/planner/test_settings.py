from __future__ import annotations

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from planner.settings import (
    AGENT_THINKING_CONFIG,
    AgentSettings,
    SchemaDelivery,
    drop_thought_parts,
)

_FIELDS = ("PLANNER_MODEL_NAME", "PLANNER_SCHEMA_DELIVERY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in _FIELDS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_pin_the_planner_to_one_concrete_model() -> None:
    """A version, not a floating alias -- the point of the pin."""
    assert AgentSettings().planner_model_name == "gemini-3.8-flash"


def test_reads_from_env(monkeypatch) -> None:
    """The model overrides from the environment, which is how one gets scored against another."""
    monkeypatch.setenv("PLANNER_MODEL_NAME", "gemini-3.1-pro-preview")

    assert AgentSettings().planner_model_name == "gemini-3.1-pro-preview"


def test_schema_delivery_is_eager_unless_the_environment_says_compact(monkeypatch) -> None:
    assert AgentSettings().planner_schema_delivery is SchemaDelivery.EAGER

    monkeypatch.setenv("PLANNER_SCHEMA_DELIVERY", "compact")

    assert AgentSettings().planner_schema_delivery is SchemaDelivery.COMPACT


def test_agent_thinking_config_is_low_with_thoughts_on() -> None:
    assert AGENT_THINKING_CONFIG.thinking_level is types.ThinkingLevel.LOW
    assert AGENT_THINKING_CONFIG.include_thoughts is True


def test_drop_thought_parts_removes_only_thought_marked_parts() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="reasoning about the plan", thought=True),
                types.Part(function_call=types.FunctionCall(name="propose_plan", args={})),
            ],
        )
    )

    dropped = drop_thought_parts(None, response)

    assert dropped is not None
    assert [part.thought for part in dropped.content.parts] == [None]
    assert dropped.content.parts[0].function_call.name == "propose_plan"
    # The original response, and the trace already built from it, are
    # untouched -- only the copy handed back loses the thought part.
    assert len(response.content.parts) == 2


def test_drop_thought_parts_keeps_a_signed_thought_part() -> None:
    """A `thought_signature` marks the part Gemini requires on replay (a
    function-call or final text part), so it survives even when `thought` is
    also set."""
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="reasoning about the plan", thought=True),
                types.Part(
                    text="reasoning that carries the signature",
                    thought=True,
                    thought_signature=b"sig",
                ),
                types.Part(function_call=types.FunctionCall(name="propose_plan", args={})),
            ],
        )
    )

    dropped = drop_thought_parts(None, response)

    assert dropped is not None
    assert [part.text for part in dropped.content.parts if part.text] == [
        "reasoning that carries the signature"
    ]
    assert dropped.content.parts[-1].function_call.name == "propose_plan"


def test_drop_thought_parts_returns_none_when_every_part_would_be_dropped() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="reasoning about the plan", thought=True)],
        )
    )

    assert drop_thought_parts(None, response) is None


def test_drop_thought_parts_is_a_noop_with_no_thought_parts() -> None:
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="the answer")])
    )

    assert drop_thought_parts(None, response) is None


def test_drop_thought_parts_is_a_noop_with_no_content() -> None:
    assert drop_thought_parts(None, LlmResponse(content=None)) is None
    empty = LlmResponse(content=types.Content(role="model", parts=[]))
    assert drop_thought_parts(None, empty) is None

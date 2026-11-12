from __future__ import annotations

import pytest
from google.genai import types
from planner.settings import (
    AGENT_THINKING_CONFIG,
    AgentSettings,
)

_FIELDS = ("PLANNER_MODEL_NAME",)


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


def test_agent_thinking_config_is_low_with_thoughts_on() -> None:
    assert AGENT_THINKING_CONFIG.thinking_level is types.ThinkingLevel.LOW
    assert AGENT_THINKING_CONFIG.include_thoughts is True

"""Grep-level content checks for the Edit Card prompt, plus a behavioral test
of `GeminiEdit`'s own handling of a drafted recreate.

The prompt checks are not behavioral tests -- there is no cheap way to
assert a model's output from a prompt alone. These pin the specific
grammar text so a future edit that silently drops it fails loudly, the same
role a snapshot test plays elsewhere.

The `GeminiEdit` test runs the real class against a fake `genai` client that
returns a canned `response.text`, so `_generate`'s JSON parse and
`__call__`'s own handling of the drafted object both run for real.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from google.genai import types
from harness.conversation import ModelCall, current_model_usage
from harness.generative import _EDIT_INSTRUCTION, EDIT_AGENT_NAME, GeminiEdit
from harness.settings import Settings
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# The instruction as the model reads it, with its line wrapping folded.
_EDIT_TEXT = " ".join(_EDIT_INSTRUCTION.split())


def test_edit_instruction_states_the_window_grammar() -> None:
    assert "window is an object of period labels" in _EDIT_TEXT
    assert '{"start": "2000", "end": "2023"}' in _EDIT_TEXT
    assert "A window of labels may leave out either side" in _EDIT_TEXT
    assert '"Since 2000" is {"start": "2000"}' in _EDIT_TEXT


def test_edit_instruction_updates_a_title_that_names_what_changed() -> None:
    assert "A title that names something you change" in _EDIT_TEXT
    assert '"... (2000-2023)"' in _EDIT_TEXT
    assert "open side read from the frame's coverage" in _EDIT_TEXT


def test_edit_instruction_states_the_relative_period_grammar() -> None:
    assert '{"anchor": "latest", "offset": N}' in _EDIT_TEXT
    assert '"The last three years" is {"start": {"anchor": "latest",' in _EDIT_TEXT
    assert "When either side is relative, state both sides" in _EDIT_TEXT
    assert (
        '"Compare with the previous year" is {"basis": [{"anchor": "latest", "offset": -1}'
        in _EDIT_TEXT
    )
    # Years a person names stay labels, in a window and in a change alike.
    assert '"2015 to 2020" is {"start": "2015", "end": "2020"}' in _EDIT_TEXT
    assert '"Compare 2015 with 2020" is {"basis": ["2015", "2020"]}' in _EDIT_TEXT
    assert "Its title names no year and no year range" in _EDIT_TEXT


class _FakeModels:
    def __init__(self, text: str, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        self._text = text
        self._usage = usage
        self.contents: list[Any] = []
        self.configs: list[Any] = []

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.contents.append(kwargs["contents"])
        self.configs.append(kwargs["config"])
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text=self._text)]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=self._usage,
            model_version=kwargs["model"],
        )


class _FakeAio:
    def __init__(self, text: str, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        self.models = _FakeModels(text, usage)


class _FakeClient:
    """Stands in for `google.genai.Client`: only the `.aio.models.
    generate_content` path `_ModelCall._generate` calls is faked."""

    def __init__(
        self, text: str, usage: types.GenerateContentResponseUsageMetadata | None = None
    ) -> None:
        self.aio = _FakeAio(text, usage)


_LATEST = {"anchor": "latest", "offset": 0}


@pytest.mark.parametrize(
    ("field", "drafted"),
    [
        (
            "recreate",
            {
                "type": "table",
                "title": "Aid",
                "select": {"donor": "country/FRA"},
                "order": {"by": "value", "direction": "descending"},
            },
        ),
        ("patch", {"order": {"by": "value", "direction": "ascending"}}),
        ("patch", {"order": None}),
    ],
    ids=["recreate", "patch", "patch-clears"],
)
async def test_edit_passes_an_order_through_as_the_model_wrote_it(
    field: str, drafted: dict[str, Any]
) -> None:
    """A drafted `order` is passed on, and the edit route makes it legal."""
    other = "recreate" if field == "patch" else "patch"
    client = _FakeClient(
        json.dumps({"text": "Done.", f"{field}_json": json.dumps(drafted), f"{other}_json": None})
    )

    out = await GeminiEdit(model_name="test-model", timeout_s=90.0, client=client)(
        message="sort it",
        stored_intent={"type": "bar_chart", "title": "Aid"},
        frame_view={"dimensions": []},
    )

    assert out[field] == drafted


@pytest.mark.parametrize(
    "period",
    [
        pytest.param(
            {"window": {"start": {"anchor": "latest", "offset": -2}, "end": _LATEST}},
            id="relative-window",
        ),
        pytest.param({"window": {"start": "2015", "end": "2020"}}, id="fixed-window"),
        pytest.param(
            {"change": {"basis": [{"anchor": "latest", "offset": -1}, _LATEST]}},
            id="relative-change",
        ),
        pytest.param({"change": {"basis": ["2015", "2020"]}}, id="fixed-change"),
    ],
)
async def test_edit_passes_a_recreates_period_through_as_the_model_wrote_it(
    period: dict[str, Any],
) -> None:
    recreate = {"type": "stat_card", "title": "Aid", "select": {"donor": "country/FRA"}, **period}
    client = _FakeClient(
        json.dumps({"text": "Done.", "patch_json": None, "recreate_json": json.dumps(recreate)})
    )

    drafted = await GeminiEdit(model_name="test-model", timeout_s=90.0, client=client)(
        message="change the period",
        stored_intent={"type": "stat_card", "title": "Aid", "select": {"donor": "country/FRA"}},
        frame_view={"dimensions": [], "rolling": True},
    )

    assert drafted["recreate"] == recreate
    (content,) = client.aio.models.contents
    assert json.loads(content)["frame_view"]["rolling"] is True
    # The call is bounded by the request's model-time budget, in milliseconds.
    (config,) = client.aio.models.configs
    assert config.http_options.timeout == 90_000


_DRAFT = json.dumps({"text": "Done.", "patch_json": '{"title": "Aid"}', "recreate_json": None})


async def _edit(client: _FakeClient, **kwargs: Any) -> dict[str, Any]:
    return await GeminiEdit(model_name="test-model", timeout_s=90.0, client=client, **kwargs)(
        message="retitle it",
        stored_intent={"type": "bar_chart", "title": "Old"},
        frame_view={"dimensions": []},
    )


@pytest.mark.parametrize(
    ("raw", "level"), [(None, "LOW"), (" medium ", "MEDIUM"), ("High", "HIGH"), ("", "LOW")]
)
async def test_the_edit_call_runs_at_the_edit_thinking_level(monkeypatch, raw, level) -> None:
    if raw is None:
        monkeypatch.delenv("EDIT_THINKING_LEVEL", raising=False)
    else:
        monkeypatch.setenv("EDIT_THINKING_LEVEL", raw)
    client = _FakeClient(_DRAFT)

    await _edit(client, thinking_level=Settings().edit_thinking_level)

    (config,) = client.aio.models.configs
    assert config.thinking_config == types.ThinkingConfig(thinking_level=types.ThinkingLevel(level))


def test_an_unknown_edit_thinking_level_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("EDIT_THINKING_LEVEL", "MINIMAL")

    with pytest.raises(ValueError, match="edit_thinking_level"):
        Settings()


async def test_the_edit_call_is_a_generation_span_with_its_usage(
    exported_spans: InMemorySpanExporter,
) -> None:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=40,
        candidates_token_count=10,
        thoughts_token_count=5,
    )

    await _edit(_FakeClient(_DRAFT, usage))

    (span,) = [s for s in exported_spans.get_finished_spans() if s.name == "chat test-model"]
    attributes = dict(span.attributes or {})
    assert {k: v for k, v in attributes.items() if k.startswith("gen_ai.")} == {
        "gen_ai.operation.name": "chat",
        "gen_ai.agent.name": EDIT_AGENT_NAME,
        "gen_ai.provider.name": "google",
        "gen_ai.system": "google",
        "gen_ai.request.model": "test-model",
        "gen_ai.response.model": "test-model",
        "gen_ai.response.finish_reasons": ("stop",),
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 15,
        "gen_ai.usage.cache_read.input_tokens": 40,
        "gen_ai.usage.reasoning.output_tokens": 5,
    }
    assert json.loads(str(attributes["langfuse.observation.usage_details"])) == {
        "input": 60,
        "output": 10,
        "input_cached_tokens": 40,
        "output_reasoning_tokens": 5,
    }
    assert attributes["dfo.model.finish_reason"] == "STOP"
    assert {k for k in attributes if k.startswith("studio.surface.")} == {
        "studio.surface.prompt_variant",
        "studio.surface.tool_desc",
        "studio.surface.instruction_hash",
        "studio.surface.declarations_hash",
        "studio.surface.history",
    }


async def test_the_edit_call_reports_its_usage_to_the_turn(
    exported_spans: InMemorySpanExporter,
) -> None:
    heard: list[ModelCall] = []
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100, candidates_token_count=10
    )
    token = current_model_usage.set(heard.append)
    try:
        await _edit(_FakeClient(_DRAFT, usage))
    finally:
        current_model_usage.reset(token)

    assert heard == [ModelCall(role="edit", model="test-model", usage=usage)]

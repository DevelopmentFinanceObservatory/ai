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
from harness.generative import _EDIT_INSTRUCTION, GeminiEdit


def test_edit_instruction_states_the_window_grammar() -> None:
    assert "window is an object of period labels" in _EDIT_INSTRUCTION
    assert '{"start": "2000", "end": "2023"}' in _EDIT_INSTRUCTION
    assert "Either side may be omitted" in _EDIT_INSTRUCTION
    assert '"since 2000" is {"start": "2000"}' in _EDIT_INSTRUCTION


def test_edit_instruction_updates_a_title_that_names_what_changed() -> None:
    assert "A title that names something you change" in _EDIT_INSTRUCTION
    assert '"... (2000-2023)"' in _EDIT_INSTRUCTION
    assert "open side read from the frame's coverage" in _EDIT_INSTRUCTION


def test_edit_instruction_states_the_relative_period_grammar() -> None:
    assert '{"anchor": "latest", "offset": N}' in _EDIT_INSTRUCTION
    assert '"The last three years" is {"start": {"anchor": "latest",' in _EDIT_INSTRUCTION
    assert "A window with a\n  relative side states both sides" in _EDIT_INSTRUCTION
    assert (
        '"Compare with\n  the previous year" is {"basis": [{"anchor": "latest", "offset": -1}'
        in (_EDIT_INSTRUCTION)
    )
    # Years a person names stay labels, in a window and in a change alike.
    assert '"2015\n  to 2020" is {"start": "2015", "end": "2020"}' in _EDIT_INSTRUCTION
    assert '"compare 2015 with 2020" is\n  {"basis": ["2015", "2020"]}' in _EDIT_INSTRUCTION
    assert "its title names no\n  year and no year range" in _EDIT_INSTRUCTION


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _FakeModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.contents: list[Any] = []

    async def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.contents.append(kwargs["contents"])
        return _FakeResponse(self._text)


class _FakeAio:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


class _FakeClient:
    """Stands in for `google.genai.Client`: only the `.aio.models.
    generate_content` path `_ModelCall._generate` calls is faked."""

    def __init__(self, text: str) -> None:
        self.aio = _FakeAio(text)


_LATEST = {"anchor": "latest", "offset": 0}


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

    drafted = await GeminiEdit(model_name="test-model", client=client)(
        message="change the period",
        stored_intent={"type": "stat_card", "title": "Aid", "select": {"donor": "country/FRA"}},
        frame_view={"dimensions": [], "rolling": True},
    )

    assert drafted["recreate"] == recreate
    (content,) = client.aio.models.contents
    assert json.loads(content)["frame_view"]["rolling"] is True

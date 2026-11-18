"""`PlannerAdapter` with its model calls answered by a `ScriptedGenai`.

`scripted_planner` builds the adapter's planner over the scripted client.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from unittest import mock

from compiler.surface import FixtureSurface, ToolSurface
from harness import gemini_wire
from harness.planner_adapter import PlannerAdapter
from harness.settings import Settings
from planner.settings import AgentSettings
from planner.tool_spec import ToolSpec
from translator.tool import Translate, build_translate

from .planner.planner_pairs import adapter_specs
from .scripted_genai import ScriptedGenai

# Maps the planner's tool specs to the specs it serves.
type Tools = Callable[[Sequence[ToolSpec]], Sequence[ToolSpec]]

MODEL = AgentSettings().planner_model_name
# Gemini signs the first function call of a response it thought before.
SIGNED = b"scripted-signature"


@contextlib.contextmanager
def numbered_call_ids() -> Iterator[None]:
    """Number the ids given a call that has none from 1, the model's and a
    turn put in the model's place alike (`harness.gemini_wire.client_call_id`),
    so each carries the same call id every time a test runs."""
    counter = itertools.count(1)

    def numbered() -> str:
        return f"adk-{next(counter)}"

    with mock.patch.object(gemini_wire, "client_call_id", numbered):
        yield


def scripted_planner(
    scripted: ScriptedGenai,
    *,
    surface: ToolSurface | None = None,
    translate: Translate | None = None,
    tools: Tools | None = None,
    on_tool_answered: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> PlannerAdapter:
    """The adapter over `surface` (an empty `FixtureSurface` by default), its model
    calls answered by `scripted`, serving the planner's tools as `tools` changes
    them, telling `on_tool_answered` each tool's answer."""
    surface = surface if surface is not None else FixtureSurface(schemas=[])
    translate = translate if translate is not None else build_translate(surface=surface)
    specs = adapter_specs(surface=surface, translate=translate)
    served = tools(specs) if tools is not None else specs
    return PlannerAdapter(
        settings=Settings(),
        surface=surface,
        translate=translate,
        specs=served,
        client=scripted.client,
        on_tool_answered=on_tool_answered,
    )

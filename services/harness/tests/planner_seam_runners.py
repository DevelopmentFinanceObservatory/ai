"""`PlannerSeam` on either planner runner, its model calls answered by a `ScriptedGenai`.

`seam_on` builds the seam on `"adk"` (ADK's `LlmAgent` on a `Gemini` over the
scripted client) or on `"pydantic_ai"` (`harness.planner_node` over the same
client), so one script drives either. `SSE` is Studio's streaming run config,
which a seam on the scripted client runs under: `ScriptedGenai` scripts the
streamed call only.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import Literal
from unittest import mock

from compiler.surface import FixtureSurface, ToolSurface
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.models import Gemini
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from harness.turn_budget import BoundedGemini
from planner.settings import AgentSettings
from planner.tool_spec import ToolSpec
from translator.tool import Translate, build_translate

from .planner.planner_pairs import seam_specs
from .scripted_genai import ScriptedGenai

type PlannerRunner = Literal["adk", "pydantic_ai"]
# A change to the planner's tools.
type Tools = Callable[[Sequence[ToolSpec]], Sequence[ToolSpec]]

PLANNER_RUNNERS: tuple[PlannerRunner, ...] = ("adk", "pydantic_ai")
MODEL = AgentSettings().planner_model_name
SSE = RunConfig(streaming_mode=StreamingMode.SSE)
# Gemini signs the first function call of a response it thought before.
SIGNED = b"scripted-signature"


@contextlib.contextmanager
def numbered_call_ids() -> Iterator[None]:
    """Number the ids ADK gives a call that has none from 1, so a turn put in the
    model's place carries the same call id on both planners."""
    counter = itertools.count(1)
    with mock.patch.object(
        adk_functions, "generate_client_function_call_id", lambda: f"adk-{next(counter)}"
    ):
        yield


def seam_on(
    runner: PlannerRunner,
    scripted: ScriptedGenai,
    *,
    surface: ToolSurface | None = None,
    bounded: bool = False,
    translate: Translate | None = None,
    tools: Tools | None = None,
) -> PlannerSeam:
    """The seam on `runner` over `surface` (an empty `FixtureSurface` by default),
    its model calls answered by `scripted`, serving the planner's tools as
    `tools` changes them. `bounded` keeps the ADK planner's model to the turn's
    budget, as the served planner's is. The Pydantic AI planner's model always
    keeps to it."""
    surface = surface if surface is not None else FixtureSurface(schemas=[])
    translate = translate if translate is not None else build_translate(surface=surface)
    specs = seam_specs(surface=surface, translate=translate)
    settings = Settings(planner_runner=runner)
    served = tools(specs) if tools is not None else specs
    if runner == "adk":
        model = (BoundedGemini if bounded else Gemini)(model=MODEL, client=scripted.client)
        return PlannerSeam(
            settings=settings, surface=surface, translate=translate, specs=served, model=model
        )
    return PlannerSeam(
        settings=settings,
        surface=surface,
        translate=translate,
        specs=served,
        client=scripted.client,
    )

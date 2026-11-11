"""The Pydantic AI planner run on a script, for tests that check what it sends and leaves.

`pydantic_run` runs the Pydantic AI planner (`planner_agent` on
`harness.planner_model.planner_model` over the scripted client) once per brief,
each run after the first handed the steps of the runs before it. It returns the
requests the scripted client recorded and the conversation each run leaves
(`Turn`), read off the planner's items.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from compiler.surface import ToolSurface
from google.adk.events import Event
from google.genai import types
from harness.planner_model import planner_model
from planner.instruction import planner_instruction
from planner.pydantic_agent import (
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    planner_agent,
    run_planner,
)
from planner.session import SessionCall
from planner.settings import AGENT_NAME
from planner.specs import planner_specs
from planner.tool_spec import ToolSpec
from translator.tool import Translate

from ..scripted_genai import Reply, Request, ScriptedGenai

MODEL = "gemini-3.8-flash"
QUESTION = "How much aid did Ghana receive?"
BRIEF = "Context for this question: the readings."

# The branch `PlannerSeam` runs the planner on (`harness.planner_seam`).
_BRANCH = "dfo_planner"

# One entry of a run's conversation: a model response's part (`types.Part`
# dumped) or a call's answer (`("answer", name, call_id, answer)`).
type Turn = tuple[Any, ...]


def seam_specs(*, surface: ToolSurface | None, translate: Translate | None) -> list[ToolSpec]:
    """The planner's tools as `harness.planner_seam.PlannerSeam` builds them over
    `surface` and `translate`, for a test that changes them before handing them
    to the seam as `specs`."""
    return planner_specs(
        translate=translate, surface=surface, question_fixed=True, reads_user_words=False
    )


@dataclass
class PairRun:
    """What the planner's runs sent and left: the requests in order, and each
    run's conversation (`Turn`s), in order."""

    requests: list[Request]
    conversations: list[list[Turn]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


async def pydantic_run(
    replies: Sequence[Reply],
    *,
    briefs: Sequence[str] = (BRIEF,),
    surface: ToolSurface | None = None,
    translate: Translate | None = None,
) -> PairRun:
    """The Pydantic AI planner's runs on `replies`, one per brief, over one state.

    Each run's tools read the events the node would have stored by then
    (`harness.planner_node`): the runs before it, each response of its own as
    it arrives, and the answers of a response once its calls have all answered.
    """
    scripted = ScriptedGenai(replies)
    planner = planner_agent(
        seam_specs(surface=surface, translate=translate),
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=True,
        ),
        model=planner_model(
            MODEL,
            client=scripted.client,
            budget=None,
        ),
    )
    state: dict[str, Any] = {}
    events = [Event(author="user", content=_user_content(QUESTION))]
    run = PairRun(requests=scripted.requests, state=state)
    for brief in briefs:
        deps = PlannerDeps(
            state=state,
            call_for=partial(SessionCall, events),
            tool_answered=lambda name, answer: None,
            tool_failed=lambda name, error: {"error": f"{name} failed"},
        )
        items: list[PlannerItem] = []
        async for item in run_planner(planner, deps, question=QUESTION, brief=brief):
            items.append(item)
            events += _stored(item)
        run.conversations.append(_item_turns(items))
    return run


def _user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def _dumped(part: types.Part) -> dict[str, Any]:
    return part.model_dump(mode="json", exclude_none=True)


def _item_turns(items: Sequence[PlannerItem]) -> list[Turn]:
    """The turns of the Pydantic AI planner's `items`."""
    turns: list[Turn] = []
    for item in items:
        if isinstance(item, PlannerResponse):
            turns.extend(("part", _dumped(part)) for part in item.parts)
        else:
            turns.extend(
                ("answer", answer.name, answer.id, answer.response) for answer in item.answers
            )
    return turns


def _stored(item: PlannerItem) -> list[Event]:
    """The event `item` stands for, if any, as far as `session_steps` reads it."""
    if isinstance(item, PlannerResponse):
        if not item.parts:
            return []
        content = types.Content(role="model", parts=list(item.parts))
    else:
        parts = [types.Part(function_response=answer) for answer in item.answers]
        content = types.Content(role="user", parts=parts)
    return [Event(author=AGENT_NAME, branch=_BRANCH, content=content)]

"""One planned Studio turn on the real root workflow, run on either planner.

`build_arm` serves the Studio routes (`tests.orchestrator_fakes.orchestrator_app`) with the
orchestrator's model scripted to the recorded `search` and `plan` calls of a planner-entry
task, and the real `PlannerSeam` as the planner, talking to a scripted google-genai client.
The `adk` arm leaves the seam as dev builds it. The `pai` arm swaps the planner node for
`pai_node.pai_planner_node` over the same seam.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import scripted as sc
import validate
from entry import (
    RecordedResolver,
    RecordedSurface,
    Task,
    TranslationCalls,
    _no_anchor_read,
    load_recording,
    load_tasks,
)
from executor.ledger import InMemoryRunLedger
from fastapi import FastAPI
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.conversation import current_conversation_id
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from pai_node import pai_planner
from tests.orchestrator_fakes import ScriptLlm, call, orchestrator_app
from translator.tool import build_translate

type Arm = Literal["adk", "pai"]

TASKS = {task.id: task for task in load_tasks()}


def orchestrator_script(task: Task) -> list[types.Part]:
    """The orchestrator's model calls for `task`: each recorded search, then the hand-off."""
    searches = [
        call("search", query=s.query, parts=s.parts, as_text=s.as_text) for s in task.searches
    ]
    asks = [a if isinstance(a, str) else a.model_dump(mode="json") for a in task.asks]
    return [*searches, call("plan", brief=task.brief, asks=asks)]


class StudioResolver(RecordedResolver):
    """`RecordedResolver` for a Studio turn, whose searches carry the turn's own context
    (the user's message and carried entries), which the eval's recording has none of. A turn
    after the first searches again, so the recording is served again once it is used up."""

    def __init__(self, calls: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(calls)
        self._all = list(calls)

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if not self._calls:
            self._calls = list(self._all)
        return await super().__call__(**{**kwargs, "context": None})


@dataclass
class Gate:
    """Holds the first translation inside `propose_plan` until `event` is set or `timeout_s`
    passes; with no `event` it only notes that the translation was reached."""

    event: Any = None
    timeout_s: float = 3.0
    reached: bool = False
    timed_out: bool = False

    async def wait(self) -> None:
        first, self.reached = not self.reached, True
        if not first or self.event is None:
            return
        try:
            await asyncio.wait_for(self.event.wait(), self.timeout_s)
        except TimeoutError:
            self.timed_out = True


@dataclass
class Built:
    app: FastAPI
    orchestrator: ScriptLlm
    client: sc.Scripted
    seam: PlannerSeam
    sessions: InMemorySessionService
    translation: TranslationCalls
    gate: Gate
    ledger: InMemoryRunLedger


async def build_arm(
    arm: Arm,
    task: Task,
    steps: Sequence[sc.Step],
    *,
    extra_orchestrator: Sequence[types.Part] = (),
    gate: Gate | None = None,
    pai: dict[str, Any] | None = None,
    tool_surface: Any = None,
) -> Built:
    """The Studio app for `task` with the planner's model answering from `steps`."""
    recording = load_recording(task.id)
    surface = RecordedSurface(recording.get("tools") or {})
    resolver = StudioResolver(recording["searches"])
    translation = TranslationCalls(
        recording.get("translation") or {}, schemas=await validate.tool_schemas()
    )
    held = gate or Gate()
    inner = build_translate(
        surface=translation,
        resolve_entities=translation.resolve_entities,
        anchor_read=_no_anchor_read,
        describe_nodes=translation.describe_nodes,
    )

    async def translate(planner_call: Any) -> Any:
        await held.wait()
        return await inner(planner_call)

    client = sc.scripted(steps)
    seam = PlannerSeam(settings=Settings(), surface=surface, translate=translate)
    sessions = InMemorySessionService()
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    planner: Any = seam
    if arm == "adk":
        seam._agent.model.client = client.client
    else:
        planner = pai_planner(
            seam, client=client.client, surface=surface, translate=translate, **(pai or {})
        )
    app, model = orchestrator_app(
        *orchestrator_script(task),
        *extra_orchestrator,
        resolver=resolver,
        planner=planner,
        session_service=sessions,
        run_ledger=ledger,
        tool_surface=tool_surface,
    )
    return Built(app, model, client, seam, sessions, translation, held, ledger)

"""A choice moved on the plan card, and a Stop, on the turn loop.

The `switch` node revises the plan in code for a scope row the user moved,
and offers it back to `approve`; a planner decision's row routes to the
planner, which replans with the pick held. Each runs under the turn's
`StopScope`, so the first Stop ends only the revision or the replan and
the card is offered again, while any other Stop cancels the node and closes
the turn.

The turns run the real nodes over a scripted orchestrator, a fake planner
and data-mcp over a Search stub (`tests.plan_turns.plan_nodes`), on
Postgres (`tests.turn_driver.TurnDriver`). A turn opens with the readings
and blocks an earlier search would have written (`tests.graph_turns.given`).
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast, override

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
)
from data_mcp.definition_store import InMemoryDefinitionStore
from data_mcp.definition_store.identity import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from executor.ledger import InMemoryRunLedger
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from harness.decision_pick import DecisionPick
from harness.graph import switch as switch_module
from harness.graph.state import (
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    PICK_REPLAN_KEY,
)
from harness.graph.switch import SCOPE_CHOICE_NOT_APPLIED, SWITCH_CANCELLED, SwitchFailedError
from harness.plan_display import choice_id, decision_option_id
from harness.planner_adapter import PlannerAdapter, store_searched_pin_readings
from harness.scope_choices import PLANNER_AXES_BLOCK_KEY, PartScope
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.sentences import StopCode
from harness.settings import Settings
from harness.turn.events import NodeDone, TurnFailed
from harness.turn.loop import Outcome, stop_turn
from harness.turn.store import Cursor, TurnStatus
from harness.turn_records import SearchReading
from planner.call import PlannerCall
from planner.plan import (
    DECISION_PICK_STATE_KEY,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    Period,
    PinSource,
    RankRequirement,
    ResolvedBy,
    SessionState,
    authored,
    decision_pick,
    load_plan,
    plan_revision,
    save_decision_pick,
    save_pin_sources,
    save_plan,
    windowed_pins,
)
from planner.plan import Plan as PlannerPlan
from planner.records import RankAmong, RankDirection, Reduction, StatVarReading
from planner.specs import PROPOSE_PLAN
from planner.workflow import ReadingSetBy, ToolId
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from submission.service import (
    AlignedPin,
    SearchedPin,
    save_aligned_pin,
    save_searched_pin,
    scope_choice_decisions,
)
from translator.discovery import RESOLVED_STATE_PREFIX

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .graph_turns import given, opening_graph
from .plan_cards import (
    _PLAN_WINDOW,
    ScriptedExecutor,
    _fixture_surface,
    _get_data_manifest,
    _Planner,
    _planner_plan,
    _row_id,
    _scoped_plan,
)
from .plan_turns import approve_answer, plan_nodes
from .planner.planner_pairs import adapter_specs
from .planner_expectations import expect, expectation_name
from .planner_host import node_ctx
from .scope_scenarios import (
    _AFRICA_TOTAL,
    _AFRICA_UNKNOWN_CLASS,
    _AFRICA_UNPOSTABLE,
    _hashed_scope,
    _switched_answer,
)
from .scripted_genai import ScriptedGenai, call_part, reply
from .scripted_planner import SIGNED
from .scripted_tools import plan_call, scripted_orchestrator
from .session_events import one_answer
from .test_planner_node import _ask_call, _proposals, _steps
from .turn_driver import SegmentRun, TurnDriver, numbered_call_ids, scripted_models

_BRIEF = "kenya debt"


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas, which `_planner_plan`'s steps are hashed against."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


def _driver(
    engine: AsyncEngine,
    planner: Any,
    *,
    search: SearchStub | None = None,
    tool_surface: Any = None,
    specs: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> TurnDriver:
    """One instance whose agent hands `_BRIEF` to `planner`, which drafts the
    card and revises its switches, posting through data-mcp over `search`.
    The turn opens with `specs` and `state` when given, as earlier searches
    would have left them."""
    models = scripted_models(orchestrator=[reply(plan_call(_BRIEF))])
    nodes = plan_nodes(
        scripted_orchestrator(),
        models,
        run_ledger=InMemoryRunLedger(),
        tool_surface=tool_surface if tool_surface is not None else ScriptedSurface(),
        executor=ScriptedExecutor(),
        planner=planner,
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search)
        if search is not None
        else None,
    )
    before = given(specs=specs, state=state) if specs or state else None
    return TurnDriver(engine, graph=opening_graph(nodes, before=before))


def _output_of(run: SegmentRun, node: str) -> Any:
    """What `node` last handed on in `run`."""
    outputs = [e.output for e in run.events if isinstance(e, NodeDone) and e.node == node]
    return outputs[-1] if outputs else None


def _pauses_on_the_card(run: SegmentRun) -> bool:
    return run.outcome is Outcome.PAUSED and run.pause.kind == "approve"


async def _until(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.005)


# --- a planner decision's pick ----------------------------------------------------------


_PLANNER_ASK = "planner_ask"


# What a `PlannerModel` answers for a planner run that ended on a question for
# the user, which the `planner` node pauses on.
_ASKED = {"asked": {"call_id": "a1", "args": {"message": "Which years?"}}}


def _picked_window(plan: PlannerPlan) -> DecisionPick:
    """The pick of a planner decision the card offers to switch
    (`plan_display.switchable_decision`): `_planner_plan`'s window, fixing the
    period, switched to its other option on `plan`."""
    return DecisionPick.on_card(authored(plan), slot="window", chosen="2018-2024")


_PICK_SWITCH = {"choice_id": "decision/window", "option_id": decision_option_id("2018-2024")}
# A drafting key the replan writes, which a replan that does not land puts back.
_DRAFTING_KEY = "planner:replan_scratch"


def _pickable_plan(
    *, schemas: list[ToolSchema], chosen: str = "2015-2024", version: int = 1
) -> PlannerPlan:
    """`_planner_plan` with its window decision fixing the period, so the card offers it."""
    plan = _planner_plan(schemas=schemas, version=version)
    (window,) = plan.decisions
    fixed = window.model_copy(
        update={"chosen": chosen, "fixes": DecisionFixes(period=Period(start=2015, end=2024))}
    )
    return plan.model_copy(update={"decisions": (fixed,)})


class _ReplanPlanner(_Planner):
    """A `PlannerModel` that drafts its first plan outright and, on every
    later call, writes `_DRAFTING_KEY` and the pick it is handed, as
    `PlannerAdapter` records it, notes the snapshot the state holds under
    `PICK_REPLAN_KEY` (`held_snapshot`), waits for `release` when given one,
    and then answers its next scripted result."""

    def __init__(self, plans: list[dict[str, Any]], release: asyncio.Event | None = None) -> None:
        super().__init__(plans)
        self.release = release
        self.replanning = asyncio.Event()
        self.held_snapshot: Any = None

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if self.calls:
            ctx = kwargs["ctx"]
            self.held_snapshot = ctx.state.get(PICK_REPLAN_KEY)
            ctx.state[_DRAFTING_KEY] = "draft"
            if (pick := kwargs.get("pick")) is not None:
                save_decision_pick(ctx.state, pick)
            self.replanning.set()
            if self.release is not None:
                await self.release.wait()
        return await super().__call__(**kwargs)


async def _carded(driver: TurnDriver) -> SegmentRun:
    """The turn's first segment, paused on the plan card."""
    paused = await driver.turn(_BRIEF)
    assert _pauses_on_the_card(paused), paused.outcome
    return paused


def _answer_switch(switch: Mapping[str, Any]) -> dict[str, Any]:
    return {"approved": None, "switch": dict(switch)}


async def test_a_planner_decisions_pick_replans_with_the_pick_held(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """A switch of a planner decision's row routes to the planner, which is
    handed the pick, no amendment, and the plan on record, and offers the
    replanned plan with nothing else moved."""
    plan = _pickable_plan(schemas=real_schemas)
    replanned = _pickable_plan(schemas=real_schemas, chosen="2018-2024", version=2)
    planner = _ReplanPlanner(
        [
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"plan": replanned.model_dump(mode="json"), "summary": {}},
        ]
    )
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause

    run = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)

    assert planner.calls[1]["pick"] == _picked_window(plan)
    assert planner.calls[1]["amendment"] is None
    assert planner.calls[1]["prior_plan"] == plan.model_dump(mode="json")
    offered = _output_of(run, "planner")
    assert offered["revision"] == plan_revision(replanned)
    assert "moved" not in offered
    assert _pauses_on_the_card(run)
    state = run.cursor.state
    assert state[_DRAFTING_KEY] == "draft"
    # The pick belongs to its entry, and only a pause records the snapshot.
    assert decision_pick(cast(SessionState, state)) is None
    assert planner.held_snapshot is None
    assert state.get(PICK_REPLAN_KEY) is None


async def test_a_switch_to_the_option_a_planner_decision_has_offers_the_plan_unchanged(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """A switch of a planner decision's row to the option it already has
    does not apply: the planner is not called again, and the card offers the
    plan on record with `switch_failed`, saying the choice is already on it."""
    plan = _pickable_plan(schemas=real_schemas)
    planner = _ReplanPlanner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause
    switch = {"choice_id": "decision/window", "option_id": decision_option_id("2015-2024")}

    run = await driver.answer(_answer_switch(switch), to=card)

    assert len(planner.calls) == 1
    offered = _output_of(run, "switch")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"]["choice_id"] == "decision/window"
    assert offered["switch_failed"]["reason"] == "That choice is already on the plan."
    assert PlannerPlan.model_validate_json(run.cursor.state["plan"]) == plan


# A pick dump an earlier release wrote: it reads what the card's requirements
# read as one digest and names the decisions to cite, and states no fixes.
_OLD_SHAPE_PICK = {
    "slot": "window",
    "chosen": "2018-2024",
    "origin_reading": "0" * 64,
    "cited": ["window"],
    "kept_requirements": {},
    "kept_decisions": {},
}


async def test_a_pick_this_release_cannot_read_offers_the_plan_unchanged(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approve payload carries a pick in an earlier release's shape, as
    one written before a deploy. No replan runs: the card offers the plan on
    record with `switch_failed`, asking the user to make the choice again,
    and no pick is left in state."""
    plan = _pickable_plan(schemas=real_schemas)
    planner = _ReplanPlanner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause
    monkeypatch.setattr(
        "harness.graph.approve.decision_pick",
        lambda *_, **__: SimpleNamespace(model_dump=lambda **_: _OLD_SHAPE_PICK),
    )

    run = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)

    assert len(planner.calls) == 1
    offered = _output_of(run, "planner")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"] == {
        **_PICK_SWITCH,
        "code": SCOPE_CHOICE_NOT_APPLIED,
        "reason": "That choice could not be applied. Make it again.",
        "retryable": True,
    }
    state = run.cursor.state
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert state.get(DECISION_PICK_STATE_KEY) is None


class _StaleSnapshotPlanner(_ReplanPlanner):
    """`_ReplanPlanner` whose first draft leaves `PICK_REPLAN_KEY` holding a
    snapshot, as a pick replan paused on an ask the user never answered does."""

    def __init__(self, plans: list[dict[str, Any]], stale: dict[str, Any]) -> None:
        super().__init__(plans)
        self.stale = stale

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if not self.calls:
            kwargs["ctx"].state[PICK_REPLAN_KEY] = self.stale
        return await super().__call__(**kwargs)


async def test_a_pick_this_release_cannot_read_puts_back_no_stale_snapshot(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot an earlier, unanswered pick replan left under
    `PICK_REPLAN_KEY` belongs to no resume. A fresh entry whose pick cannot
    be read leaves the plan and the drafting state as they stand, so the
    state on record matches the card it offers."""
    plan = _pickable_plan(schemas=real_schemas)
    stale = {
        "plan": _pickable_plan(
            schemas=real_schemas, chosen="2018-2024", version=2
        ).model_dump_json(),
        _DRAFTING_KEY: "stale",
    }
    planner = _StaleSnapshotPlanner([{"plan": plan.model_dump(mode="json"), "summary": {}}], stale)
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    carded = await _carded(driver)
    assert carded.cursor.state[PICK_REPLAN_KEY] == stale
    monkeypatch.setattr(
        "harness.graph.approve.decision_pick",
        lambda *_, **__: SimpleNamespace(model_dump=lambda **_: _OLD_SHAPE_PICK),
    )

    run = await driver.answer(_answer_switch(_PICK_SWITCH), to=carded.pause)

    assert _output_of(run, "planner")["switch_failed"]["code"] == SCOPE_CHOICE_NOT_APPLIED
    state = run.cursor.state
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert state.get(_DRAFTING_KEY) != "stale"
    assert state.get(PICK_REPLAN_KEY) is None


@pytest.mark.parametrize(
    ("code", "final", "reported", "retryable"),
    [
        ("budget_exhausted", False, "budget_exhausted", False),
        (None, False, StopCode.PLAN_NOT_READY.value, False),
        (StopCode.DISCOVERY_UNAVAILABLE.value, True, StopCode.DISCOVERY_UNAVAILABLE.value, True),
    ],
    ids=["coded", "uncoded", "service-unreachable"],
)
async def test_a_refused_pick_replan_offers_the_plan_unchanged_and_puts_its_state_back(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    code: str | None,
    final: bool,
    reported: str,
    retryable: bool,
) -> None:
    """The planner refuses the replan: the card offers the plan on record
    again with the switch node's `switch_failed`, carrying the refusal's code,
    or the planner's `plan_not_ready` for a refusal stating none, and the
    sentence for the user, and the drafting state is put back. The same pick
    can succeed again only where a scope switch's revision could, whatever
    the refusal's `final` says."""
    plan = _pickable_plan(schemas=real_schemas)
    refused = {
        "text": "The draft budget is spent.",
        "final": final,
        "code": code,
        "reason": "This request has used its budget of 4 plan drafts.",
    }
    planner = _ReplanPlanner(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}, {"refused": refused}]
    )
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause

    run = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)

    offered = _output_of(run, "planner")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"] == {
        **_PICK_SWITCH,
        "code": reported,
        "reason": refused["reason"],
        "retryable": retryable,
    }
    assert _pauses_on_the_card(run)
    assert run.cursor.state.get(_DRAFTING_KEY) is None
    assert PlannerPlan.model_validate_json(run.cursor.state["plan"]) == plan


async def test_a_stopped_pick_replan_offers_the_plan_unchanged_and_puts_its_state_back(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """The user stops the replan while the planner drafts: the draft is
    cancelled, and the card offers the plan on record again under
    `switch_cancelled`, as a stopped scope switch does."""
    plan = _pickable_plan(schemas=real_schemas)
    planner = _ReplanPlanner(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}], release=asyncio.Event()
    )
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause

    resumed = asyncio.ensure_future(driver.answer(_answer_switch(_PICK_SWITCH), to=card))
    await asyncio.wait_for(planner.replanning.wait(), timeout=5)
    assert await driver.stop()
    run = await asyncio.wait_for(resumed, timeout=5)

    offered = _output_of(run, "planner")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"]["code"] == SWITCH_CANCELLED
    assert offered["switch_failed"]["retryable"] is True
    assert _pauses_on_the_card(run)
    assert run.cursor.state.get(_DRAFTING_KEY) is None


class _AskingOnPickPlanner(_ReplanPlanner):
    """`_ReplanPlanner` that, on its second call, asks the user a question
    after writing `_DRAFTING_KEY` and the pick, as `PlannerAdapter` records it."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if len(self.calls) == 1:
            kwargs["ctx"].state[_DRAFTING_KEY] = "draft"
            save_decision_pick(kwargs["ctx"].state, kwargs["pick"])
            self.calls.append(kwargs)
            return _ASKED
        return await super().__call__(**kwargs)


async def test_a_planner_ask_during_a_pick_replan_keeps_its_state_and_resumes_with_the_pick(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """The planner asks the user a question while replanning for a pick. The
    replan runs as a task so a Stop can reach it, and the ask still pauses
    the turn, with the replan's state kept for the resume, which is handed
    the pick again and offers the replanned plan."""
    plan = _pickable_plan(schemas=real_schemas)
    replanned = _pickable_plan(schemas=real_schemas, chosen="2018-2024", version=2)
    planner = _AskingOnPickPlanner(
        [
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"plan": replanned.model_dump(mode="json"), "summary": {}},
        ]
    )
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause

    paused = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)
    assert paused.pause.kind == _PLANNER_ASK
    assert paused.cursor.state[_DRAFTING_KEY] == "draft"

    run = await driver.answer({"answer": "2018"}, to=paused.pause)

    assert planner.calls[2]["pick"] == _picked_window(plan)
    assert _output_of(run, "planner")["revision"] == plan_revision(replanned)
    assert _pauses_on_the_card(run)


@pytest.mark.parametrize("ending", ["refused", "stopped"])
async def test_a_pick_replan_failing_after_an_ask_puts_back_the_state_from_before_the_pick(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], ending: str
) -> None:
    """The planner asks a question while replanning for a pick, and the
    resumed replan is refused or the user stops it. The card offers the plan
    unchanged with `switch_failed`, and the state is as it stood before the
    pick, with no drafting state, no pick for a later revision of the plan
    to hold, and no record of the replan."""
    plan = _pickable_plan(schemas=real_schemas)
    refused = {"text": "No plan.", "final": False, "code": "plan_not_ready", "reason": "No plan."}
    planner = _AskingOnPickPlanner(
        [
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"refused": refused},
        ],
        release=asyncio.Event() if ending == "stopped" else None,
    )
    driver = _driver(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    card = (await _carded(driver)).pause
    paused = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)
    assert paused.pause.kind == _PLANNER_ASK
    assert paused.cursor.state[PICK_REPLAN_KEY] is not None

    resumed = asyncio.ensure_future(driver.answer({"answer": "2018"}, to=paused.pause))
    if ending == "stopped":
        await asyncio.wait_for(planner.replanning.wait(), timeout=5)
        assert await driver.stop()
    run = await asyncio.wait_for(resumed, timeout=5)

    offered = _output_of(run, "planner")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"]["code"] == (
        SWITCH_CANCELLED if ending == "stopped" else "plan_not_ready"
    )
    assert _pauses_on_the_card(run)
    state = run.cursor.state
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert state.get(_DRAFTING_KEY) is None
    assert decision_pick(cast(SessionState, state)) is None
    assert state.get(PICK_REPLAN_KEY) is None


# --- a Stop outside a switch ---------------------------------------------------------------


class _StallingPlanner:
    """A `PlannerModel` that blocks mid-draft until released, recording
    whether its own task was cancelled and what it wrote to the turn's
    state: a turn stopped mid-draft must leave the draft gone and nothing it
    wrote after the Stop."""

    def __init__(self, *, ready: asyncio.Event, release: asyncio.Event) -> None:
        self._ready = ready
        self._release = release
        self.task: asyncio.Task[Any] | None = None
        self.state: dict[str, Any] | None = None

    async def __call__(self, *, ctx: Any, **_kwargs: Any) -> dict[str, Any]:
        self.task = asyncio.current_task()
        self.state = ctx.state
        self.state["planner_writes"] = [*self.state.get("planner_writes", []), "before"]
        self._ready.set()
        await self._release.wait()
        self.state["planner_writes"] = [*self.state.get("planner_writes", []), "after"]
        return {"plan": {}, "summary": {}}


async def test_a_turn_closed_mid_draft_cancels_the_planner_and_writes_no_state_after(
    pg_engine: AsyncEngine,
) -> None:
    """A Stop while the planner drafts, outside any switch, cancels the
    `planner` node and closes the turn `cancelled`. The draft must not
    survive: it is a live model and tool round trip that would otherwise
    keep running, and keep writing state, after the turn has closed."""
    ready, release = asyncio.Event(), asyncio.Event()
    planner = _StallingPlanner(ready=ready, release=release)
    driver = _driver(pg_engine, planner)

    running = asyncio.ensure_future(driver.turn(_BRIEF))
    await asyncio.wait_for(ready.wait(), timeout=5)
    assert planner.task is not None and not planner.task.done(), "the draft must be in flight"
    assert await driver.stop()
    run = await asyncio.wait_for(running, timeout=5)

    # If the draft survived the close it is still blocked on `release`;
    # release it and let it finish, so a late write shows up if one lands.
    release.set()
    await asyncio.sleep(0.05)

    assert run.outcome is Outcome.CANCELLED
    assert planner.task.cancelled(), "a turn closed mid-draft must cancel the draft, not orphan it"
    assert planner.state is not None
    assert planner.state.get("planner_writes") == ["before"], (
        "no draft write may land after the close"
    )
    assert "planner_writes" not in run.cursor.state


class _ReportsThenWaits(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` sends one progress report and then
    waits for `release` before answering, standing in for data-mcp polling
    a background apis operation."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.waiting: asyncio.Task[Any] | None = None
        self.cancelled = False

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            assert on_progress is not None
            on_progress(ProgressReport(progress=3, message="Preparing the dataset"))
            self.waiting = asyncio.current_task()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def test_stopping_a_turn_mid_step_cancels_the_run_and_leaves_no_task_behind(
    pg_engine: AsyncEngine,
) -> None:
    """A Stop while an approved plan's step waits on its call cancels the
    `execute` node, and the plan's run, a task of its own, with it."""
    surface = _ReportsThenWaits()
    driver = _driver(pg_engine, ScriptedPlannerModel(), tool_surface=surface)
    card = (await _carded(driver)).pause

    running = asyncio.ensure_future(
        driver.answer(approve_answer(card.payload["revision"]), to=card)
    )
    await _until(lambda: surface.waiting is not None)
    assert await driver.stop()
    run = await asyncio.wait_for(running, timeout=5)

    assert run.outcome is Outcome.CANCELLED
    assert surface.waiting is not None and surface.waiting.done()
    assert surface.cancelled


# --- the switch node: a scope choice moved on the plan card, in code ----------


class _RevisingPlanner(_Planner):
    """A `PlannerModel` drafting `plans` and answering each revision with `revised`."""

    def __init__(self, plans: list[dict[str, Any]], revised: dict[str, Any]) -> None:
        super().__init__(plans)
        self._revised = revised
        self.revisions: list[dict[str, Any]] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        kwargs["ctx"].state["planner:rounds"] = 1
        return self._revised


# The dataset `_StoppedPlanner`'s revision searches for an input it pinned none for.
_SEARCHED_PIN = "spec_" + "7" * 32


class _StoppedPlanner(_RevisingPlanner):
    """A `_RevisingPlanner` whose revision the user stops while it runs: it writes
    drafting state, searches a dataset for an input it pinned none for and stores
    its reading (`store_searched_pin_readings`), as a revision of a share does,
    takes the Stop on the turn's open `StopScope` as the loop does, and waits to
    be cancelled."""

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        ctx = kwargs["ctx"]
        ctx.state["planner:rounds"] = 1
        block = {"spec_id": _SEARCHED_PIN, "resolved": {"scope_choices": None}}
        save_searched_pin(
            cast(SessionState, ctx.state),
            SearchedPin(
                requirement_key="total",
                status="definite",
                spec_id=_SEARCHED_PIN,
                digest={
                    "part_digests": [
                        {
                            "variable": "US total ODA",
                            "status": "definite",
                            "spec_id": _SEARCHED_PIN,
                            "spec_block": block,
                        }
                    ]
                },
                asked_roles=("donor",),
            ),
        )
        store_searched_pin_readings(cast(SessionState, ctx.state))
        assert _SEARCHED_PIN in [r["spec_id"] for r in ctx.state[ORCHESTRATOR_SEARCHES_KEY]]
        assert ctx.stop.take("stop_1"), "the revision runs under the turn's open stop scope"
        return await self._stopped()

    async def _stopped(self) -> dict[str, Any]:
        """What the revision does once the Stop is taken: waits to be cancelled."""
        # Cancelled at once; the bound fails the test, not the suite, if not.
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")


class _FinishingStoppedPlanner(_StoppedPlanner):
    """A `_StoppedPlanner` whose revision finishes in the step the Stop comes in,
    so the switch's wait sees both done."""

    async def _stopped(self) -> dict[str, Any]:
        return self._revised


class _Switch(NamedTuple):
    """A switch answered on the card (`_switch`): the instance, the planner, the
    plan, the scope and the segment the answer ran."""

    driver: TurnDriver
    planner: _RevisingPlanner
    plan: PlannerPlan
    scope: dict[str, Any]
    run: SegmentRun


async def _switch(
    engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    search: SearchStub,
    revised: dict[str, Any] | None = None,
    *,
    option_id: str = "sector:120+130",
    aligned: bool = False,
    entity_groups: dict[str, Any] | None = None,
    stored_unreadable: bool = False,
    stored_groups: list[dict[str, Any]] | None = None,
    planner_type: type[_RevisingPlanner] = _RevisingPlanner,
    earlier_whole: AlignedPin | None = None,
) -> _Switch:
    """A turn paused on a plan pinning the compiled US health aid source, its
    reading stored (marked as one the planner compiled to align a basis when
    `aligned`, as one whose sets were not readable when `stored_unreadable`,
    and as one resolved with `stored_groups` when given) beside its block
    naming `entity_groups` when given, held as the whole `earlier_whole` an
    earlier plan compiled from its part when given, then answered with a
    switch to `option_id`, by default its sector to CRS 120 and 130, through
    `search`."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    plan = _scoped_plan(pin, scope, schemas=real_schemas)
    planner = planner_type(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}],
        revised or {"refused": {"text": "unused"}},
    )
    reading = SearchReading.from_part(
        {
            "variable": "health aid",
            "status": "definite",
            "spec_id": pin,
            "scope": scope,
            "planner_axes": ["measure_basis"] if aligned else [],
            "entity_groups": stored_groups or [],
            "entity_groups_unreadable": stored_unreadable,
        }
    )
    state: dict[str, Any] = {ORCHESTRATOR_SEARCHES_KEY: [reading.model_dump(mode="json")]}
    if earlier_whole is not None:
        save_aligned_pin(
            cast(SessionState, state),
            earlier_whole.model_copy(update={"spec_id": pin}),
            warnings=(),
        )
    specs = (
        {
            pin: {
                "spec_id": pin,
                "resolved": {"scope_choices": scope, "entity_groups": entity_groups},
            }
        }
        if entity_groups is not None
        else None
    )
    driver = _driver(engine, planner, search=search, specs=specs, state=state)
    card = (await _carded(driver)).pause
    choice_id = _row_id(plan, option_id.split(":", maxsplit=1)[0])
    run = await driver.answer(
        _answer_switch({"choice_id": choice_id, "option_id": option_id}), to=card
    )
    return _Switch(driver, planner, plan, scope, run)


@pytest.mark.parametrize("basis_row", [True, False], ids=["basis-row", "no-basis-row"])
async def test_a_switch_compiles_through_search_repins_the_plan_and_says_what_moved(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], basis_row: bool
) -> None:
    """The switch node posts the scope's base identity with its entries and
    the switch, registers what Search compiles, re-pins the plan to it with
    no planner call, stores the new reading, and offers the revised plan
    with the basis the switch moved, under the id its row has, or would have
    on a plan showing no basis row."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    if not basis_row:
        basis = {d.slot for d in revised.decisions if d.scope and d.scope.axis == "measure_basis"}
        revised = PlannerPlan.model_validate(
            {
                **revised.model_dump(),
                "decisions": [d for d in revised.decisions if d.slot not in basis],
                "evidence_requirements": [
                    r.model_copy(
                        update={"decision_ids": tuple(i for i in r.decision_ids if i not in basis)}
                    )
                    for r in revised.evidence_requirements
                ],
            }
        )
    search = SearchStub(answer={scope["id"]: envelope})

    switched = await _switch(
        pg_engine, real_schemas, search, {"plan": revised.model_dump(mode="json"), "summary": {}}
    )

    planner, plan, run = switched.planner, switched.plan, switched.run
    assert len(planner.calls) == 1  # the first draft; the switch drafts nothing
    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert (entry["variable_text"], entry["identity"]) == (scope["id"], scope["base"]["identity"])
    assert entry["scope_entries"] == [
        scope["entries"][1],
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": f"{scope['id']}:sector"},
        },
    ]
    (revision,) = planner.revisions
    assert revision["inherits"] == {new_pin: scope["compiled"]["spec_id"]}
    assert [r["source_spec_id"] for r in revision["evidence_requirements"]] == [new_pin]
    assert PlannerPlan.model_validate(revision["prior_plan"]) == plan
    state = run.cursor.state
    assert new_pin in state["planner:pin_sources"]
    assert PlannerPlan.model_validate_json(state["plan"]) == revised
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert (stored.text, stored.spec_id) == ("health aid", new_pin)
    assert stored.scope is not None
    assert stored.scope.choices[0].active == "sector:120+130"
    assert _pauses_on_the_card(run)
    offered = _output_of(run, "switch")
    assert offered["revision"] == plan_revision(revised)
    assert offered["moved"] == [
        {
            "choice_id": _row_id(revised, "measure_basis"),
            "from_label": "gross disbursements, current USD",
            "to_label": "grant equivalents",
            "reason": "It conflicts with another choice.",
        }
    ]


def test_one_move_across_two_switched_sources_with_no_row_is_listed_once(
    real_schemas: list[ToolSchema],
) -> None:
    """Two switched sources, each pinned by a requirement of the plan and each
    moved from gross disbursements to grant equivalents, with no basis row on
    the card: the move is listed once, under the id a row over both would have."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    after = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]

    def source(prefix: str) -> Any:
        before, answered = copy.deepcopy(scope), copy.deepcopy(after)
        for raw in (before, answered):
            raw["id"] = f"{prefix}_{raw['id']}"
            for choice in raw["choices"]:
                choice["id"] = f"{raw['id']}:{choice['axis']}"
        return switch_module._Switched(
            pin=f"{prefix}_pin",
            reading=cast(Any, None),
            scope=PartScope.from_search(before),
            envelope={},
            spec_id=f"{prefix}_spec",
            after=PartScope.from_search(answered),
        )

    (dataset,) = [
        r
        for r in _planner_plan(schemas=real_schemas).evidence_requirements
        if isinstance(r, DatasetRequirement)
    ]
    plan = SimpleNamespace(
        decisions=(),
        evidence_requirements=[
            dataset.model_copy(update={"id": "first", "source_spec_id": "a_spec"}),
            dataset.model_copy(update={"id": "second", "source_spec_id": "b_spec"}),
        ],
    )

    moved = switch_module._moved_entries(
        cast(Any, plan), [source("a"), source("b")], axis="sector", stated=cast(Any, plan)
    )

    assert [entry["choice_id"] for entry in moved] == ["first:measure_basis"]


async def test_a_switch_on_a_pin_an_earlier_plan_declared_a_whole_is_applied(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """The switched source was compiled for an earlier plan as a whole with its
    sector left open, and the plan now pins it as a plain requirement with no
    `whole_of`: it is no whole, so the user's pick of one sector is posted to
    Search and the revised plan offered."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    search = SearchStub(answer={scope["id"]: envelope})
    part = "spec_" + "4" * 32
    whole = AlignedPin(origin=part, option_ids=(), spec_id="", digest={}, whole_axis="sector")

    switched = await _switch(
        pg_engine,
        real_schemas,
        search,
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        earlier_whole=whole,
    )

    assert len(search.posted) == 1
    assert len(switched.planner.revisions) == 1
    offered = _output_of(switched.run, "switch")
    assert "switch_failed" not in offered
    assert offered["revision"] == plan_revision(revised)


_AFRICA_POSTED = {"destination": {"label": "Africa", "ref": "africa"}}


@pytest.mark.parametrize(
    ("group", "stored_unreadable", "stored_groups", "posted"),
    [
        pytest.param(_AFRICA_TOTAL, False, None, _AFRICA_POSTED, id="block-reads-stored-readable"),
        pytest.param(_AFRICA_TOTAL, True, None, _AFRICA_POSTED, id="block-reads-stored-unreadable"),
        pytest.param(
            _AFRICA_UNKNOWN_CLASS,
            True,
            None,
            _AFRICA_POSTED,
            id="block-of-an-unknown-class-stored-unreadable",
        ),
        pytest.param(
            _AFRICA_UNPOSTABLE,
            False,
            [{"role": "destination", **_AFRICA_TOTAL}],
            _AFRICA_POSTED,
            id="block-unreadable-stored-readable",
        ),
        # The block names no sets and the reading was stored with the G7 on its
        # donors: the stored set is posted, since the block's none says nothing of it.
        pytest.param(
            None,
            False,
            [{"role": "origin", "label": "G7"}],
            {"origin": {"label": "G7"}},
            id="block-names-none-stored-readable",
        ),
    ],
)
async def test_a_switch_posts_the_parts_groups_as_every_resubmit_posts_them(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    group: dict[str, Any] | None,
    stored_unreadable: bool,
    stored_groups: list[dict[str, Any]] | None,
    posted: dict[str, Any],
) -> None:
    """The part's block names Africa as the set its recipients were asked for,
    with the total party Search states for it. The switch posts the set as a
    compile on another basis does (`compiler.sources.wire_groups`): its label
    and ref, and no total, which Search recomputes. It reads the block again,
    so a reading stored when an earlier release could not read the set switches
    once this release can. A set naming a class of party this release does not
    know is posted by its label and ref, all the resubmit sends of it. Where what
    the block's set posts does not read, as a set stating a registry id that is
    not text, or where the block names no sets, it posts the
    readable sets the reading was stored with."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    await _switch(
        pg_engine,
        real_schemas,
        search,
        entity_groups={"destination": group} if group is not None else {},
        stored_unreadable=stored_unreadable,
        stored_groups=stored_groups,
    )

    (sent,) = search.posted
    (entry,) = sent["input"]["entries"]
    assert entry.get("entity_groups") == posted


async def test_a_switch_whose_parts_sets_cannot_be_read_offers_the_plan_unchanged(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """The part's block names a set on its recipients with a registry id that is not
    text, so what the switch would post of it cannot be read. Posting the part without it would let
    Search count another population, so the switch posts nothing and offers the
    plan unchanged under `switch_failed`, which asking again cannot get past."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    switched = await _switch(
        pg_engine, real_schemas, search, entity_groups={"destination": _AFRICA_UNPOSTABLE}
    )

    assert search.posted == []
    offered = _output_of(switched.run, "switch")
    assert offered["revision"] == plan_revision(switched.plan)
    failed = offered["switch_failed"]
    assert (failed["code"], failed["retryable"]) == (SCOPE_CHOICE_NOT_APPLIED, False)
    assert PlannerPlan.model_validate_json(switched.run.cursor.state["plan"]) == switched.plan


@pytest.mark.parametrize(
    ("option_id", "kept"),
    [("sector:120+130", True), ("measure_basis:grant_equivalent", False)],
    ids=["another-axis", "the-basis"],
)
async def test_a_switch_keeps_the_planners_basis_mark_unless_it_switches_the_basis(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], option_id: str, kept: bool
) -> None:
    """A reading the planner compiled to align a basis keeps that mark, on the
    new reading, its stored block and its pin's block, through a switch of its
    sector, since its basis entry is still the planner's, and the revised card
    states that basis as the planner's; a switch of the basis is the user's,
    drops the mark, and the card states that basis as the user's."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    answered = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]
    basis = answered["choices"][2]
    basis["provenance"] = "explicit"
    answered["entries"].append(
        {
            "axis": "measure_basis",
            "option_id": basis["active"],
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": basis["id"]},
        }
    )
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    search = SearchStub(answer={scope["id"]: envelope})

    switched = await _switch(
        pg_engine,
        real_schemas,
        search,
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        option_id=option_id,
        aligned=True,
    )

    state = switched.run.cursor.state
    specs = await switched.driver.turns.specs(switched.run.cursor)
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    marked = ["measure_basis"] if kept else []
    assert (stored.spec_id, list(stored.planner_axes)) == (new_pin, marked)
    assert specs[new_pin].get(PLANNER_AXES_BLOCK_KEY, []) == marked
    pinned = state["planner:pin_sources"][new_pin]["block"]
    assert pinned.get(PLANNER_AXES_BLOCK_KEY, []) == marked
    card = scope_choice_decisions({new_pin: PinSource(block=pinned, settled=True)}).get(new_pin, ())
    assert [
        d.resolved_by for d in card if d.scope is not None and d.scope.axis == "measure_basis"
    ] == ([ResolvedBy.PLANNER_DEFAULT] if kept else [ResolvedBy.USER])


@pytest.mark.parametrize(
    ("refused_by", "code", "retryable"),
    [
        ("search", "requested_option_unavailable", False),
        ("search", "transient_unavailable", True),
        ("revision", "operation_unsupported", False),
        ("revision", "discovery_unavailable", True),
        ("revision", "surface_unavailable", True),
    ],
    ids=["search", "search-transient", "revision", "revision-transient", "revision-surface"],
)
async def test_a_switch_that_cannot_apply_offers_the_plan_unchanged_with_why(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    refused_by: str,
    code: str,
    retryable: bool,
) -> None:
    """Search refusing the switch (`requested_option_unavailable`), or the
    revision failing, offers the same plan again with the failed switch, its
    code, one sentence saying why and whether asking again can succeed, which
    a Search that could not answer "just now" can,
    leaves the readings as they were and the planner's drafting state as it
    found it. A revision's refusal gives the reason its own
    (`harness.sentences.refusal_reason`); only one Search could not answer
    in time can succeed later. The part Search answered stays in the turn's
    resolve store, which holds whatever the switch's outcome."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = (
        SearchStub(
            refuse={
                scope["id"]: {
                    "code": code,
                    "title": "Requested Option Unavailable",
                    "detail": "Scope entry 1 cannot apply to this part: option_unknown.",
                    "pointer": "/input/entries/0/scope_entries/1",
                    "retryable": retryable,
                }
            }
        )
        if refused_by == "search"
        else SearchStub(answer={scope["id"]: envelope})
    )

    why = "Official development assistance would then count different providers' spending."
    refused = {
        "refused": {
            "text": "This cannot be calculated as planned, so nothing was recorded.",
            "reason": why,
            "final": False,
            "code": code,
        }
    }

    switched = await _switch(pg_engine, real_schemas, search, refused)

    planner, plan, run = switched.planner, switched.plan, switched.run
    assert len(planner.calls) == 1
    assert len(planner.revisions) == (0 if refused_by == "search" else 1)
    offered = _output_of(run, "switch")
    assert offered["revision"] == plan_revision(plan)
    assert "refusal_note" not in offered
    assert offered["switch_failed"] == {
        "choice_id": _row_id(plan, "sector"),
        "option_id": "sector:120+130",
        # A Search that did not answer states no code of its own.
        "code": SCOPE_CHOICE_NOT_APPLIED if refused_by == "search" and retryable else code,
        "reason": f"Search {'could not' if retryable else 'cannot'} count health (CRS 120) "
        "and population and reproductive health (CRS 130) "
        f"{'just now' if retryable else 'for this source'}."
        if refused_by == "search"
        else why,
        "retryable": retryable,
    }
    state = run.cursor.state
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        switched.scope["compiled"]["spec_id"]
    ]
    # The fake planner drafted with no pin sources or rounds of its own.
    assert state.get("planner:rounds") is None
    assert state.get("planner:pin_sources") is None
    resolved = [
        value for key, value in state.items() if key.startswith(RESOLVED_STATE_PREFIX) and value
    ]
    assert len(resolved) == (0 if refused_by == "search" else 1)
    assert _pauses_on_the_card(run)


@pytest.mark.parametrize(
    "planner_type",
    [_StoppedPlanner, _FinishingStoppedPlanner],
    ids=["revision-cancelled", "revision-finishes-with-the-stop"],
)
async def test_a_switch_the_user_stops_offers_the_plan_unchanged_to_answer_again(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], planner_type: type[_StoppedPlanner]
) -> None:
    """A Stop while the revision runs cancels it and offers the plan
    unchanged under `switch_cancelled`, which asking again can get past, on a
    live pause, with the drafting state, the plan, the readings and the
    blocks the revision wrote put back, a searched input's reading among
    them. A revision that finishes in the same step as the Stop ends stopped
    too."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )

    switched = await _switch(
        pg_engine,
        real_schemas,
        SearchStub(answer={scope["id"]: envelope}),
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        planner_type=planner_type,
    )

    planner, plan, run = switched.planner, switched.plan, switched.run
    assert len(planner.revisions) == 1
    offered = _output_of(run, "switch")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"] == {
        "choice_id": _row_id(plan, "sector"),
        "option_id": "sector:120+130",
        "code": SWITCH_CANCELLED,
        "reason": "You stopped this change, so the plan is as it was.",
        "retryable": True,
    }
    state = run.cursor.state
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        scope["compiled"]["spec_id"]
    ]
    assert state.get("planner:rounds") is None
    assert _SEARCHED_PIN not in await switched.driver.turns.specs(run.cursor)
    assert _pauses_on_the_card(run)


class _BrokenPlanner(_RevisingPlanner):
    """A `_RevisingPlanner` whose revision writes drafting state and a searched
    reading, then raises an error of its own with no Stop."""

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        ctx = kwargs["ctx"]
        ctx.state["planner:rounds"] = 1
        ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [
            *ctx.state[ORCHESTRATOR_SEARCHES_KEY],
            {"spec_id": _SEARCHED_PIN},
        ]
        raise RuntimeError("revision broke")


async def test_a_switch_whose_revision_raises_puts_the_state_back_and_raises(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """A revision ending on an error other than `SwitchFailedError` closes the
    turn failed at `switch` on that error, and nothing the revision wrote is
    stored: the drafting state, the plan and the readings are as the card's
    pause left them for the next turn."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )

    switched = await _switch(
        pg_engine,
        real_schemas,
        SearchStub(answer={scope["id"]: envelope}),
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        planner_type=_BrokenPlanner,
    )

    run = switched.run
    assert run.outcome is Outcome.FAILED
    failed = run.events[-1]
    assert isinstance(failed, TurnFailed) and failed.node == "switch"
    assert str(failed.error) == "revision broke"
    state = run.cursor.state
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        scope["compiled"]["spec_id"]
    ]
    assert state.get("planner:rounds") is None
    assert PlannerPlan.model_validate_json(state["plan"]) == switched.plan


def _unwinding_revision(
    unwinding: asyncio.Event, *, error: type[BaseException], linger: float = 0
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """A revision that waits until cancelled, then sets `unwinding`, waits
    `linger` seconds and raises `error` as its cleanup ends."""

    async def revision() -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await asyncio.sleep(linger)
            raise error from None
        raise AssertionError("unreachable")

    return revision


_UNWIND_ERRORS = pytest.mark.parametrize(
    "error", [asyncio.CancelledError, RuntimeError], ids=["clean-cancel", "cleanup-raises"]
)


@_UNWIND_ERRORS
async def test_a_stopped_revision_ends_the_switch_as_stopped_however_it_unwinds(
    error: type[BaseException],
) -> None:
    """A Stop cancels the revision, whose cleanup ends cleanly or raises an error of
    its own: either way the user stopped the switch, so it ends under
    `switch_cancelled`, which the `switch` node answers by putting the plan back."""
    ctx = node_ctx()
    unwinding = asyncio.Event()
    revision = _unwinding_revision(unwinding, error=error)
    switch = asyncio.ensure_future(switch_module._unless_stopped(ctx, revision()))
    await asyncio.sleep(0)
    assert ctx.stop.take("stop_1")

    with pytest.raises(SwitchFailedError) as stopped:
        await asyncio.wait_for(switch, timeout=5)
    assert stopped.value.code == SWITCH_CANCELLED
    assert not ctx.stop.armed


async def test_no_stop_is_taken_once_the_revision_has_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The revision finishes first, and a Stop arrives while the switch
    awaits its cleanup: the scope has closed, so the Stop is not taken (it
    is the turn's), and the switch applies its revision."""
    ctx = node_ctx()
    real_wait = asyncio.wait
    taken: list[bool] = []

    async def stop_during_cleanup(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_when") is None:
            # The cleanup's wait on what the race left unfinished.
            taken.append(ctx.stop.take("stop_1"))
        return await real_wait(*args, **kwargs)

    monkeypatch.setattr(switch_module.asyncio, "wait", stop_during_cleanup)

    async def revision() -> dict[str, Any]:
        return {"revised": True}

    revised = await asyncio.wait_for(switch_module._unless_stopped(ctx, revision()), timeout=5)

    assert taken == [False]
    assert revised == {"revised": True}


async def test_a_replan_that_asked_in_the_same_wait_as_the_stop_stays_paused() -> None:
    """A pick's replan ends on the planner's question to the user as the Stop
    lands. The question is the user's to answer, so `_unless_stopped` answers
    it for the planner node to pause on, and no `switch_failed` follows."""
    ctx = node_ctx()

    async def replan() -> dict[str, Any]:
        assert ctx.stop.take("stop_1")
        return _ASKED

    answered = await asyncio.wait_for(switch_module._unless_stopped(ctx, replan()), timeout=5)

    assert answered == _ASKED


@_UNWIND_ERRORS
async def test_a_switch_cancelled_while_its_stopped_revision_unwinds_stays_cancelled(
    error: type[BaseException],
) -> None:
    """A Stop cancels the revision, and the switch itself is cancelled while the
    revision's own cleanup still runs, which then ends cleanly or raises: the
    switch ends cancelled, and offers no plan as if only the Stop had ended it."""
    ctx = node_ctx()
    unwinding = asyncio.Event()
    revision = _unwinding_revision(unwinding, error=error, linger=5)

    switch = asyncio.ensure_future(switch_module._unless_stopped(ctx, revision()))
    await asyncio.sleep(0)
    assert ctx.stop.take("stop_1")
    await asyncio.wait_for(unwinding.wait(), timeout=5)
    switch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(switch, timeout=5)


class _HeldRevision(_RevisingPlanner):
    """A `_RevisingPlanner` whose revision signals `revising` and then waits
    to be cancelled."""

    def __init__(self, plans: list[dict[str, Any]], revised: dict[str, Any]) -> None:
        super().__init__(plans, revised)
        self.revising = asyncio.Event()

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        kwargs["ctx"].state["planner:rounds"] = 1
        self.revising.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")


async def test_the_first_stop_ends_only_the_switch_and_a_stop_while_paused_ends_the_turn(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """A Stop posted while the switch's revision runs reaches it through the
    conversation's Stop request: it ends the revision only, the turn pauses on
    the plan card again with `switch_cancelled` and the state as it stood,
    and the segment acts on that request once. A second Stop finds the turn paused and no live
    segment, and closes it `cancelled`."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    _, envelope = _switched_answer(scope)
    plan = _scoped_plan(pin, scope, schemas=real_schemas)
    planner = _HeldRevision([{"plan": plan.model_dump(mode="json"), "summary": {}}], {})
    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
    )
    driver = _driver(
        pg_engine,
        planner,
        search=SearchStub(answer={scope["id"]: envelope}),
        state={ORCHESTRATOR_SEARCHES_KEY: [reading.model_dump(mode="json")]},
    )
    card = (await _carded(driver)).pause
    switch = {"choice_id": _row_id(plan, "sector"), "option_id": "sector:120+130"}

    resumed = asyncio.ensure_future(driver.answer(_answer_switch(switch), to=card))
    await asyncio.wait_for(planner.revising.wait(), timeout=5)
    assert await driver.stop()
    run = await asyncio.wait_for(resumed, timeout=5)

    assert _pauses_on_the_card(run)
    assert run.pause.interrupt_id != card.interrupt_id
    assert _output_of(run, "switch")["switch_failed"]["code"] == SWITCH_CANCELLED
    assert run.cursor.state.get("planner:rounds") is None
    assert PlannerPlan.model_validate_json(run.cursor.state["plan"]) == plan

    assert not await driver.stop()
    segment_id = await driver._claim()
    try:
        stopped = await stop_turn(driver.turns, await driver.open_cursor(), segment_id=segment_id)
    finally:
        await driver.studio.drop_lease(driver.principal, driver.conversation_id, segment_id)

    assert stopped.status is TurnStatus.CANCELLED
    assert await driver.turns.open_cursor(driver.principal, driver.conversation_id) is None


# The source the planner wrote on the deployed plan's numerator, in the
# question's words, and the provenance Search states for the composite of
# bilateral and imputed multilateral aid the user switched it to.
_STATED_SOURCE = "OECD Creditor Reporting System (CRS)"
_INCLUSIVE_PROVENANCE = {
    "ref": "provenance/CRSAndONEImputedMultilateralSectors",
    "label": "CRS and ONE Imputed Multilateral Sectors",
}


class _RecordedFixtures(FixtureSurface):
    """A `FixtureSurface` recording each call's tool and arguments in `calls`."""

    def __init__(
        self, *, schemas: list[ToolSchema], responses: Mapping[str, ToolCallResult]
    ) -> None:
        super().__init__(schemas=schemas, responses=responses)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    @override
    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        self.calls.append((tool.name, arguments))
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


class _TranslatingPlanner(_Planner):
    """A `PlannerModel` drafting `plans` and revising through the real
    `harness.planner_adapter.PlannerAdapter.revise`: the proposal chain, the
    translation's discovery and the compile, over `schemas`, answering
    `responses`, with no model."""

    def __init__(
        self,
        plans: list[dict[str, Any]],
        *,
        schemas: list[ToolSchema],
        responses: Mapping[str, ToolCallResult] | None = None,
    ) -> None:
        super().__init__(plans)
        self.surface = _RecordedFixtures(schemas=schemas, responses=responses or {})
        self._adapter = PlannerAdapter(settings=Settings(), surface=self.surface)

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return await self._adapter.revise(**kwargs)


class _SwitchedOff(NamedTuple):
    """A switch off a stated source (`_switch_off_a_stated_source`): the segment
    the switch ran, the planner that revised, the pin Search answered and the
    copy of it over the plan's years."""

    run: SegmentRun
    planner: _TranslatingPlanner
    new_pin: str
    copy_pin: str


class _RefusingPickPlanner(_TranslatingPlanner):
    """A `_TranslatingPlanner` whose replan for a pick asks the user a question,
    after recording the pick and drafting state as `PlannerAdapter` does, and is
    refused once resumed."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        pick = kwargs.get("pick")
        if pick is None:
            return await super().__call__(**kwargs)
        self.calls.append(kwargs)
        ctx = kwargs["ctx"]
        if kwargs.get("answer") is not None:
            return {"refused": {"text": "No plan.", "final": False, "code": "plan_not_ready"}}
        save_decision_pick(ctx.state, pick)
        ctx.state[_DRAFTING_KEY] = "draft"
        return _ASKED


async def _switch_off_a_stated_source(
    engine: AsyncEngine, real_schemas: list[ToolSchema], *, before_switch: str | None = None
) -> _SwitchedOff:
    """The deployed failure: the planner stated the CRS as the health
    numerator's source, and the user switched it to bilateral plus imputed
    multilateral, a dataset Search names otherwise and holds from 1995. The
    switch posts the identity over every year its search answered, which the
    plan pins. Admission copies the pin over the years the plan reads, and the
    translation resolves the copy, which Search answers over those years.

    `before_switch` is what happened on the card before the switch:
    `"refused_pick_replan"` switches the window decision, which then fixes
    the period, and the planner's replan asks a question and is refused once
    answered (`_RefusingPickPlanner`)."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    new_pin, envelope = _switched_answer(scope)
    block = envelope["parts"][0]["spec"]
    block["resolved"] |= {
        "coverage": {
            "kind": "exact",
            "has_data": True,
            "window": {"start_year": 1995, "end_year": 2024},
        },
        "facets": {"provenance": _INCLUSIVE_PROVENANCE},
        "entities": [{"role": "country", "ref": "country/KEN", "label": "Kenya"}],
    }
    scoped = _scoped_plan(pin, scope, schemas=real_schemas)
    plan = PlannerPlan.model_validate(
        {
            **scoped.model_dump(),
            "asks": [_BRIEF],
            "answered": [{"ask": 0, "outputs": ["spending"]}],
            "translation": None,
            "workflow": None,
            "evidence_requirements": [
                r.model_copy(update={"source": _STATED_SOURCE})
                for r in scoped.evidence_requirements
                if isinstance(r, DatasetRequirement)
            ],
        }
    )
    if before_switch == "refused_pick_replan":
        plan = plan.model_copy(
            update={
                "decisions": tuple(
                    d.model_copy(
                        update={"fixes": DecisionFixes(period=Period(start=2015, end=2024))}
                    )
                    if d.slot == "window"
                    else d
                    for d in plan.decisions
                )
            }
        )
    identity = block["definition"]["identity"]
    copy_pin, copied = windowed_identity(identity, _PLAN_WINDOW)
    over_window = copy.deepcopy(envelope)
    copied_block = over_window["parts"][0]["spec"]
    copied_block["spec_id"] = copied_block["definition"]["spec_id"] = copy_pin
    copied_block["definition"]["identity"] = copied
    copied_block["resolved"]["coverage"]["window"]["start_year"] = 2015
    planner = (
        _RefusingPickPlanner if before_switch == "refused_pick_replan" else _TranslatingPlanner
    )(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}],
        schemas=real_schemas,
        responses={
            "resolve_definition": ToolCallResult(
                is_error=False, meta={SEARCH_ENVELOPE_META_KEY: over_window}
            )
        },
    )
    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
    )
    driver = _driver(
        engine,
        planner,
        search=SearchStub(answer={scope["id"]: envelope}),
        state={ORCHESTRATOR_SEARCHES_KEY: [reading.model_dump(mode="json")]},
    )
    card = (await _carded(driver)).pause
    if before_switch == "refused_pick_replan":
        paused = await driver.answer(_answer_switch(_PICK_SWITCH), to=card)
        assert paused.pause.kind == _PLANNER_ASK
        refused = await driver.answer({"answer": "2018"}, to=paused.pause)
        assert _output_of(refused, "planner")["switch_failed"]["code"] == "plan_not_ready"
        card = refused.pause

    run = await driver.answer(
        _answer_switch({"choice_id": _row_id(plan, "sector"), "option_id": "sector:120+130"}),
        to=card,
    )
    return _SwitchedOff(run, planner, new_pin, copy_pin)


@pytest.mark.parametrize("before_switch", [None, "refused_pick_replan"])
async def test_a_switch_moves_a_requirement_off_the_source_it_stated_and_the_plan_runs(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], before_switch: str | None
) -> None:
    """A switch off a stated source (`_switch_off_a_stated_source`) drops the
    stated source with the pin it left, so the translation takes the new pin
    with no search, the plan compiles, and its run accepts what the new
    dataset returns. The translation resolves the copy admission made over the
    plan's years, and searches nothing. A pick on the card belongs to its
    planner entry, so the revision holds none after a replan for one was
    refused following a question the planner asked."""
    run, planner, _, copy_pin = await _switch_off_a_stated_source(
        pg_engine, real_schemas, before_switch=before_switch
    )

    assert [(tool, arguments.get("spec_id")) for tool, arguments in planner.surface.calls] == [
        ("resolve_definition", copy_pin)
    ]
    offered = _output_of(run, "switch")
    assert "switch_failed" not in offered, offered.get("switch_failed")
    revised = PlannerPlan.model_validate_json(run.cursor.state["plan"])
    (dataset,) = [r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)]
    assert (dataset.source_spec_id, dataset.source) == (copy_pin, None)
    assert revised.workflow is not None
    assert [step.args.get("spec_id") for step in revised.workflow.steps] == [copy_pin]
    approved = PlannerPlan.model_validate(
        revised.model_dump()
        | {
            "status": "approved",
            "approved_revision": plan_revision(revised),
            "approved_at": datetime.now(UTC),
        }
    )
    manifest = _get_data_manifest(spec_id=copy_pin)
    manifest["citation"]["sources"] = [_INCLUSIVE_PROVENANCE]
    fetched = ToolCallResult(is_error=False, structured_content=manifest, text="fetched")

    _, outcome = await execute_approved_plan(
        approved,
        surface=_fixture_surface(real_schemas, get_data_result=fetched),
        fetch_series_stats=scripted_fetch_series_stats,
    )

    assert isinstance(outcome, Accepted), outcome


async def test_a_later_period_amendment_reads_the_switched_dataset_over_every_year_it_holds(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema]
) -> None:
    """After a switch off a stated source (`_switch_off_a_stated_source`), the
    stored reading is of the switched dataset, the planner's block for it
    states the years Search holds it for, from 1995, and the plan's copy over
    its own years is held as a copy of that dataset. A later amendment of the period
    reads the dataset whole and copies it over the new years."""
    run, _, new_pin, copy_pin = await _switch_off_a_stated_source(pg_engine, real_schemas)

    state = run.cursor.state
    shown = state["planner:pin_sources"][new_pin]["block"]
    assert shown["resolved"]["coverage"]["window"]["start_year"] == 1995
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert stored.spec_id == new_pin
    assert windowed_pins(cast(SessionState, state))[copy_pin].windowed_from == new_pin


# --- a switch onto a family the mint adds by a declared composite ----------------


def _recipient_open(block: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    """`identity` with its recipient left open, and `block` reading it under the
    spec_id it hashes to: Kenya cut from the entities the block resolves, and the
    recipient side counted at 140 parties (`role_axis_facts`), a count this test
    sets, not a recorded one."""
    identity = copy.deepcopy(identity)
    del identity["entities"]["destination"]
    identity["open_roles"] = ["destination"]
    spec_id = spec_id_for(canonicalize(identity))
    block = copy.deepcopy(block)
    block["spec_id"] = spec_id
    block["definition"] |= {"spec_id": spec_id, "identity": identity}
    resolved = block["resolved"]
    resolved["entities"] = [e for e in resolved["entities"] if e["role"] != "destination"]
    resolved["role_axis_facts"] = [
        {"role": "destination", "member_count": 140, "members_available": True}
    ]
    return block


def _dac2a_provider_blocks() -> tuple[dict[str, Any], dict[str, Any]]:
    """US net ODA to every recipient in 2019 on DAC2a, as the bilateral block the
    card shows and the inclusive block Search answers its provider switch with.

    Both are read from the block dfo-apis serves for the inclusive answer
    (`tests/fixtures/dac2a_inclusive_spec_block.json`), its recipient left open
    (`_recipient_open`). The bilateral block is its bilateral component: the
    composite's first component identity, one StatVar served, no composite flag,
    the provider row on bilateral flows as the user's words chose. The inclusive
    block is the composite compiled from that base."""
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "dac2a_inclusive_spec_block.json").read_text()
    )
    scope = recorded["resolved"]["scope_choices"]
    bilateral_component, _ = scope["compiled"]["components"]
    bilateral = _recipient_open(recorded, bilateral_component["identity"])
    inclusive = _recipient_open(recorded, recorded["definition"]["identity"])
    base = {**scope["base"], "spec_id": bilateral["spec_id"]}
    base["identity"] = bilateral["definition"]["identity"]

    shown = bilateral["resolved"]
    shown["facets"]["provenance"] = {"ref": "provenance/OECDDAC2A", "label": "OECD DAC2a"}
    shown["stat_vars"] = {
        **{k: v for k, v in shown["stat_vars"].items() if k != "sums_components"},
        "member_count": 1,
        "served_count": 1,
    }
    shown_scope = shown["scope_choices"]
    shown_scope |= {
        "id": base["spec_id"],
        "base": base,
        "entries": [],
        "compiled": {"spec_id": base["spec_id"], "identity": base["identity"], "components": []},
    }
    for choice in shown_scope["choices"]:
        choice["id"] = f"{base['spec_id']}:{choice['axis']}"
        if choice["axis"] == "provider":
            choice["active"] = "provider:bilateral"
    switched_scope = inclusive["resolved"]["scope_choices"]
    switched_scope |= {
        "id": base["spec_id"],
        "base": base,
        "compiled": {
            **switched_scope["compiled"],
            "spec_id": inclusive["spec_id"],
            "identity": inclusive["definition"]["identity"],
        },
    }
    for choice in switched_scope["choices"]:
        choice["id"] = f"{base['spec_id']}:{choice['axis']}"
    return bilateral, inclusive


async def test_a_ranking_switched_to_dac2a_s_inclusive_scope_compiles_its_sum(
    real_schemas: list[ToolSchema],
) -> None:
    """The deployed failure: a card ranking the recipients of US net ODA in 2019 on
    DAC2a's bilateral flows, switched to bilateral plus imputed multilateral. The
    inclusive answer binds every slot and serves two StatVars, the lines the mint
    adds (`sums_components`), and the plan states no reading. The revision
    re-pins the ranking's dataset and compiles: the rule reads the family summed,
    so the rank reads one measure."""
    bilateral, inclusive = _dac2a_provider_blocks()
    pin = bilateral["spec_id"]
    scoped = scope_choice_decisions({pin: PinSource(block=bilateral, settled=True)})[pin]
    measure = Decision(
        slot="measure",
        description="Which measure answers the question.",
        options=["DAC2a net ODA", "another source"],
        chosen="DAC2a net ODA",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="DAC2a reports ODA by donor and recipient.",
    )
    every = Decision(
        slot="every_recipient",
        description="Whether to rank every recipient or named ones.",
        options=["every recipient", "named recipients"],
        chosen="every recipient",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="The question asks which recipients received the most.",
        fixes=DecisionFixes(among=RankAmong.POPULATION),
    )
    oda = DatasetRequirement(
        id="oda",
        decision_ids=["measure", *(d.slot for d in scoped)],
        measure="US net ODA",
        entities=[Entity(name="United States", role="donor")],
        across=("recipient",),
        period=Period(start=2019, end=2019),
        source_spec_id=pin,
    )
    top = RankRequirement(
        id="top",
        decision_ids=["measure", "every_recipient"],
        input_id="oda",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        reduction=Reduction.SUM,
        limit=10,
    )
    plan = PlannerPlan(
        question="Which countries received the most US net ODA in 2019?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[measure, every, *scoped],
        evidence_requirements=[oda, top],
        version=1,
    )
    ctx = node_ctx()
    state = cast(SessionState, ctx.state)
    save_pin_sources(state, {pin: PinSource(block=bilateral, settled=True)})
    reading = SearchReading.from_part(
        {
            "variable": "US net ODA",
            "status": "definite",
            "spec_id": pin,
            "scope": PartScope.from_search(bilateral["resolved"]["scope_choices"]).model_dump(
                mode="json"
            ),
            "asked_roles": ["donor"],
        }
    )
    ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]
    answered = {"parts": [{"status": "definite", "spec": inclusive}]}
    planner = _TranslatingPlanner(
        [],
        schemas=real_schemas,
        responses={
            "resolve_definition": ToolCallResult(
                is_error=False, meta={SEARCH_ENVELOPE_META_KEY: answered}
            )
        },
    )
    (row,) = [d for d in plan.decisions if d.scope is not None and d.scope.axis == "provider"]

    await switch_module._switched_plan(
        ctx,
        plan,
        choice_id=choice_id(plan, row),
        option_id="provider:inclusive",
        planner=cast(Any, planner),
        surface=planner.surface,
    )

    revised = PlannerPlan.model_validate_json(ctx.state["plan"])
    assert revised.workflow is not None
    (source,) = revised.workflow.sources
    assert source.spec_id == inclusive["spec_id"]
    assert (source.stat_var_reading, source.stat_var_reading_set_by) == (
        StatVarReading.SUM,
        ReadingSetBy.RULE,
    )
    fetch = next(step for step in revised.workflow.steps if step.tool.name == "get_data")
    assert fetch.args["stat_vars"] == "sum"


# --- a switch after a planner entry that asked the user -----------------------------------

# An ISO time a turn records as it runs (a reading's `observed_at`, a resolve's time).
_RECORDED_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)")


# The wire id the loop mints for a pause.
_INTERRUPT_ID = re.compile(r"int_[0-9a-f]+")
# The id Pydantic AI gives each run of the orchestrator, in its stored history.
_RUN_ID = re.compile(r'"run_id": "[0-9a-f-]{36}"')


def _stable(value: Any) -> Any:
    """`value` as JSON, each time recorded in it put as `<time>`, each
    orchestrator run id as `<run>` and each pause's wire id as
    `<interrupt N>`, numbered in the order it first appears."""
    text = _RUN_ID.sub('"run_id": "<run>"', json.dumps(value, default=str))
    text = _RECORDED_TIME.sub("<time>", text)
    numbered: dict[str, str] = {}
    return json.loads(
        _INTERRUPT_ID.sub(
            lambda found: numbered.setdefault(found[0], f"<interrupt {len(numbered) + 1}>"), text
        )
    )


async def _hold_state(engine: AsyncEngine, cursor: Cursor, state: Mapping[str, Any]) -> None:
    """Store `state` as the paused turn's state, at the cursor's version, as
    an earlier search of the turn would have left it."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE turn_cursors SET state = CAST(:state AS json) WHERE id = :id"),
            {"state": json.dumps(dict(state)), "id": cursor.id},
        )


async def test_a_switch_after_an_entry_revises_the_plan(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], request: pytest.FixtureRequest
) -> None:
    """A turn whose planner asks which source the user means, is answered with
    the scope's pin, and drafts the plan pinning it; the card then takes a
    sector switch through the real `switch` node, which re-pins the plan
    through `PlannerAdapter.revise` with no model call. The revision reads the
    entry's stored ask and answer, so the sector decision it makes is the
    user's, and the plan pins the copy, over the plan's years, of the dataset
    Search answered the switch with. The turn's transcript, the steps the
    planner reads from it, the state and the spec blocks are the stored ones."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    scoped = _scoped_plan(pin, scope, schemas=real_schemas)
    # The plan answers the brief's one ask, as the switch's revision checks.
    plan = PlannerPlan.model_validate(
        {**scoped.model_dump(), "asks": [_BRIEF], "answered": [{"ask": 0, "outputs": ["totals"]}]}
    )
    _, envelope = _switched_answer(scope)
    # Admission copies the pin Search answered over the years the plan reads.
    copy_pin, _ = windowed_identity(
        envelope["parts"][0]["spec"]["definition"]["identity"], _PLAN_WINDOW
    )

    async def translate(call: PlannerCall) -> dict[str, object]:
        draft = load_plan(call.state)
        assert draft is not None
        assert plan.translation is not None and plan.workflow is not None
        save_plan(
            call.state,
            draft.with_translation(translation=plan.translation, workflow=plan.workflow),
        )
        return {"outcome": "ok"}

    scripted = ScriptedGenai(
        [
            reply(_ask_call("a1", candidates=[{"spec_id": pin, "label": "health aid"}])),
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=SIGNED)),
        ]
    )
    surface = FixtureSurface(schemas=[])
    adapter = PlannerAdapter(
        settings=Settings(),
        surface=surface,
        translate=translate,
        specs=_proposals(records=plan)(adapter_specs(surface=surface, translate=translate)),
        client=scripted.client,
    )
    driver = _driver(pg_engine, adapter, search=SearchStub(answer={scope["id"]: envelope}))
    with numbered_call_ids():
        asked = await driver.turn(_BRIEF)
        assert asked.pause.kind == _PLANNER_ASK, "the planner's ask pauses the turn"
        carded = await driver.answer(one_answer({"spec_id": pin}), to=asked.pause)
        assert _pauses_on_the_card(carded)
        reading = SearchReading.from_part(
            {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
        )
        await _hold_state(
            pg_engine,
            carded.cursor,
            {**carded.cursor.state, ORCHESTRATOR_SEARCHES_KEY: [reading.model_dump(mode="json")]},
        )
        switch = {"choice_id": _row_id(plan, "sector"), "option_id": "sector:120+130"}
        switched = await driver.answer(_answer_switch(switch), to=carded.pause)

    offered = _output_of(switched, "switch")
    assert "switch_failed" not in offered, offered.get("switch_failed")
    state = switched.cursor.state
    revised = PlannerPlan.model_validate_json(state["plan"])
    (dataset,) = [r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)]
    assert dataset.source_spec_id == copy_pin
    sector = next(d for d in revised.decisions if d.slot.startswith("scope_sector_"))
    assert sector.resolved_by is ResolvedBy.USER
    records = await driver.records(switched.cursor)
    expect(
        expectation_name(request),
        _stable(
            {
                "records": [{"agent": r.agent, "kind": r.kind, "body": r.body} for r in records],
                "steps": _steps(records),
                # The spec blocks are recorded on their own, as the store keeps them.
                "state": {
                    key: value for key, value in state.items() if key != ORCHESTRATOR_SPECS_KEY
                },
                "specs": await driver.turns.specs(switched.cursor),
            }
        ),
    )

"""Behavioural tests for the plan path of the turn graph: the `planner`,
`approve` and `execute` nodes the orchestrator's `plan` tool reaches, and
the helpers they share.

The turn tests run the plan path on the turn loop (`_plan_runner`): the
agent hands a brief to a planner, the turn pauses on the plan card, and the
answer resumes it into `execute` and `plan_done`. The rest are unit tests
of the node helpers. Everything runs offline against scripted fakes and a
test Postgres database: no model, no MCP server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import FrameRef
from compiler.caller_clients import MAX_CALLERS
from compiler.evidence import check_evidence, evidence_from_source
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
)
from data_mcp.server import mcp as data_mcp
from executor.classify import ToolFailure
from executor.completion import FramePublished
from executor.ledger import InMemoryRunLedger
from executor.outcomes import (
    ISSUE_REMEDIES,
    Accepted,
    EvidenceRecord,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
)
from executor.run import execute_approved_plan
from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    covers_every_row,
    every_series_selects,
    over_cap_dimension,
    over_series_limit,
    select_binding_ok,
)
from harness.conversation import current_conversation_id
from harness.decision_pick import DecisionPick
from harness.early_mint import EarlyMints
from harness.graph import answer as answer_module
from harness.graph import approve as approve_module
from harness.graph.events import PlannerStarted, StepProgress
from harness.graph.execute import (
    _failed_step,
    _mark_masked_step_failed,
    _planner_refusal,
    _replan_refusal,
    record_served_scopes,
)
from harness.graph.plan_done import authored_report, frame_unavailable_reason, plan_done_node
from harness.graph.state import (
    APPROVE_PAUSE,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    PICK_REPLAN_KEY,
    UNREADABLE_PLAN_NOTE,
    UNREADABLE_PLAN_UNPLANNED_NOTE,
    frame_id_for,
    frame_store_ref,
)
from harness.plan_display import decision_option_id
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.seeds import stale_reason
from harness.sentences import FAILURE_SENTENCES, FailureClass, failure_cause
from harness.studio_translate import failure_detail
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn_records import SearchReading, basis_line
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import AggregateOperation, GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
    ASKS_STATE_KEY,
    WINDOWED_PINS_STATE_KEY,
    AggregateRequirement,
    Coverage,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    GrowthRequirement,
    Period,
    PlanStatus,
    ResolvedBy,
    WindowedPin,
    authored,
    open_window,
    plan_revision,
    repinned,
    save_decision_pick,
    with_authored_requirements,
)
from planner.plan import Plan as PlannerPlan
from planner.records import (
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    Reduction,
    RelativePeriod,
    RollingWindow,
    Severity,
)
from planner.workflow import (
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from translator.discovery import PartFault, PartNotResolvedError

from .graph_turns import agent_read, opening_graph, visited
from .plan_cards import (
    SPEC_A,
    ScriptedExecutor,
    _analysis_result,
    _fixture_surface,
    _get_data_manifest,
    _Planner,
    _planner_plan,
)
from .plan_turns import approve_answer, plan_nodes
from .planner_host import node_ctx
from .scripted_genai import ScriptedGenai, reply, text_part
from .scripted_tools import plan_call, scripted_orchestrator
from .share_plan import ALL_DONORS, DAC_DONORS, US_DONOR, us_share_of_kenya_oda
from .split_plan import FIRST_YEAR, SPEC_DEBT_HISTORY, split_draft, stored_split_plan
from .test_turn_records import _example_part, _manifest, _served_identity
from .turn_driver import CONVERSATION, SegmentRun, TurnDriver, scripted_models

SPEC_B = "spec_" + "b" * 32
REF_A = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
REF_B = {"spec_id": SPEC_B, "content_version": "cv_" + "2" * 32}


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas. A `WorkflowStep` hashed against one of
    these passes `execute_approved_plan`'s own surface check."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


class _CallRecordingSurface:
    """Wraps a `ToolSurface`, recording every call -- `FixtureSurface` itself does not."""

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append((tool.name, dict(arguments)))
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


_BRIEF = "kenya debt"
# The pause kind of each pause a turn here answers, by the interrupt a test names.
_PAUSE_KINDS = {APPROVE_PAUSE: "approve", "planner_ask": "planner_ask"}


class _PlanTurn:
    """One conversation's plan turn on the turn loop: `driver` runs its
    segments, and `last` is the segment that ran last."""

    def __init__(self, driver: TurnDriver) -> None:
        self.driver = driver
        self.last: SegmentRun | None = None

    async def run(self, segment: Any) -> list[TurnEvent]:
        """Await `segment`, keep it as `last`, and return its events. A node
        that raised fails the turn, and its error is raised here."""
        self.last = await segment
        failed = [event for event in self.last.events if isinstance(event, TurnFailed)]
        if failed:
            raise failed[-1].error
        return self.last.events


def _plan_runner(
    engine: AsyncEngine,
    planner: Any,
    *,
    tool_surface: Any = None,
    executor: Any = None,
    early_mints: EarlyMints | None = None,
    conversation_id: str = CONVERSATION,
) -> tuple[_PlanTurn, ScriptedGenai]:
    """The plan path whose agent hands `_BRIEF` to `planner` and then answers
    in plain text, in conversation `conversation_id`, and the agent's
    scripted client; `tool_surface` runs the approved plan, and with `None`
    `execute` refuses."""
    models = scripted_models(
        orchestrator=[reply(plan_call(_BRIEF)), *(reply(text_part("done")) for _ in range(4))]
    )
    executor = executor or ScriptedExecutor()
    nodes = plan_nodes(
        scripted_orchestrator(executor=executor),
        models,
        run_ledger=InMemoryRunLedger(),
        tool_surface=tool_surface,
        executor=executor,
        planner=planner,
        early_mints=early_mints,
    )
    driver = TurnDriver(engine, graph=opening_graph(nodes), conversation_id=conversation_id)
    return _PlanTurn(driver), models.orchestrator


async def _start(turn: _PlanTurn) -> list[TurnEvent]:
    """Run the turn that asks for a plan."""
    return await turn.run(turn.driver.turn(_BRIEF))


async def _resume(turn: _PlanTurn, interrupt_id: str, response: dict[str, Any]) -> list[TurnEvent]:
    """Answer the pause the turn holds, of the kind `interrupt_id` names, with `response`."""
    assert turn.last is not None
    pause = turn.last.pause
    assert pause.kind == _PAUSE_KINDS[interrupt_id], pause.kind
    return await turn.run(turn.driver.answer(response, to=pause))


def _session_state(turn: _PlanTurn) -> dict[str, Any]:
    """The turn's state as its last segment stored it."""
    assert turn.last is not None
    return turn.last.cursor.state


async def _store_state(turn: _PlanTurn, key: str, value: Any) -> None:
    """Store `value` under `key` in the paused turn's state, for a test that
    needs state no real node would write."""
    assert turn.last is not None
    state = {**turn.last.cursor.state, key: value}
    async with turn.driver.studio.engine.begin() as conn:
        await conn.execute(
            text("UPDATE turn_cursors SET state = CAST(:state AS json) WHERE id = :id"),
            {"state": json.dumps(state), "id": turn.last.cursor.id},
        )


def _pending_interrupt(events: list[TurnEvent]) -> str | None:
    """The interrupt of the pause the turn ended on, if it ended on one."""
    kinds = {kind: interrupt for interrupt, kind in _PAUSE_KINDS.items()}
    paused = [event for event in events if isinstance(event, Paused)]
    return kinds[paused[-1].kind] if paused else None


def _output_of(events: list[TurnEvent], node: str) -> Any:
    outputs = [
        event.output
        for event in events
        if isinstance(event, NodeDone) and event.node == node and event.output is not None
    ]
    return outputs[-1] if outputs else None


def handed_back(model: ScriptedGenai) -> tuple[str, dict[str, Any]]:
    """The agent's last request as it reads it, and the last `plan` result in it."""
    request = model.requests[-1]
    (result, *_) = [
        part.function_response.response or {}
        for content in reversed(request.content_list)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    return agent_read(request), result


# --- execute's served scopes ---------------------------------------------------


class _ManifestSurface:
    """Answers `retrieval_manifest` with `manifest` for the one frame it holds."""

    def __init__(self, frame: FrameRef, manifest: dict[str, Any]) -> None:
        self.frame = frame
        self.manifest = manifest

    def retrieval_manifest(self, *, frame_ref: Mapping[str, str | None]) -> dict[str, Any] | None:
        return self.manifest if frame_ref == self.frame.model_dump() else None


def test_a_runs_retrieval_keeps_how_it_was_served_on_the_turns_scoped_reading() -> None:
    """A plan step's retrieval is kept on the reading of its spec_id with
    scope choices, as a fetch of it would be; a reading without any, and
    state no retrieval touched, are left as they are."""
    scoped = SearchReading.from_part({**_example_part(), "spec_id": SPEC_A})
    other = SearchReading(text="life", status="definite", spec_id="spec_" + "c" * 32)
    frame = FrameRef(spec_id=SPEC_A, content_version="cv_" + "1" * 32)
    outcome = Accepted(
        revision="rev",
        evidence=(
            EvidenceRecord(
                requirement_id="aid",
                step_id="aid_op",
                frame_ref=frame,
                spec_id=SPEC_A,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="CRS",
            ),
        ),
        steps=(),
        results={},
    )
    surface = _ManifestSurface(frame, _manifest(_served_identity(residuals=False)))
    state: dict[str, Any] = {
        ORCHESTRATOR_SEARCHES_KEY: [r.model_dump(mode="json") for r in (scoped, other)]
    }
    untouched: dict[str, Any] = {ORCHESTRATOR_SEARCHES_KEY: [other.model_dump(mode="json")]}

    record_served_scopes(cast("Any", state), cast("Any", surface), outcome)
    record_served_scopes(cast("Any", untouched), cast("Any", surface), outcome)

    kept, unchanged = (SearchReading.model_validate(r) for r in state[ORCHESTRATOR_SEARCHES_KEY])
    assert kept.fetch is not None and kept.fetch.served is not None
    assert kept.fetch.served.residuals("destination") == frozenset()
    assert unchanged == other
    assert untouched == {ORCHESTRATOR_SEARCHES_KEY: [other.model_dump(mode="json")]}
    assert basis_line(state[ORCHESTRATOR_SEARCHES_KEY], specs={}, plans=[]) == (
        "Counted as: health (CRS 120) and population and reproductive health (CRS 130) · "
        "Africa's countries only · "
        "gross disbursements, current USD."
    )


def test_a_windowed_copys_retrieval_is_kept_on_the_reading_it_was_copied_from() -> None:
    """A search over 2024 found the part; the plan reads 2015 to 2024 from
    its windowed copy. The copy's retrieval is kept on the search's reading,
    and the plan's source names the reading it was copied from, so the
    basis line states how the copy's figures were counted."""
    copy_id = "spec_" + "d" * 32
    searched = SearchReading.from_part({**_example_part(), "spec_id": SPEC_A})
    frame = FrameRef(spec_id=copy_id, content_version="cv_" + "1" * 32)
    outcome = Accepted(
        revision="rev",
        evidence=(
            EvidenceRecord(
                requirement_id="aid",
                step_id="aid_op",
                frame_ref=frame,
                spec_id=copy_id,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="CRS",
            ),
        ),
        steps=(),
        results={},
    )
    surface = _ManifestSurface(frame, _manifest(_served_identity(residuals=False), spec_id=copy_id))
    pin = WindowedPin(spec_id=copy_id, identity={}, windowed_from=SPEC_A)
    state: dict[str, Any] = {
        ORCHESTRATOR_SEARCHES_KEY: [searched.model_dump(mode="json")],
        WINDOWED_PINS_STATE_KEY: [pin.model_dump(mode="json")],
    }

    record_served_scopes(cast("Any", state), cast("Any", surface), outcome)

    plan = {"sources": [{"spec_id": copy_id, "windowed_from": SPEC_A}]}
    assert basis_line(state[ORCHESTRATOR_SEARCHES_KEY], specs={}, plans=[plan]) == (
        "Counted as: health (CRS 120) and population and reproductive health (CRS 130) · "
        "Africa's countries only · "
        "gross disbursements, current USD."
    )


# --- execute's run: progress relay and cancellation ---------------------------


_PREPARING = "Preparing the dataset (in progress, 3s)"
_RECEIVED = "Result received; checking it"


def _step_progress(event: object) -> dict[str, Any] | None:
    return dataclasses.asdict(event) if isinstance(event, StepProgress) else None


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
            on_progress(ProgressReport(progress=3, message=_PREPARING))
            self.waiting = asyncio.current_task()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def _approved_run(
    engine: AsyncEngine, surface: ToolSurface, *, observe: Callable[[TurnEvent], None]
) -> list[TurnEvent]:
    """The events of an approved scripted plan's run against `surface`, each
    told to `observe` as the segment emits it."""
    runner, _ = _plan_runner(engine, ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)
    runner.driver.observe = observe
    return await _resume(
        runner, APPROVE_PAUSE, approve_answer(_output_of(events, "planner")["revision"])
    )


async def test_execute_relays_a_report_while_its_step_is_still_running(
    pg_engine: AsyncEngine,
) -> None:
    """The surface answers only once the test has seen the report, so the
    report must reach the event stream while the call is still waiting. Once
    a call answers, its step reports that, so the step's line stops showing
    the server's wait while the run goes on. The reader keeps the latest
    report of each step still waiting for it, as Studio's stream does."""
    surface = _ReportsThenWaits()
    waiting: list[bool] = []
    reports: dict[str, StepProgress] = {}
    seen: list[StepProgress] = []

    def read() -> None:
        seen.extend(reports.values())
        reports.clear()
        if not surface.release.is_set():
            waiting.append(surface.waiting is not None and not surface.waiting.done())
            surface.release.set()

    def observe(event: TurnEvent) -> None:
        if isinstance(event, StepProgress):
            if not reports:
                asyncio.get_running_loop().call_soon(read)
            reports[event.step_id] = event

    async with asyncio.timeout(10):
        events = await _approved_run(pg_engine, surface, observe=observe)

    assert waiting == [True]
    progress = [_step_progress(event) for event in seen if _step_progress(event)]
    retrieve = {
        "step_id": "retrieve",
        "tool": "get_data",
        "position": 1,
        "count": 2,
        "description": "life expectancy at birth in Kenya, 1960 to 2023.",
        "gives_answer": False,
    }
    totals = {
        "step_id": "totals",
        "tool": "aggregate",
        "position": 2,
        "count": 2,
        "description": (
            "The average of life expectancy at birth, grouped by year. "
            "Result: one figure per year, 1960-2023."
        ),
        "gives_answer": True,
    }
    assert progress == [
        {**retrieve, "message": _PREPARING},
        {**retrieve, "message": _RECEIVED},
        {**totals, "message": _RECEIVED},
    ]
    assert _emitted_by(events, next(e for e in events if _step_progress(e))) == "execute"
    assert "plan_done" in visited(events)


def _emitted_by(events: list[TurnEvent], event: object) -> str:
    """The node that emitted `event`: the next node to finish after it."""
    after = events[events.index(event) :]
    return next(e.node for e in after if isinstance(e, NodeDone))


async def test_a_plan_whose_card_places_fail_still_runs_with_unplaced_steps(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The plan card places only label the step lines, so a failure building
    them is logged and the approved plan runs with every step unplaced."""

    def fails(plan: object) -> dict[str, object]:
        raise ValueError("no card for this plan")

    monkeypatch.setattr("harness.graph.execute._card_places", fails)

    with caplog.at_level("ERROR", logger="harness.graph.execute"):
        async with asyncio.timeout(10):
            seen = await _approved_run(pg_engine, ScriptedSurface(), observe=lambda event: None)

    progress = [_step_progress(event) for event in seen if _step_progress(event)]
    assert progress
    assert all(report["position"] is None for report in progress)
    assert "plan_done" in visited(seen)
    (logged,) = [r for r in caplog.records if r.name == "harness.graph.execute"]
    assert logged.exc_info is not None


# --- version-bound approval -----------------------------------------------------


@pytest.mark.parametrize(
    "wrong",
    [
        lambda revision: approve_answer("not-a-real-revision"),
        lambda revision: {"approved": revision},
        lambda revision: {"approved": revision, "at": "2026-10-01T09:30:00"},
        lambda revision: {"approved": revision, "at": "yesterday"},
    ],
    ids=["stale_revision", "no_time", "time_without_offset", "unreadable_time"],
)
async def test_stale_version_approval_routes_to_amend(
    pg_engine: AsyncEngine, wrong: Callable[[str], Any]
) -> None:
    """A wrong answer, one naming another revision or stating no
    timezone-aware approval time, routes to amend; `ScriptedPlannerModel`
    redrafts the same canned steps as the next version, bumping `version`
    and `previous_version` the way any `PlannerModel` must."""
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(pg_engine, ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_PAUSE
    first = _output_of(events, "planner")
    assert first["new_plan"] is True

    events = await _resume(runner, APPROVE_PAUSE, wrong(first["revision"]))

    # Wrong answer: nothing executed; the planner redrafted to version 2 and
    # a fresh approval interrupt is pending.
    assert surface.calls == []
    assert _pending_interrupt(events) == APPROVE_PAUSE
    amended = _output_of(events, "planner")
    assert amended["version"] == 2
    assert amended["previous_version"] == 1
    assert amended["status"] == "pending_approval"
    assert amended["revision"] != first["revision"]
    assert "new_plan" not in amended

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(amended["revision"]))
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "execute")["plan_version"] == 2


async def test_a_plan_approved_unchanged_opens_a_span_and_logs_the_facts(
    pg_engine: AsyncEngine, exported_spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    """Version 1 approved unchanged: `approve`'s `plan.approved` span and log
    line record the version and that no amendment preceded it."""
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(pg_engine, ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)

    assert _pending_interrupt(events) == APPROVE_PAUSE
    plan_payload = _output_of(events, "planner")
    assert plan_payload["version"] == 1

    with caplog.at_level("INFO", logger="harness.graph.approve"):
        events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_payload["revision"]))

    assert _output_of(events, "execute")["plan_version"] == 1

    spans = [s for s in exported_spans.get_finished_spans() if s.name == "plan.approved"]
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes is not None
    assert attributes["dfo.plan_version"] == 1
    assert attributes["dfo.amended"] is False

    record = next(r for r in caplog.records if "plan approved" in r.getMessage())
    assert "plan_version=1" in record.getMessage()
    assert "amended=False" in record.getMessage()


# --- final frames: what `plan_done` describes ------------------------------------


_FRAME_DO_NOT_RETRY = "this step's data view could not be prepared"


@pytest.mark.parametrize(
    ("retry_mode", "reason"),
    [
        pytest.param(["retry_after_delay"], _FRAME_DO_NOT_RETRY, id="list"),
        pytest.param({"mode": 1}, _FRAME_DO_NOT_RETRY, id="dict"),
        pytest.param(None, _FRAME_DO_NOT_RETRY, id="none"),
        pytest.param(
            "retry_after_delay", "this step's data view is temporarily unavailable", id="string"
        ),
    ],
)
def test_a_frame_unavailable_reason_reads_a_retry_mode_of_any_type(
    retry_mode: Any, reason: str
) -> None:
    assert frame_unavailable_reason({"retry_mode": retry_mode}) == reason


def _planner_plan_with_intermediate_step(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real, three-step plan: get_data -> totals (aggregate, an
    intermediate step) -> change (growth, the one terminal step).

    `_planner_plan`'s two-step plan has one non-terminal step, so a failure
    after it leaves one frame. A third step lets two steps, one of them
    genuinely intermediate, complete before the terminal one fails.
    """
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="totals",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="totals",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A},
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "totals", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


_CHANGE_REFUSAL = ToolCallResult(
    is_error=True,
    structured_content={
        "code": "entity_ambiguous",
        "retry_mode": "correct_and_retry",
        "message": "more than one entity matches 'Kenya'",
        "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
    },
    text="refused",
)


def _stopping_at_growth_surface(real_schemas: list[ToolSchema]) -> FixtureSurface:
    """get_data and totals (aggregate) succeed; change (growth) always
    refuses. A run against this surface completes a retrieval and a derived
    step before exhausting the replan budget."""
    named = {schema.name: schema for schema in real_schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": _CHANGE_REFUSAL,
        },
    )


def _planner_plan_with_two_terminal_steps(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real plan with two independent terminal steps reading the same
    retrieval directly: `totals` (aggregate) and `change` (growth), at
    different positions in `outcome.steps` (`workflow.steps` is
    `[get_data, totals, change]`, so `totals` sits at index 1 and `change`
    at index 2). `Translation.terminal` is a tuple with `min_length=1`, not
    `max_length=1` (`planner.operations.py`), so a plan answering through
    more than one terminal step is a real shape, not a hypothetical one --
    and `is_final_step` must mark both `totals` and `change` final, since
    neither is any more the question's answer than the other."""
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="spending",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt totals and growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="get_data",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["totals", "change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A},
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["totals", "change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


@pytest.mark.parametrize(
    "build_plan",
    [
        # totals feeds change, the one terminal step.
        _planner_plan_with_intermediate_step,
        # totals is a terminal step of its own, finished beside the failed change.
        _planner_plan_with_two_terminal_steps,
    ],
    ids=["intermediate", "sibling_terminal"],
)
async def test_a_stopped_plan_hands_back_no_derived_steps_figures(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], build_plan: Callable[..., PlannerPlan]
) -> None:
    """change fails past the replan budget, leaving get_data's and totals'
    frames in `completed`. totals' derived frame is never published on a
    stopped run, so neither its frame nor its digest reaches the agent,
    whether totals fed change or was a terminal step of its own; the
    retrieval's digest still does, and `plan_done` lists no frame to chart."""
    plan = build_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    executor = ScriptedExecutor()
    runner, model = _plan_runner(
        pg_engine,
        planner,
        tool_surface=_stopping_at_growth_surface(real_schemas),
        executor=executor,
    )
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))
    assert _pending_interrupt(events) == APPROVE_PAUSE
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(replanned["revision"]))

    executed = _output_of(events, "execute")
    assert executed["failed_step"]["step_id"] == "change"
    assert [step["step_id"] for step in executed["frame_ref_steps"]] == ["get_data"]
    assert [step["digest"] for step in executed["step_summaries"]] == ["fetched"]
    assert executor.calls == []
    _, body = handed_back(model)
    assert body["frames"] == []
    assert body["step_summaries"] == ["fetched"]


async def test_two_terminal_steps_both_come_back_to_the_agent(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """`totals` (step index 1) and `change` (step index 2) are both
    terminal. Finality inferred from `max(step)` would mark only `change`
    final and drop `totals`; `execute`'s Accepted branch states
    `is_final_step` on each entry, so `plan_done` describes and lists both."""
    plan = _planner_plan_with_two_terminal_steps(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    named = {schema.name: schema for schema in real_schemas}
    surface = FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": ToolCallResult(
                is_error=False,
                # A distinct spec_id: totals and change must be different frames.
                structured_content=_analysis_result(spec_id="ops_" + "4" * 32),
                text="computed",
            ),
        },
    )
    executor = ScriptedExecutor()
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface, executor=executor)
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert len(_output_of(events, "execute")["frame_refs"]) == 2
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "describe_frame"]
    _, body = handed_back(model)
    assert len(body["frames"]) == 2


# --- select_binding_ok: the select checks `make_charts` runs before a write ---


_MULTISERIES_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
        {
            "name": "observationAbout",
            "plural": True,
            "values": [
                f"country/{code}" for code in ("BLZ", "CRI", "GTM", "HND", "NIC", "PAN", "SLV")
            ],
        },
    ],
    "series_count": 7,
}


def _population_view(*, plural: str) -> dict[str, Any]:
    """The population frame's view: one `stat_var`, seven countries, and
    a `unit` column holding `USDollar` on some rows and null on others, which
    apis reports as plural with the one non-null value listed. `plural`
    names a further dimension marked plural with two listed values."""
    return {
        "dimensions": [
            {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
            {
                "name": "observationAbout",
                "plural": True,
                "values": ["country/GTM", "country/HND"],
                "total_count": 7,
            },
            {"name": "unit", "plural": True, "values": ["USDollar"], "total_count": 1},
            {"name": plural, "plural": True, "values": ["a", "b"], "total_count": 2},
        ]
    }


@pytest.mark.parametrize("facet", ["measurementMethod", "provenance"])
def test_stat_card_may_leave_plural_facets_unpinned(facet: str) -> None:
    """apis refuses an unbound facet only when it splits a series, so a
    population card pinning the country passes with `unit` = `{USDollar,
    null}` and a second plural facet column left unbound."""
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    view = _population_view(plural=facet)
    assert select_binding_ok(type_="stat_card", select=select, view=view) is None


def test_stat_card_still_refuses_an_unpinned_non_facet_plural_dimension() -> None:
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    reason = select_binding_ok(
        type_="stat_card", select=select, view=_population_view(plural="sex")
    )
    assert reason is not None
    assert reason["code"] == "select_invalid"


# --- the select_binding_ok array-dimension cap, per component type -----------


def test_select_binding_ok_permits_a_table_with_two_array_dimensions() -> None:
    """The platform's catalogue grammar allows `table` to carry arrays on
    more than one dimension (their cross-product, at most 24 series); only
    `line_chart` and `bar_chart` are limited to at most one array-valued
    dimension."""
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert select_binding_ok(type_="table", select=select, view={}) is None


def test_select_binding_ok_still_rejects_a_line_chart_with_two_array_dimensions() -> None:
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert select_binding_ok(type_="line_chart", select=select, view={}) is not None


def _countries(count: int) -> list[str]:
    return [f"country/C{index:02d}" for index in range(count)]


# The platform's release document as presentation-mcp's tests record it.
_RELEASE = Path(__file__).parents[2] / "presentation-mcp/tests/fixtures/release.json"


def test_a_line_chart_lists_no_more_values_than_the_catalogue_allows() -> None:
    select_schema = json.loads(_RELEASE.read_text())["intent_schemas_tolerant"]["line_chart"][
        "properties"
    ]["select"]
    (array,) = [
        branch
        for branch in select_schema["additionalProperties"]["anyOf"]
        if branch["type"] == "array"
    ]
    cap = array["maxItems"]

    allowed = over_series_limit(
        type_="line_chart", select={"observationAbout": _countries(cap)}, view={}
    )
    over = over_series_limit(
        type_="line_chart", select={"observationAbout": _countries(cap + 1)}, view={}
    )

    assert cap == LINE_CHART_MAX_VALUES
    assert (allowed, over) == (False, True)


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
def test_a_select_is_over_the_series_limit_past_24_series(type_: str) -> None:
    """apis refuses a component past 24 series even though the catalogue
    grammar's `maxItems` allows an array of 30, and a table's cross product
    counts against it."""
    assert not over_series_limit(
        type_=type_, select={"observationAbout": _countries(24), "stat_var": "a"}, view={}
    )
    assert over_series_limit(
        type_=type_, select={"observationAbout": _countries(25), "stat_var": "a"}, view={}
    )
    crossed = {"observationAbout": _countries(5), "stat_var": ["a", "b", "c", "d", "e"]}
    assert over_series_limit(type_="table", select=crossed, view={})


def _counted_view(**total_counts: int) -> dict[str, Any]:
    """A `describe_frame` view whose dimensions carry the apis service's `total_count`."""
    return {
        "dimensions": [
            {"name": name, "plural": count > 1, "values": [], "total_count": count}
            for name, count in total_counts.items()
        ]
    }


@pytest.mark.parametrize(
    ("type_", "select", "view"),
    [
        ("bar_chart", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30)),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30)),
        (
            "line_chart",
            {"donor": [f"donor/{index}" for index in range(5)]},
            _counted_view(stat_var=5, donor=30),
        ),
        (
            "table",
            {"donor": [f"donor/{index}" for index in range(20)]},
            _counted_view(stat_var=2, donor=30),
        ),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30, unit=30)),
        (
            "line_chart",
            {"stat_var": "a", "donor": [f"donor/{index}" for index in range(9)]},
            _counted_view(stat_var=1, donor=30),
        ),
        (
            "table",
            {"stat_var": "a", "unit": ["USDollar", "EUR"]},
            _counted_view(stat_var=1, donor=30, unit=3, measurementMethod=5),
        ),
    ],
)
def test_over_series_limit_matches_dfo_apis_series_estimate(
    type_: str, select: dict[str, Any], view: dict[str, Any]
) -> None:
    """The harness computes the apis service's own estimate (`_check_series_estimate`
    in dfo-data-tools' `assembler/resolve.py`): over every coordinate
    dimension, the bound array's distinct values, 1 for a single bound value, and
    `max(total_count, 1)` for an unbound dimension, on every component type
    the assembler resolves. Each case here is one apis refuses on that
    estimate: an unbound plural dimension, a `line_chart` whose array
    lists more than eight distinct values or multiplies past 24, and an
    unbound dimension multiplying a table's array. An unbound facet column
    counts 1 and a bound one counts its values."""
    assert over_series_limit(type_=type_, select=select, view=view)


def test_over_cap_dimension_names_the_one_dimension_that_multiplies() -> None:
    view = _counted_view(stat_var=1, financingProvider=86, financingRecipient=1)
    wide = _counted_view(stat_var=5, financingProvider=86, financingRecipient=1)

    assert over_cap_dimension(select={}, view=view) == ("financingProvider", 86)
    assert over_cap_dimension(
        select={"financingProvider": ["a", "b"], "stat_var": "x"}, view=wide
    ) == ("financingProvider", 2)
    assert over_cap_dimension(select={}, view=wide) is None
    repeated = {"financingProvider": ["donor/01"] * 30, "stat_var": "x"}
    assert over_cap_dimension(select=repeated, view=wide) is None
    facets = _counted_view(stat_var=1, unit=1, financingProvider=1)
    assert over_cap_dimension(select={"unit": ["USD", "EUR"], "stat_var": "x"}, view=facets) is None


def test_over_series_limit_counts_an_unbound_dimension_at_its_total_count() -> None:
    """An unbound dimension holding 24 values is within the cap, and binding
    it to a single value brings 30 down to 1. Scripted fixtures list bare
    dimension names, or entries without `total_count`; neither counts past
    one for want of the figure."""
    for select, view in [
        ({"stat_var": "a"}, _counted_view(stat_var=1, donor=24)),
        ({"stat_var": "a", "donor": "donor/1"}, _counted_view(stat_var=1, donor=30)),
        ({"stat_var": "a"}, _counted_view(stat_var=1, donor=24, unit=30, provenance=30)),
        ({"stat_var": "a"}, {"dimensions": ["stat_var", "observationAbout"]}),
        ({"stat_var": "a"}, _MULTISERIES_VIEW),
        # A repeated member resolves to one series, whatever the chart type.
        ({"stat_var": "a", "donor": ["donor/1"] * 30}, _counted_view(stat_var=1, donor=40)),
    ]:
        for type_ in ("line_chart", "bar_chart", "table"):
            assert not over_series_limit(type_=type_, select=select, view=view), view


def test_select_binding_ok_refuses_a_dimension_the_view_does_not_declare() -> None:
    """apis estimates series over its own dimensions only, and refuses a
    select binding any other name as an invalid binding; the harness gives
    the same reason."""
    select = {"stat_var": "a", "donor": [f"donor/{index}" for index in range(30)]}
    reason = select_binding_ok(type_="bar_chart", select=select, view=_counted_view(stat_var=1))
    assert reason is not None
    assert (reason["code"], reason["retry_mode"]) == ("select_invalid", "correct_and_retry")
    assert reason["fix"] == {"dimensions": ["stat_var"]}
    flow = {"dimensions": ["stat_var", "financingProvider", "financingRecipient"]}
    reason = select_binding_ok(type_="table", select={"observationAbout": "country/MWI"}, view=flow)
    assert reason is not None
    assert reason["fix"] == {"dimensions": ["stat_var", "financingProvider", "financingRecipient"]}


@pytest.mark.parametrize(
    ("type_", "view", "selects"),
    [
        ("stat_card", {"dimensions": [{"name": "donor", "values": ["a"]}]}, [{"donor": "a"}]),
        # A dimension of one value before one of several.
        (
            "bar_chart",
            {
                "dimensions": [
                    {"name": "recipient", "values": ["k", "u"], "total_count": 2},
                    {"name": "donor", "values": ["a"], "total_count": 1},
                ]
            },
            [{"donor": "a"}, {"recipient": ["k", "u"]}],
        ),
        # A view that lists fewer values than the dimension holds names no series set.
        (
            "table",
            {"dimensions": [{"name": "donor", "values": ["a", "b"], "total_count": 3}]},
            [],
        ),
        # A facet may hold rows with no value, which no select on it names.
        ("table", {"dimensions": [{"name": "unit", "values": ["USD"], "total_count": 1}]}, []),
        # A select listing more lines than a line_chart draws is offered,
        # and make_charts cuts it.
        (
            "line_chart",
            {"dimensions": [{"name": "donor", "values": list("abcdefghi"), "total_count": 9}]},
            [{"donor": list("abcdefghi")}],
        ),
    ],
)
def test_every_series_selects_bind_one_dimension_the_view_lists_in_full(
    type_: str, view: dict[str, Any], selects: list[dict[str, Any]]
) -> None:
    assert every_series_selects(type_=type_, view=view) == selects


def test_a_select_covers_every_row_only_when_none_is_empty_on_its_dimension() -> None:
    """The view counts a dimension's values without its empty cells, so only
    the rows say whether a select on it names every series."""
    rows = [{"donor": "a", "recipient": "k"}, {"donor": None, "recipient": "total"}, {"donor": "a"}]
    assert covers_every_row({"recipient": ["k", "total"]}, rows[:2])
    assert not covers_every_row({"donor": "a"}, rows[:2])
    assert not covers_every_row({"recipient": ["k", "total"]}, rows)


def test_select_binding_ok_rejects_a_non_mapping_select() -> None:
    """A `select_json` that parsed to a JSON array or scalar is a rejection
    reason here, not an `AttributeError` on `.items()`: `make_charts` hands
    it whatever `json.loads` made of the model's `select_json`."""
    assert select_binding_ok(type_="table", select=["not", "a", "mapping"], view={}) is not None


# --- the Plan objects stay real ------------------------------------------------


async def test_plan_amendment_lineage_is_real(pg_engine: AsyncEngine) -> None:
    """The plan carried through amend cycles is a valid `planner.plan.Plan`."""
    runner, _ = _plan_runner(pg_engine, ScriptedPlannerModel())
    await _start(runner)
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(99))

    output = _output_of(events, "planner")
    plan_fields = {k: v for k, v in output.items() if k not in ("summary", "revision")}
    plan = PlannerPlan.model_validate(plan_fields)
    assert plan.version == 2
    assert plan.previous_version == 1
    assert plan.decisions[0].chosen in plan.decisions[0].options


# --- the planner node against a planner model --------------------------------


async def test_the_planner_node_validates_the_planners_plan_and_writes_it_to_state(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """The planner returns `{"plan": <planner.plan.Plan JSON>, "summary": ...}`;
    the node validates it as a `planner.plan.Plan`, and emits `summary` and
    `revision` beside it -- both the record and the wire keys, not one
    constructed from parts the way the frozen adapter's protocol was.
    """
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner(
        [{"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}]
    )
    runner, _ = _plan_runner(pg_engine, planner)

    events = await _start(runner)

    output = _output_of(events, "planner")
    assert output["summary"] == {"version": 1}
    assert output["revision"] == plan_revision(plan)
    plan_fields = {k: v for k, v in output.items() if k not in ("summary", "revision", "new_plan")}
    validated = PlannerPlan.model_validate(plan_fields)
    assert validated == plan


async def test_the_planner_node_raises_on_a_malformed_plan_rather_than_constructing_one(
    pg_engine: AsyncEngine,
) -> None:
    """A draft the planner could not itself validate must not reach the approve gate."""
    planner = _Planner([{"plan": {"decisions": "not-a-list"}, "summary": {}}])
    runner, _ = _plan_runner(pg_engine, planner)

    with pytest.raises(ValidationError):
        await _start(runner)


class _GatedPlanner:
    """A `PlannerModel` that drafts only once its caller has seen the
    planner node's start event, then pauses on a question for the user."""

    def __init__(self) -> None:
        self.seen = asyncio.Event()
        self.drafted = False

    async def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        await asyncio.wait_for(self.seen.wait(), timeout=5)
        self.drafted = True
        return _ASKED


async def test_the_planner_node_announces_a_draft_before_the_planner_returns(
    pg_engine: AsyncEngine,
) -> None:
    """The node's first event reaches the runner stream while the planner is
    still drafting, so Studio can say a plan is under way."""
    planner = _GatedPlanner()
    runner, _ = _plan_runner(pg_engine, planner)
    started = []

    def observe(event: TurnEvent) -> None:
        if isinstance(event, PlannerStarted):
            started.append({"revising": event.revising})
            planner.seen.set()

    runner.driver.observe = observe
    await _start(runner)

    assert started == [{"revising": False}]
    assert planner.drafted


async def test_a_refused_redraft_over_an_unreadable_plan_says_the_plan_was_discarded(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """The user approved a plan the release cannot read, and the planner
    refuses to draft it again. No card follows to say what became of the plan
    they approved, so the planner's answer the agent reads opens with it."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner(
        [
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"refused": {"text": "The draft budget is spent.", "final": True}},
        ]
    )
    runner, model = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    await _store_state(runner, "plan", json.dumps({"question": "kenya debt", "tool_calls": []}))

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(1))

    refused = _output_of(events, "planner")
    assert refused == {
        "text": f"{UNREADABLE_PLAN_UNPLANNED_NOTE} The draft budget is spent.",
        "final": True,
        # The scripted refusal states no code.
        "code": None,
    }
    assert _pending_interrupt(events) is None
    assert (
        f"The planner could not make a plan: {UNREADABLE_PLAN_UNPLANNED_NOTE} The draft "
        "budget is spent." in agent_read(model.requests[-1])
    )


_PLANNER_ASK = "planner_ask"


# What a `PlannerModel` answers for a planner run that ended on a question for
# the user, which the `planner` node pauses on under `_PLANNER_ASK`.
_ASKED = {"asked": {"call_id": _PLANNER_ASK, "args": {"message": "Which years?"}}}


class _AskingOnAmendPlanner(_Planner):
    """A `PlannerModel` that drafts its first plan outright and asks the user
    a question on every later call that no answer resumes."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) > 1 and kwargs.get("answer") is None:
            return _ASKED
        return self._plans[0]


async def test_a_planner_ask_resumed_over_an_unreadable_plan_drafts_from_the_start(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """The planner asks the user a question while amending a plan, and the
    plan on record is unreadable when the answer resumes it. The `planner`
    node finds that itself: nothing is left to amend, so the planner is handed
    no amendment and no prior plan, and the card says why."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _AskingOnAmendPlanner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    events = await _resume(runner, APPROVE_PAUSE, {"approved": None, "amendment": "only 2020"})
    assert _pending_interrupt(events) == _PLANNER_ASK
    assert planner.calls[1]["amendment"] == "only 2020"
    await _store_state(runner, "plan", json.dumps({"question": "kenya debt", "tool_calls": []}))

    events = await _resume(runner, _PLANNER_ASK, {"answer": "2020"})

    assert _pending_interrupt(events) == APPROVE_PAUSE
    assert planner.calls[2]["amendment"] is None
    assert planner.calls[2]["prior_plan"] is None
    redrafted = _output_of(events, "planner")
    assert redrafted["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert redrafted["restarted"] is True
    assert "new_plan" not in redrafted


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


async def test_an_approval_for_a_superseded_revision_routes_to_amend(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """The `approve` node keys a model-drafted plan on
    `plan_revision(plan)`, not the version integer. An answer naming a
    revision that does not match `ctx.state["plan"]` -- because an
    amendment already replaced it -- routes to "amend" the same way a stale
    version does, even when the superseding plan happens to share the
    same version number (`propose_plan` can start a fresh plan at version 1
    for what it reads as a different question, so two unrelated plans can
    both be version 1 in one conversation).
    """
    plan_v1 = _planner_plan(question="kenya debt", schemas=real_schemas)
    plan_v2 = _planner_plan(question="kenya debt, revised", schemas=real_schemas)
    planner = _Planner(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    executor = ScriptedExecutor()
    runner, _ = _plan_runner(
        pg_engine, planner, tool_surface=_fixture_surface(real_schemas), executor=executor
    )
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_PAUSE
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    # A wrong answer routes to amend, which redrafts -- an unrelated plan
    # that also happens to be version 1.
    events = await _resume(runner, APPROVE_PAUSE, approve_answer("not-a-real-revision"))
    assert _pending_interrupt(events) == APPROVE_PAUSE
    amended = _output_of(events, "planner")
    assert amended["version"] == 1
    revision_2 = amended["revision"]
    assert revision_2 != revision_1

    # The first card's revision no longer names what's pending: superseded,
    # not approved -- nothing executes and a fresh approval is still pending.
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision_1))
    assert executor.calls == []
    assert _pending_interrupt(events) == APPROVE_PAUSE
    assert _output_of(events, "planner")["revision"] == revision_2

    # The correct, current revision approves cleanly: `approve()` succeeds,
    # the gate is passed (no fresh approval interrupt), and the approved
    # workflow's steps ran through `execute_approved_plan` against the
    # tool surface: the executor only describes the final frame.
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision_2))
    assert _pending_interrupt(events) is None
    assert [call["tool"] for call in executor.calls] == ["describe_frame"]
    assert _output_of(events, "execute")["plan_version"] == 1


# The scripted plan as the code at commit 2563638 serialised it, beside the
# revision that code computed (`tests/planner/test_revision_format.py`). Its
# workflow is pinned to the schemas `ScriptedSurface` served at that commit,
# which it still serves.
_BASE_FORMAT_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["aggregate"]


async def test_the_scripted_surface_declares_the_schemas_the_saved_plan_names() -> None:
    declared = {schema.name: schema for schema in await ScriptedSurface().list_tools()}

    for step in _BASE_FORMAT_PLAN["plan"]["workflow"]["steps"]:
        assert declared[step["tool"]["name"]].input_schema_hash == step["input_schema_hash"]


async def test_a_pending_plan_saved_at_the_base_revision_format_is_approved_unamended(
    pg_engine: AsyncEngine,
) -> None:
    """The card a user holds names the revision computed when the plan was
    saved. A plan saved without the fields its models have since gained
    still hashes to that revision, so the answer approves it and it runs."""
    planner = _Planner([{"plan": _BASE_FORMAT_PLAN["plan"], "summary": {}}])
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_PAUSE
    assert _output_of(events, "planner")["revision"] == _BASE_FORMAT_PLAN["revision"]
    # The session holds the plan exactly as it was saved.
    await _store_state(runner, "plan", json.dumps(_BASE_FORMAT_PLAN["plan"]))

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(_BASE_FORMAT_PLAN["revision"]))

    assert _pending_interrupt(events) is None
    assert len(planner.calls) == 1
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "execute")["plan_version"] == 1
    finished = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    assert finished.status is PlanStatus.COMPLETED
    assert finished.approved_revision == _BASE_FORMAT_PLAN["revision"]


async def test_an_approved_plan_saved_at_the_base_revision_format_executes() -> None:
    """`approved_revision` was computed when the plan was approved;
    `Plan.start_execution` compares it with the revision computed now."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
    )

    async def published(frame: FrameRef) -> FramePublished:
        return FramePublished(replay_key="rpk_" + "e" * 64)

    finished, outcome = await execute_approved_plan(
        approved,
        surface=ScriptedSurface(),
        publisher=published,
        fetch_series_stats=scripted_fetch_series_stats,
    )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED


async def test_a_saved_plan_whose_retrieval_posts_its_spec_id_alone_runs_as_saved() -> None:
    """The saved retrieval step carries `spec_id` alone, as `get_data` takes
    it, so the call is made as saved."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
    )
    surface = _CallRecordingSurface(ScriptedSurface())

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome
    assert surface.calls[0] == ("get_data", {"spec_id": "spec_" + "f" * 32})


# A population rank as the code at commit 2563638 compiled it against the tool
# schemas the servers declared then.
_BASE_FORMAT_LIVE_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["rank"]


async def test_an_approved_plan_compiled_against_an_older_get_data_is_refused_for_reapproval(
    real_schemas: list[ToolSchema],
) -> None:
    """Its retrieval step names the argument schema `get_data` declared when it
    was compiled. The run stops before any call, with the step and the remedy
    named, and the plan keeps its revision."""
    saved = _BASE_FORMAT_LIVE_PLAN["plan"]
    (retrieval,) = (s for s in saved["workflow"]["steps"] if s["tool"]["name"] == "get_data")
    approved = PlannerPlan.model_validate(
        saved
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_LIVE_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
    )
    surface = _CallRecordingSurface(_fixture_surface(real_schemas, analysis_tool="rank"))

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == retrieval["id"]
    assert outcome.detail == (
        "'get_data' takes different arguments than the plan was compiled against; "
        "the plan needs compiling and approving again"
    )
    assert outcome.revision == _BASE_FORMAT_LIVE_PLAN["revision"]
    assert surface.calls == []


async def test_a_refused_amendment_still_offers_the_pending_version_for_approval(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A `PlannerModel` can refuse to draft at all
    (`harness.planner_adapter.PlannerAdapter`'s own `MAX_PLAN_ROUNDS` cap,
    surfaced as `{"refused": {"text": ...}}` rather than a raised
    `PlannerAdapterError`). The planner node never writes `ctx.state["plan"]`
    on that path, so the plan the user already had -- still
    `PENDING_APPROVAL` -- stays exactly as it was. Since that plan is
    still real and still pending, the refusal is not a dead end: the node
    routes back to `approve`, which raises a fresh interrupt for the same
    plan, and answering it with the plan's own original revision approves
    it -- proving the refusal's claim that the version is "still there to
    approve" by actually approving it, not only by reading the sentence
    that says so.
    """
    plan_v1 = _planner_plan(question="kenya debt", schemas=real_schemas)
    planner = _Planner(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {
                "refused": {
                    "text": "This request has used its budget of 4 plan drafts, so nothing "
                    "was recorded. Your next message starts with a fresh budget."
                }
            },
        ]
    )
    executor = ScriptedExecutor()
    runner, _ = _plan_runner(
        pg_engine, planner, tool_surface=_fixture_surface(real_schemas), executor=executor
    )
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_PAUSE
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    events = await _resume(runner, APPROVE_PAUSE, approve_answer("not-a-real-revision"))

    # Not a dead end: a fresh approval interrupt for the same, unmodified
    # plan comes back instead, carrying why the amendment did not land.
    assert "plan_done" not in visited(events)
    assert _pending_interrupt(events) == APPROVE_PAUSE
    refused_output = _output_of(events, "planner")
    assert refused_output["revision"] == revision_1
    assert "budget of 4 plan drafts" in refused_output["refusal_note"]
    assert "still there to approve" in refused_output["refusal_note"]

    # `ctx.state["plan"]` was only ever written by the first, successful
    # planner call -- the refused second call left it untouched, so the
    # plan the user already saw is still exactly what's on record.
    assert PlannerPlan.model_validate_json(_session_state(runner)["plan"]) == plan_v1

    # Answering the fresh interrupt with the version's own original
    # revision approves it -- the round-cap refusal cost the user their
    # amendment, not their already-drafted plan. The plan's own compiled
    # workflow runs it through to completion against the tool surface.
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision_1))
    assert _pending_interrupt(events) is None
    approved = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    assert approved.status is PlanStatus.COMPLETED


# --- the execute node: the compiled workflow, the tool surface, the outcome map --


async def test_the_execute_node_runs_the_approved_workflow_through_the_surface(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """`execute` calls `execute_approved_plan` against the injected
    `ToolSurface`; the executor sees no plan step, only `plan_done`'s
    description of the final frame."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(_fixture_surface(real_schemas))
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=surface, executor=executor)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))

    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert [call["tool"] for call in executor.calls] == ["describe_frame"]
    executed = _output_of(events, "execute")
    # `Accepted.results` is terminal-steps-only: one frame ref
    # for "totals", not one per step.
    assert executed["frame_refs"] == [
        {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    ]
    assert [(step["tool"], step["digest"]) for step in executed["step_summaries"]] == [
        ("get_data", "fetched"),
        ("aggregate", "computed"),
    ]
    assert executed["plan_version"] == 1
    assert _output_of(events, "execute")["plan_version"] == 1


class _ServedPairsPlatform:
    """presentation-mcp over apis, which serves a frame under the pair it
    minted: `describe_frame` refuses `frame_unavailable` on any other pair and
    answers ok on one it serves."""

    def __init__(self, served: list[dict[str, str]]) -> None:
        self.served = served
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress
        self.calls.append({"tool": tool, "args": args})
        if args["frame_ref"] not in self.served:
            return {
                "outcome": "refusal",
                "refusal": {
                    "code": "frame_unavailable",
                    "retry_mode": "do_not_retry",
                    "origin": "service",
                    "fix": None,
                    "message": "no stored version",
                },
            }
        return {"outcome": "ok", "view": {"dimensions": ["stat_var"]}}


async def test_a_final_retrieval_read_with_extra_columns_is_shown_by_its_manifests_pair(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A retrieval the plan answers with is held by data-mcp under a pair of its
    own once read with extra columns, and apis serves only its manifest's pair.
    `execute` hands on the manifest's pair, so `plan_done` describes the series
    and registers it for charting, while the total reading it was given the
    frame's own pair. Each frame is registered with the pair the store holds
    it under, which is the typed pair for the retrieval and its own pair for
    the total."""
    plan = _planner_plan(schemas=real_schemas, terminal=("get_data", "totals"))
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    served = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
    typed = {"spec_id": SPEC_A, "content_version": "cv_" + "9" * 32}
    total = {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    surface = _CallRecordingSurface(
        _fixture_surface(
            real_schemas,
            get_data_result=ToolCallResult(
                is_error=False,
                structured_content=_get_data_manifest() | {"frame_ref": typed},
                text="fetched",
            ),
        )
    )
    platform = _ServedPairsPlatform([served, total])
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface, executor=platform)
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    (aggregate,) = [args for name, args in surface.calls if name == "aggregate"]
    assert aggregate["frame_ref"] == typed
    assert _output_of(events, "execute")["frame_refs"] == [served, total]
    assert [(call["tool"], call["args"]["frame_ref"]) for call in platform.calls] == [
        ("describe_frame", served),
        ("describe_frame", total),
    ]
    _, body = handed_back(model)
    assert [entry["frame"] for entry in body["frames"]] == [
        frame_id_for(served),
        frame_id_for(total),
    ]
    assert "frames_not_chartable" not in body
    frames = _session_state(runner)[ORCHESTRATOR_FRAMES_KEY]
    assert [(frame["ref"], frame["store_ref"]) for frame in frames.values()] == [
        (served, typed),
        (total, total),
    ]


_FINAL_RETRIEVAL = {"step_id": "get_data", "tool": "get_data"}


@pytest.mark.parametrize(
    ("state", "node_input"),
    [
        (
            {},
            {
                "frame_refs": [REF_A],
                "frame_ref_steps": [_FINAL_RETRIEVAL | {"is_final_step": True}],
            },
        ),
        (
            {
                ASKS_STATE_KEY: ["Kenya's debt service"],
                ORCHESTRATOR_KEPT_ASKS_KEY: {
                    "0": {
                        "steps": [_FINAL_RETRIEVAL | {"frame_ref": REF_A, "digest": "fetched"}],
                        "totals": [],
                    }
                },
            },
            {"text": "no plan avoids the failure", "final": True},
        ),
    ],
    ids=["execute-event", "kept-ask"],
)
async def test_a_hand_back_recorded_without_store_pairs_registers_its_frames(
    state: dict[str, Any], node_input: dict[str, Any]
) -> None:
    """A hand-back event or kept ask recorded without store pairs names each
    frame by the pair apis serves alone. Its frame is registered without
    a store pair, so the store is opened by the frame's own pair."""
    ctx = node_ctx(state=dict(state))
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())

    await plan_done.run(ctx, node_input)

    assert ctx.state[ORCHESTRATOR_FRAMES_KEY] == {frame_id_for(REF_A): {"ref": REF_A, "view": {}}}
    entry = ctx.state[ORCHESTRATOR_FRAMES_KEY][frame_id_for(REF_A)]
    assert frame_store_ref(entry) == REF_A


@pytest.mark.parametrize(
    ("whole", "named"),
    [(DAC_DONORS, "from official donors (DAC)"), (ALL_DONORS, "from every donor")],
    ids=["dac_donors", "all_donors"],
)
async def test_a_share_frame_is_registered_with_the_whole_its_plan_divides_by(
    decision: Callable[..., Decision], whole: dict[str, Any], named: str
) -> None:
    """The final frame of a share step carries the whole the approved plan's
    denominator source holds, which `make_charts` writes into the card's title."""
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=US_DONOR)
    ctx = node_ctx(state={"plan": plan.model_dump_json()})
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())
    node_input = {
        "frame_refs": [REF_A],
        "frame_ref_steps": [{"step_id": "share", "tool": "share_of", "is_final_step": True}],
    }

    await plan_done.run(ctx, node_input)

    assert ctx.state[ORCHESTRATOR_FRAMES_KEY][frame_id_for(REF_A)]["whole"] == named


def test_a_report_names_an_added_sum_by_the_whole_it_sums(
    decision: Callable[..., Decision],
) -> None:
    """A stop on the sum `propose_plan` adds over a whole's open side names the
    whole the model wrote, in the report's ids and its prose; a spec id is kept."""
    plan = us_share_of_kenya_oda(decision, whole=ALL_DONORS, numerator=US_DONOR, whole_of=True)
    assert [entry.sum_id for entry in plan.whole_sums] == ["all_oda_sum"]
    assert not plan.rolling_copies

    report = authored_report(
        {
            "step_id": "all_oda_sum",
            "requirement_id": "all_oda_sum",
            "message": "step all_oda_sum read no rows",
            "spec_id": "all_oda_sum",
        },
        plan,
    )

    assert report == {
        "step_id": "all_oda",
        "requirement_id": "all_oda",
        "message": "step all_oda read no rows",
        "spec_id": "all_oda_sum",
    }


async def test_a_summed_familys_total_reaches_the_agent_in_the_hand_back(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execute` hands `plan_done` the sentences `_stat_var_totals` writes,
    and `plan_done` gives each to the agent under `totals`."""
    sentence = (
        'By default, the figures for "OECD Creditor Reporting System (CRS)" are the '
        "sum of 74 indicators."
    )
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [sentence])
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, model = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert _output_of(events, "execute")["stat_var_totals"] == [sentence]
    _, body = handed_back(model)
    assert body["totals"] == [sentence]


async def test_a_none_tool_surface_refuses_loudly_rather_than_executing_nothing(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """`create_root_workflow(tool_surface=None)`, stated explicitly, must not
    let an approved plan reach execute silently: reading a plan the node
    cannot run has to fail loudly."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, _ = _plan_runner(pg_engine, planner)
    await _start(runner)
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="no tool surface"):
        await _resume(runner, APPROVE_PAUSE, approve_answer(revision))


async def test_a_rediscovery_outcome_replans_naming_the_requirement(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A retrieval step's own evidence binding names the requirement and the
    source a `RediscoveryNeeded` refusal reports -- a bare retry would only
    reproduce it, so `execute` routes back to the planner naming what to
    rediscover, never retrying the call itself."""
    plan = _planner_plan(schemas=real_schemas, bind_evidence=True)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))

    # Replanned (MAX_REPLANS == 1 covers one round): a fresh approval is
    # pending rather than the turn closing on the first refusal.
    assert _pending_interrupt(events) == APPROVE_PAUSE
    assert planner.calls[-1]["refusal"] == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
            "fix": {},
        },
        "also_stopped": [],
    }


async def test_a_method_choice_refusal_replans_once_then_closes_with_failed_step(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """`MAX_REPLANS == 1`: a second identical refusal closes the turn with
    what ran and the failed step named, rather than a third plan."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "entity_ambiguous",
            "retry_mode": "correct_and_retry",
            "message": "more than one entity matches 'Kenya'",
            "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    executor = ScriptedExecutor()
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface, executor=executor)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))
    assert _pending_interrupt(events) == APPROVE_PAUSE
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_PAUSE, approve_answer(replanned["revision"]))

    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "more than one entity matches 'Kenya'",
        "retry_mode": "correct_and_retry",
        # The `fix` keeps only what the user's sentence reads.
        "fix": {},
        "disposition": "method_choice",
        "cause": "data",
        "wave_remedy": "replan",
    }
    # get_data ran and completed before the refusal; its frame is kept.
    assert executed["frame_refs"] == [REF_A]
    assert [step["digest"] for step in executed["step_summaries"]] == ["fetched"]
    _, body = handed_back(model)
    # The refusal's retry mode and `fix` word the user's sentence, and its
    # cause, disposition and wave's remedy decide what follows. The agent
    # reads the rest, with what happened in the user's words in place of the
    # tool's message.
    hidden = ("retry_mode", "fix", "disposition", "cause", "wave_remedy")
    shown = {k: v for k, v in executed["failed_step"].items() if k not in hidden}
    assert body["failed_step"] == {**shown, "message": failure_cause("entity_ambiguous", {})}


async def test_a_stop_classified_refusal_closes_the_turn_without_spending_a_replan(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """An `over_ceiling` refusal naming no `ceiling_kind` at all -- an older
    upstream, or a raw payload assembled by hand -- is one
    `executor.classify.classify_refusal` cannot place as narrowing with scope
    or as naming a different spec, so it stays a `STOP`: `execute` closes the
    turn on this first refusal instead of spending the turn's one replan on a
    guess. Unlike `test_a_method_choice_refusal_
    replans_once_then_closes_with_failed_step`'s `entity_ambiguous`, no
    second approval is ever pending."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many series.",
            "fix": {"detail": "too many series"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    # A plain `Refused` (the `_stopped_by_refusal` catch-all `STEP_REFUSED`,
    # not a `MethodChoiceNeeded`/`RediscoveryNeeded`) carries `tool=None`,
    # but `_failed_step` reports the tool's own `code`/`message` off
    # `outcome.payload` -- the `ToolFailure` the raw call actually returned
    # -- rather than the internal `RefusalKind` that closed the turn.
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series.",
        "retry_mode": "correct_and_retry",
        "fix": {},
        "disposition": "stop",
        "cause": "data",
        "wave_remedy": "user",
    }
    assert executed["frame_refs"] == [REF_A]
    _, body = handed_back(model)
    assert body["failed_step"] == {
        "step_id": "totals",
        "code": "over_ceiling",
        "message": failure_cause("over_ceiling", {}),
    }


async def test_a_member_cardinality_ceiling_closes_the_turn_without_replanning(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A `member_cardinality` `over_ceiling` refusal is fixed by the resolved
    spec_id's own five-tuple and the index version behind it: no scope edit
    narrows it, and rediscovering the same spec by paraphrase search
    reproduces the identical refusal. `classify_refusal` classifies it
    `STOP`, so `execute` closes the turn on this first refusal instead of
    spending the turn's one replan on a rediscovery that cannot help."""
    plan = _planner_plan(schemas=real_schemas, bind_evidence=True)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: 2933 members. A narrower question "
            "resolves to a smaller definition.",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: 2933 members. A narrower question "
        "resolves to a smaller definition.",
        "retry_mode": "correct_and_retry",
        "fix": {"ceiling_kind": "member_cardinality"},
        "disposition": "stop",
        "cause": "data",
        "wave_remedy": "user",
    }
    _, body = handed_back(model)
    assert (body["failed_step"]["step_id"], body["failed_step"]["code"]) == (
        "get_data",
        "over_ceiling",
    )


async def test_a_member_cardinality_ceilings_remedy_reaches_the_agent_in_the_hand_back(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """`fix.remedy`, the one user-facing sentence naming what the user can do
    after a `member_cardinality` ceiling's dead end, reaches
    the agent verbatim in the hand-back's `failed_step`, beside the code and
    message."""
    plan = _planner_plan(schemas=real_schemas, bind_evidence=True)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    remedy = (
        "This spec_id's definition has more member series than one request can "
        "serve. scope.entities and scope.window do not change that. A narrower "
        "question, such as one naming a sector for aid flows, resolves to a "
        "smaller definition."
    )
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": f"over the coverage ceiling: 2933 members. {remedy}",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
                "remedy": remedy,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface)
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert _pending_interrupt(events) is None
    assert _output_of(events, "execute")["failed_step"]["remedy"] == remedy
    _, body = handed_back(model)
    assert body["failed_step"]["remedy"] == remedy


async def test_a_family_returned_unsummed_closes_the_turn_on_a_message_the_user_can_act_on(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A `get_data` asked to sum its family declares three StatVars, so the
    sum did not happen. The same plan draws the same frame, so `execute`
    closes the turn on this first refusal: no replan, no second approval of
    an identical plan. The agent reads what happened and the executor's
    remedy, both written for the user, in place of the count of `stat_vars`
    the refusal's `detail` states."""
    plan = _planner_plan(schemas=real_schemas, stat_vars="sum")
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    unsummed = ToolCallResult(
        is_error=False,
        structured_content=_get_data_manifest() | {"stat_var_count": 3},
        text="fetched",
    )
    surface = _fixture_surface(real_schemas, get_data_result=unsummed)
    runner, model = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert _pending_interrupt(events) is None
    assert len(planner.calls) == 1
    message = (
        "The data service returned this measure's component series side by side and did "
        "not add them up, so there is no total to report."
    )
    remedy = "Ask for the component series separately, or ask for the total again later."
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "family_not_summed",
        "message": message,
        "retry_mode": "do_not_retry",
        "fix": {},
        "disposition": "stop",
        "cause": "data",
        "wave_remedy": "user",
        "remedy": remedy,
    }
    assert executed["frame_refs"] == []
    text, body = handed_back(model)
    assert body["failed_step"] == {
        "step_id": "get_data",
        "code": "family_not_summed",
        "message": failure_cause("family_not_summed", {}),
        "remedy": remedy,
    }
    assert "distinct stat_vars" not in text


async def test_an_over_ceiling_rows_refusal_still_replans(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """A `rows`/`bytes` ceiling narrows with `scope.entities`/`scope.window`,
    so -- unlike `member_cardinality` above -- it is a method choice, not a
    dead end, and replans once like any other correctable refusal."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many rows. Narrow the scope and retry.",
            "fix": {"detail": "too many rows", "ceiling_kind": "rows"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(revision))

    # A method choice, not a dead end: the planner is asked again (the
    # scripted planner redrafts the same plan, so the version does not bump),
    # and a fresh approval is pending rather than the turn closing.
    assert _pending_interrupt(events) == APPROVE_PAUSE
    assert len(planner.calls) == 2
    assert planner.calls[-1]["refusal"]["code"] == "over_ceiling"


_MOVED = (EvidenceIssueCode.STAT_VAR_FAMILY_MOVED, Severity.ERROR)
_REBOUND = (EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT, Severity.ERROR)
_UNCONFIRMED = (EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT, Severity.WARNING)
_SHORT = (EvidenceIssueCode.COVERAGE_GAPS, Severity.ERROR)


# What one stop of a stubbed wave failed on: its evidence issues, each a code
# and severity, or `None` for a tool refusal `classify_refusal` calls `STOP`.
type _StopSpec = Sequence[tuple[EvidenceIssueCode, Severity]] | None


def _stop(revision: str, step_id: str, spec: _StopSpec) -> Refused:
    """A stop at `step_id`: a `requirement_unmet` one on `spec`'s issues, or a
    tool's `spec_not_exportable` refusal when `spec` is `None`."""
    if spec is None:
        return Refused(
            revision=revision,
            refusal=RefusalKind.STEP_REFUSED,
            detail="'get_data' refused: spec_not_exportable",
            step_id=step_id,
            payload=ToolFailure(
                code="spec_not_exportable",
                message="the definition mints no dataset",
                retry_mode="do_not_retry",
            ),
        )
    return Refused(
        revision=revision,
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail=f"requirement {step_id!r} is not met by what 'get_data' returned",
        step_id=step_id,
        requirement_id=step_id,
        issues=tuple(
            EvidenceIssue(
                code=code,
                severity=severity,
                requirement_id=step_id,
                detail=f"{code.value} on {step_id}",
            )
            for code, severity in spec
        ),
    )


@pytest.mark.parametrize(
    ("primary", "also", "closed_on"),
    [
        ((_MOVED,), (), "stat_var_family_moved"),
        ((_REBOUND,), (), "stat_var_membership_drift"),
        ((_MOVED, _UNCONFIRMED), (), "stat_var_family_moved"),
        ((_MOVED, _REBOUND), (), "stat_var_family_moved"),
        ((_MOVED, _SHORT), (), "stat_var_family_moved"),
        ((_MOVED,), ((_SHORT,),), "stat_var_family_moved"),
        (None, ((_SHORT,),), None),
        (None, ((_MOVED,), None), "stat_var_family_moved"),
    ],
    ids=[
        "family-moved",
        "binding-rebound",
        "family-moved-beside-a-caveat",
        "family-grown-past-one-series",
        "family-moved-and-coverage-short",
        "family-moved-beside-a-replannable-stop",
        "tool-stop-beside-a-replannable-stop",
        "tool-stop-beside-dead-ends",
    ],
)
async def test_a_wave_of_dead_ends_closes_the_turn(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
    primary: _StopSpec,
    also: tuple[_StopSpec, ...],
    closed_on: str | None,
) -> None:
    """A wave whose every stop is a dead end, or whose remedy is the user's,
    closes the turn on the stop that decides its remedy, the first of the
    most severe, with no replan. An evidence stop is a dead end when the
    StatVar family's digest moved since Search approved it and that move
    explains every error, a family grown past the one series approved
    included: a replan in the same turn approves the digest the turn already
    holds, so the turn closes on the source-gone sentence, which asks the user
    to ask again. A tool's `STOP` refusal is another. A moved family is the
    user's however the rest of its wave fails, and so is a binding that fails
    the same way for every search. A wave of stops a new plan gets past
    (`closed_on` None) replans."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])

    async def _stub_wave(drafted_plan: Any, **_kwargs: Any) -> Any:
        revision = plan_revision(drafted_plan)
        siblings = tuple(_stop(revision, f"sibling_{n}", spec) for n, spec in enumerate(also))
        stop = _stop(revision, "get_data", primary)
        return drafted_plan, stop.model_copy(update={"also_stopped": siblings})

    monkeypatch.setattr("harness.graph.execute.execute_approved_plan", _stub_wave)
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert len(planner.calls) == (2 if closed_on is None else 1)
    if closed_on is None:
        assert _pending_interrupt(events) == APPROVE_PAUSE
        return
    assert _pending_interrupt(events) is None
    failed = _output_of(events, "execute")["failed_step"]
    # Another spec avoids `spec_not_exportable`, and the user has to accept a
    # moved family, so the family decides the wave wherever it stands.
    assert failed["code"] == closed_on
    assert failed["wave_remedy"] == "user"
    if closed_on == "stat_var_family_moved":
        gone = FAILURE_SENTENCES[FailureClass.SOURCE_GONE]
        assert failure_detail(failed) == f"{gone.cause}, {gone.advice}."


@pytest.mark.parametrize(
    ("retryable", "beside", "wave_remedy"),
    [
        (True, False, "wait"),
        (False, False, "system"),
        (True, True, "wait"),
    ],
    ids=["retryable", "permanent", "retryable-beside-a-replannable-stop"],
)
async def test_a_transport_stop_closes_the_turn_without_a_replan(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
    retryable: bool,
    beside: bool,
    wave_remedy: str,
) -> None:
    """A step the tool surface failed closes the turn without a replan, which
    would ask the user to approve again a plan no new draft gets past: the
    executor spent its retries on a retryable one, so the user tries again
    later, and no later attempt gets past a permanent one. A replannable stop
    in the same wave does not route it to the planner, since the transport
    stop decides the wave."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])

    async def _stub_wave(drafted_plan: Any, **_kwargs: Any) -> Any:
        revision = plan_revision(drafted_plan)
        stop = Refused(
            revision=revision,
            refusal=RefusalKind.TRANSPORT_FAILED,
            detail="could not call 'get_data'",
            step_id="get_data",
            retryable=retryable,
        )
        if beside:
            stop = stop.model_copy(
                update={"also_stopped": (_stop(revision, "sibling", (_SHORT,)),)}
            )
        return drafted_plan, stop

    monkeypatch.setattr("harness.graph.execute.execute_approved_plan", _stub_wave)
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert len(planner.calls) == 1
    assert _pending_interrupt(events) is None
    failed = _output_of(events, "execute")["failed_step"]
    assert failed["step_id"] == "get_data"
    assert failed["wave_remedy"] == wave_remedy


async def test_an_approval_stored_over_accept_provider_difference_offers_the_card_again(
    pg_engine: AsyncEngine, decision: Callable[..., Decision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user approves the US share of ODA to Kenya, and the approved plan is stored
    with `accept_provider_difference` on its share, as plans approved before that field
    left the plan are. `execute` reads it awaiting approval, with no approval recorded:
    nothing runs, the turn does not raise, and the card is offered again under the
    revision of the plan as it reads."""
    plan = us_share_of_kenya_oda(decision, whole=DAC_DONORS, numerator=US_DONOR)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=surface)
    await _start(runner)
    decide = approve_module.decide

    def stored_with_the_field(ctx: Any, answer: dict[str, Any], *, early_mints: Any) -> Any:
        routed = decide(ctx, answer, early_mints=early_mints)
        stored = json.loads(ctx.state["plan"])
        (share,) = [r for r in stored["evidence_requirements"] if r["kind"] == "share_of"]
        share["accept_provider_difference"] = True
        ctx.state["plan"] = json.dumps(stored)
        return routed

    monkeypatch.setattr(approve_module, "decide", stored_with_the_field)

    events = await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))

    assert _pending_interrupt(events) == APPROVE_PAUSE
    offered = _output_of(events, "execute")
    assert offered["revision"] == plan_revision(plan)
    assert surface.calls == []
    held = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    assert (held.status, held.approved_revision, held.approved_at) == (
        PlanStatus.PENDING_APPROVAL,
        None,
        None,
    )


async def test_a_not_approved_outcome_raises_instead_of_replanning(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Refused(NOT_APPROVED)` means the workflow's own `approve` node let an
    unapproved plan through -- replanning would loop on the same defect, so
    `execute` raises instead. The `approve` node cannot itself
    produce this in normal operation (it calls `plan.approve()` immediately
    before routing here), so the outcome is stubbed to prove this node's own
    branch rather than trying to coax `approve` into a defect it structurally
    prevents."""
    plan = _planner_plan(schemas=real_schemas)
    planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])

    async def _stub_not_approved(
        drafted_plan: Any, *, surface: Any, max_concurrent_steps: int = 4, **_kwargs: Any
    ) -> Any:
        del surface, max_concurrent_steps
        return drafted_plan, Refused(
            revision=plan_revision(drafted_plan),
            refusal=RefusalKind.NOT_APPROVED,
            detail="this plan is approved, not approved; nothing runs",
        )

    monkeypatch.setattr("harness.graph.execute.execute_approved_plan", _stub_not_approved)
    runner, _ = _plan_runner(pg_engine, planner, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="not_approved"):
        await _resume(runner, APPROVE_PAUSE, approve_answer(revision))


def test_a_refused_stop_that_read_a_source_names_it_in_the_replan_payload() -> None:
    refused = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement not met",
        step_id="totals",
        spec_id=SPEC_A,
    )

    assert _replan_refusal(refused, origins={})["spec_id"] == SPEC_A
    # A stop that read a windowed copy names the origin the planner was shown,
    # in the payload and in each sibling stop.
    payload = _replan_refusal(
        refused.model_copy(update={"also_stopped": (refused,)}), origins={SPEC_A: SPEC_B}
    )
    assert payload["spec_id"] == SPEC_B
    assert [stop["spec_id"] for stop in payload["also_stopped"]] == [SPEC_B]


def test_a_rediscovery_stop_names_the_spec_it_read_whatever_the_origins() -> None:
    """Only a `Refused` stop maps a windowed copy to its origin. A rediscovery
    holds the copy's own spec_id."""
    rediscovery = RediscoveryNeeded(
        revision="rev1",
        step_id="totals",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="totals",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="definition_not_held", message="gone", retry_mode="correct_and_retry"
        ),
        completed=(),
    )

    payload = _replan_refusal(rediscovery, origins={SPEC_A: SPEC_B})

    assert payload["spec_id"] == SPEC_A


def test_the_replan_refusal_and_failed_step_maps_cover_every_outcome_kind() -> None:
    """`_replan_refusal`'s payload shapes, pinned directly against each outcome
    kind -- not merely reachable through one graph run apiece."""
    method_choice = MethodChoiceNeeded(
        revision="rev1",
        step_id="totals",
        tool=ToolId(server="analysis-mcp", name="aggregate"),
        code="entity_ambiguous",
        candidates=["country/KEN", "country/KEN-old"],
        refusal=ToolFailure(
            code="entity_ambiguous", message="ambiguous", retry_mode="correct_and_retry"
        ),
        completed=(),
    )
    assert _replan_refusal(method_choice, origins={}) == {
        "kind": "method_choice_needed",
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "candidates": ["country/KEN", "country/KEN-old"],
        "refusal": {
            "code": "entity_ambiguous",
            "message": "ambiguous",
            "retry_mode": "correct_and_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(method_choice) == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "ambiguous",
        "retry_mode": "correct_and_retry",
        "fix": {},
        "disposition": "method_choice",
        "cause": "data",
        "wave_remedy": "replan",
    }

    rediscovery = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    assert _replan_refusal(rediscovery, origins={}) == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        # An empty `fix` carries nothing to choose from -- `candidates_of`
        # reads `None`, not an empty dict standing in for "no candidates".
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "message": "the source moved",
            "retry_mode": "refresh_then_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(rediscovery) == {
        "step_id": "get_data",
        "tool": "get_data",
        "code": "rediscovery_needed",
        "message": "the source moved",
        "retry_mode": "refresh_then_retry",
        "fix": {},
        "disposition": "rediscover",
        "cause": "data",
        "wave_remedy": "replan",
    }

    refused = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="totals",
    )
    assert _replan_refusal(refused, origins={}) == {
        "kind": "refused",
        "refusal": "empty_answer",
        "detail": "no rows matched",
        "step_id": "totals",
        "requirement_id": None,
        "issues": [],
        "also_stopped": [],
    }
    failed = _failed_step(refused)
    assert (failed["step_id"], failed["tool"], failed["code"]) == ("totals", None, "empty_answer")
    assert "no rows matched" not in failed["message"]

    # A `STEP_REFUSED` stop carries the tool's own `ToolFailure` as
    # `payload` -- `_failed_step` reports its `code`/`message`, not
    # `refusal.value`/`detail`, so the agent is told what the
    # tool actually said ("over_ceiling") rather than the internal kind
    # that closed the turn ("step_refused").
    refused_with_payload = Refused(
        revision="rev1",
        refusal=RefusalKind.STEP_REFUSED,
        detail="'aggregate' refused: refused",
        step_id="totals",
        payload=ToolFailure(
            code="over_ceiling",
            message="over the coverage ceiling: too many series",
            retry_mode="correct_and_retry",
        ),
    )
    assert _failed_step(refused_with_payload) == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series",
        "retry_mode": "correct_and_retry",
        "fix": {},
        "disposition": "stop",
        "cause": "data",
        "wave_remedy": "user",
    }

    # A `REQUIREMENT_UNMET` stop carries `check_evidence`'s own findings as
    # `issues` -- `_failed_step` reports the specific `EvidenceIssueCode`
    # and detail of whichever ERROR(s) fired, not `refusal.value`/`detail`'s
    # generic "requirement X is not met by what Y returned", which names
    # neither which of the ten sub-checks failed nor what it found. A
    # `WARNING`-severity issue on the same outcome (a real shape:
    # `check_evidence` can return caveats alongside the errors that stopped
    # the run) is left out of the reported code/message; only errors decide
    # what the run reports as its reason.
    refused_with_issues = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
    )
    assert _failed_step(refused_with_issues) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_pin_mismatch",
        "message": "the requirement pins 'spec_a', the source found is 'spec_b'",
        "cause": "data",
        "wave_remedy": "user",
    }

    # No ERROR-severity issue at all (every finding a caveat) falls back to
    # the first issue given, not nothing. `check_evidence` does not produce
    # this case (it only raises when `errors(findings)` is non-empty), but
    # `_failed_step` owes a code/message pair for whatever `issues` it is
    # handed.
    refused_with_only_warnings = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
        ),
    )
    assert _failed_step(refused_with_only_warnings) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_unverified",
        "message": "the source states no provenance",
        "cause": "data",
        "wave_remedy": "replan",
    }

    # A stop with every donor named missing. The failed step carries what the
    # user can do about it, since the code alone says only that the names could
    # not be resolved.
    refused_names_missing = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'donor_oda' is not met by what 'get_data' returned",
        step_id="donor_oda",
        requirement_id="donor_oda",
        issues=tuple(
            EvidenceIssue(
                code=EvidenceIssueCode.ENTITY_MISSING,
                severity=Severity.ERROR,
                requirement_id="donor_oda",
                detail=f"the source resolves no entity named {name!r}",
                required=name,
            )
            for name in ("United States", "Germany")
        ),
    )
    names_missing = _failed_step(refused_names_missing)
    assert names_missing.pop("remedy") == ISSUE_REMEDIES[EvidenceIssueCode.ENTITY_MISSING]
    assert names_missing == {
        "step_id": "donor_oda",
        "tool": None,
        "code": "entity_missing",
        "message": "the source resolves no entity named 'United States'; the source "
        "resolves no entity named 'Germany'",
        "cause": "data",
        "wave_remedy": "user",
    }


_GET_DATA = ToolId(server="data-mcp", name="get_data")


def _with_a_second_rolling_reader(decision: Callable[..., Decision]) -> PlannerPlan:
    """`stored_split_plan` with `last_three`, a growth over the newest three
    years, reading `debt` too, so the split has it read `debt_history`."""
    plan = stored_split_plan(decision)
    window = RollingWindow(
        window_start=RelativePeriod(offset=-3), window_end=RelativePeriod(offset=0)
    )
    last_three = GrowthRequirement(
        id="last_three", input_id="debt", window_start=2021, window_end=2024, rolling=window
    )
    draft = split_draft(decision)
    split = with_authored_requirements(draft, [*authored(draft).evidence_requirements, last_three])
    assert split.requirement("last_three").input_id == "debt_history"  # type: ignore[union-attr]
    assert plan.translation is not None and plan.workflow is not None
    (growth,) = [step for step in plan.workflow.steps if step.id == "last_five"]
    return split.with_translation(
        translation=plan.translation.model_copy(
            update={
                "operations": (
                    *plan.translation.operations,
                    GrowthOperation(
                        id="last_three",
                        requirement_id="last_three",
                        input="debt_history",
                        window_start=2021,
                        window_end=2024,
                        rolling=window,
                    ),
                ),
                "terminal": (*plan.translation.terminal, "last_three"),
            }
        ),
        workflow=plan.workflow.model_copy(
            update={
                "steps": (
                    *plan.workflow.steps,
                    growth.model_copy(update={"id": "last_three", "operation_id": "last_three"}),
                ),
                "terminal_steps": (*plan.workflow.terminal_steps, "last_three"),
            }
        ),
    )


_ROLLED_START = Refused(
    revision="rev1",
    refusal=RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET,
    detail=(
        "step 'last_five' would start in 1995, counted from 2023, before 2000, "
        f"the first year {SPEC_DEBT_HISTORY} holds"
    ),
    step_id="last_five",
    requirement_id="debt_history",
    payload=ToolFailure(
        code="rolling_window_outside_dataset",
        message="The data for external debt service paid starts in 2000.",
        retry_mode="do_not_retry",
    ),
)
_REDISCOVERY = RediscoveryNeeded(
    revision="rev1",
    step_id="debt_history",
    tool=_GET_DATA,
    requirement_id="debt_history",
    spec_id=SPEC_DEBT_HISTORY,
    refusal=ToolFailure(
        code="upstream_version_changed", message="", retry_mode="refresh_then_retry"
    ),
    completed=(),
)
_CAPABILITY = Refused(
    revision="rev1",
    refusal=RefusalKind.STEP_REFUSED,
    detail="'get_data' refused: capability_unavailable",
    step_id="debt_history",
    requirement_id="debt_history",
    payload=ToolFailure(
        code="capability_unavailable",
        message="this deployment cannot read debt_history's window",
        retry_mode="do_not_retry",
    ),
)


@pytest.mark.parametrize(
    ("stop", "second_reader", "ids", "failed", "unavailable"),
    [
        pytest.param(
            _ROLLED_START,
            False,
            ("last_five", "debt"),
            {"last_five": "last_five"},
            {},
            id="rolled-start-before-the-copy",
        ),
        pytest.param(
            _ROLLED_START,
            True,
            ("last_five", "debt"),
            {"last_five": "last_five"},
            {},
            id="rolled-start-beside-a-second-reader",
        ),
        pytest.param(
            _REDISCOVERY,
            False,
            ("debt", "debt"),
            {"last_five": "last_five"},
            {},
            id="copy-retrieval-needs-rediscovery",
        ),
        pytest.param(
            _REDISCOVERY,
            True,
            ("debt", "debt"),
            {"last_five": "last_five", "last_three": "last_three"},
            {},
            id="copy-retrieval-read-by-two-rolling-steps",
        ),
        pytest.param(
            _CAPABILITY,
            False,
            ("debt", "debt"),
            {"last_five": "last_five"},
            {SPEC_DEBT_HISTORY: "this deployment cannot read debt's window"},
            id="copy-retrieval-capability-unavailable",
        ),
    ],
)
async def test_a_stop_on_a_copy_names_the_requirement_the_model_wrote(
    decision: Callable[..., Decision],
    stop: Refused | RediscoveryNeeded,
    second_reader: bool,
    ids: tuple[str, str],
    failed: dict[str, str],
    unavailable: dict[str, str],
) -> None:
    """A run stopped on a copy the split added (`debt_history`) names no copy.
    The planner reads the step and requirement the model wrote (`debt`). The
    agent and the plan record read the stop as landing on each step the model
    wrote that reads the copy, `last_five` and, where it is there, `last_three`:
    `debt` ran on its own pin and feeds `total`. A stop on `last_five` itself
    lands on it alone. A capability the copy's retrieval lacks marks the
    copy's own pin unavailable, and leaves `debt`'s."""
    plan = stored_split_plan(decision)
    readers = ["last_five"]
    if second_reader:
        plan = _with_a_second_rolling_reader(decision)
        readers.append("last_three")
    plan = PlannerPlan.model_validate(
        {
            **plan.model_dump(),
            "asks": ["debt service each year", *(f"its change, {step}" for step in readers)],
            "answered": [
                {"ask": 0, "outputs": ["total"]},
                *({"ask": n, "outputs": [step]} for n, step in enumerate(readers, start=1)),
            ],
        }
    )

    refusal = _planner_refusal(stop, plan=plan, origins={})
    ctx = node_ctx(state={"plan": plan.model_dump_json()})
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())
    node_input = {"frame_refs": [], "frame_ref_steps": [], "failed_step": _failed_step(stop)}
    handed = (await plan_done.run(ctx, node_input)).output

    assert (refusal["step_id"], refusal["requirement_id"]) == ids
    assert "debt_history" not in json.dumps(refusal)
    assert "debt_history" not in json.dumps(handed)
    (record,) = ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    assert [step["id"] for step in record["steps"] if step["failed"]] == list(
        dict.fromkeys(failed.values())
    )
    assert handed["answers"] == "plan"
    result = handed["body"]
    assert result["failed_step"]["step_id"] == "last_five"
    assert [
        (entry["step"], entry["status"], entry.get("failed_at"))
        for ask in result["asks"]
        for entry in ask["answered_by"]
    ] == [
        ("total", "not_run", None),
        *(
            (step, "failed", failed[step]) if step in failed else (step, "not_run", None)
            for step in readers
        ),
    ]
    assert ctx.state[ORCHESTRATOR_UNAVAILABLE_KEY] == unavailable


def test_a_stop_on_a_copy_renames_it_only_where_a_field_names_a_step(
    decision: Callable[..., Decision],
) -> None:
    """The planner reads `debt` for the copy in the step, the requirement and
    the tool's message. A ref the tool's `fix` holds keeps the text it has,
    though it ends in the copy's id."""
    plan = stored_split_plan(decision)
    stop = RediscoveryNeeded(
        revision="rev1",
        step_id="debt_history",
        tool=_GET_DATA,
        requirement_id="debt_history",
        spec_id=SPEC_DEBT_HISTORY,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="debt_history needs its source found again",
            retry_mode="refresh_then_retry",
            fix={"topic": "debt_history", "path": "dc/topic/debt_history"},
        ),
        completed=(),
    )

    refusal = _planner_refusal(stop, plan=plan, origins={})

    assert (refusal["step_id"], refusal["requirement_id"]) == ("debt", "debt")
    assert refusal["refusal"]["message"] == "debt needs its source found again"
    assert refusal["refusal"]["fix"] == {"topic": "debt_history", "path": "dc/topic/debt_history"}


def test_a_window_short_gap_on_a_copy_names_the_origin_and_the_years_its_retrieval_requested(
    decision: Callable[..., Decision], source: Callable[..., SelectedSource]
) -> None:
    """The copy keeps its origin's stated 2015-2024 and is retrieved from the
    dataset's first year. The gap the planner reads names the origin and both."""
    plan = stored_split_plan(decision)
    copy = plan.requirement("debt_history")
    assert isinstance(copy, DatasetRequirement)
    retrieved = source(
        spec_id=SPEC_DEBT_HISTORY,
        coverage_window=Period(start=FIRST_YEAR, end=2023),
        identity={"stat_var": "DebtServicePaid", "requested_window": open_window(FIRST_YEAR)},
    )
    findings = check_evidence(
        copy.model_copy(update={"coverage": Coverage(on_missing="fail")}),
        evidence_from_source(retrieved),
    )
    stop = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'debt_history' is not met by what 'get_data' returned",
        step_id="debt_history",
        requirement_id="debt_history",
        issues=tuple(findings),
    )

    refusal = _planner_refusal(stop, plan=plan, origins={})

    (gap,) = [issue for issue in refusal["issues"] if issue["code"] == "window_short"]
    # A finding naming no series reads as it did before findings could name one.
    assert set(gap) == {"code", "severity", "requirement_id", "detail", "found", "required"}
    assert gap["requirement_id"] == "debt"
    assert gap["detail"] == (
        "the source covers 2000-2023, the requirement asks for 2015-2024, "
        "and its retrieval requests every year from 2000"
    )
    assert "debt_history" not in json.dumps(refusal)


@pytest.mark.parametrize("kind", list(RefusalKind))
def test_a_stop_no_tool_refused_reaches_the_orchestrator_by_its_kind_alone(
    kind: RefusalKind,
) -> None:
    """A run's own stop carries the ledger's failure reason or an exception's
    text in `detail`, which can name literal spec ids: the orchestrator is told
    the stop's code and a sentence for its kind, never the detail."""
    detail = f"the run failed before it came to an outcome: store {'ops_' + 'd' * 32} refused"
    refused = Refused(revision="rev1", refusal=kind, detail=detail, run_id="run_1")

    failed = _failed_step(refused)

    assert failed["code"] == kind.value
    assert failed["message"]
    assert "ops_" not in json.dumps(failed)


def test_replan_refusal_names_every_sibling_a_wave_surfaced() -> None:
    """`also_stopped` carries the rest of a wave's stops in the primary's own
    per-kind shape, empty when the wave surfaced only the one outcome
    (`executor.run._Run.execute`'s siblings, folded into the outcome
    `_run_workflow` returns)."""
    sibling = RediscoveryNeeded(
        revision="rev1",
        step_id="fetch_population",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="population",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the population source moved too",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    primary = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
        also_stopped=(sibling,),
    )

    payload = _replan_refusal(primary, origins={})

    assert payload["also_stopped"] == [
        {
            "kind": "rediscovery_needed",
            "step_id": "fetch_population",
            "tool": "get_data",
            "requirement_id": "population",
            "spec_id": SPEC_A,
            "candidates": None,
            "refusal": {
                "code": "upstream_version_changed",
                "message": "the population source moved too",
                "retry_mode": "refresh_then_retry",
                "fix": {},
            },
        }
    ]
    assert _replan_refusal(sibling, origins={})["also_stopped"] == []


def test_mark_masked_step_failed_corrects_every_sibling_a_wave_surfaced() -> None:
    """A wave whose primary and one `also_stopped` sibling both passed their
    raw call but were refused by a post-call check (`_check_evidence`) each
    carry a distinct `step_id`. `_mark_masked_step_failed` corrects both
    Activity feed records, each with its own stop's code, in either order."""

    class _RecorderStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def mark_step_failed(self, *, step_id: str, tool: str, code: str, detail: str) -> None:
            self.calls.append({"step_id": step_id, "code": code, "detail": detail})

    sibling = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="fetch_population",
    )
    primary = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
        also_stopped=(sibling,),
    )

    pin_record = {
        "step_id": "fetch_gdp",
        "code": "source_pin_mismatch",
        "detail": "the requirement pins 'spec_a', the source found is 'spec_b'",
    }
    empty_record = {
        "step_id": "fetch_population",
        "code": "empty_answer",
        "detail": "no rows matched",
    }

    recorder = _RecorderStub()
    _mark_masked_step_failed(recorder, primary, tools={})
    assert recorder.calls == [pin_record, empty_record]

    # With the milder stop first, its record keeps its own code although the
    # sibling decides the wave.
    swapped = sibling.model_copy(
        update={"also_stopped": (primary.model_copy(update={"also_stopped": ()}),)}
    )
    recorder = _RecorderStub()
    _mark_masked_step_failed(recorder, swapped, tools={})
    assert recorder.calls == [empty_record, pin_record]


# --- the seed intake's wording ---------------------------------------------------


def test_a_stale_reason_for_another_dataset_omits_a_spec_id_search_did_not_name() -> None:
    part = {"status": "definite"}
    unnamed = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part)
    named = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part, answered="spec_x")
    assert stale_reason(unnamed) == "Search now resolves it to another dataset"
    assert stale_reason(named) == "Search now resolves it to another dataset (spec_x)"


# --- the switch node: a scope choice moved on the plan card, in code ----------


# --- a switch onto a family the mint adds by a declared composite ----------------


@pytest.mark.parametrize(
    ("exact_codes", "kept"), [(frozenset({"DT.ODA.ALLD.CD"}), True), (frozenset(), False)]
)
def test_a_repinned_requirement_keeps_its_series_code_only_where_the_new_dataset_is_it(
    exact_codes: frozenset[str], kept: bool
) -> None:
    stated = DatasetRequirement(
        id="aid",
        decision_ids=["window"],
        measure="aid",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
        source="World Bank",
        source_ref="provenance/WorldBank",
        source_exclude=True,
        source_code="DT.ODA.ALLD.CD",
        source_spec_id="spec_" + "a" * 32,
    )

    moved = repinned(stated, "spec_" + "b" * 32, exact_codes)

    assert (moved.source_spec_id, moved.source, moved.source_ref) == (
        "spec_" + "b" * 32,
        None,
        None,
    )
    assert moved.source_exclude is False
    assert moved.source_code == ("DT.ODA.ALLD.CD" if kept else None)


# --- early mints at the card offer ---------------------------------------------


class _EarlySends:
    """`EarlyMints`' send: records each call's arguments, then answers
    `outcome`, raises it, or waits on `hold` first when one is set."""

    def __init__(self, outcome: str | Exception = "ok", *, hold: asyncio.Event | None = None):
        self.outcome = outcome
        self.hold = hold
        self.sent: list[dict[str, Any]] = []
        self.cancelled = 0

    async def __call__(self, args: dict[str, Any]) -> str:
        self.sent.append(args)
        if self.hold is not None:
            try:
                await self.hold.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


async def test_each_card_offered_starts_its_retrievals_once_with_the_arguments_execution_sends(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
) -> None:
    """The first card starts its retrieval, an amendment's card starts its own,
    a re-offer of that card starts nothing, and the run the approval starts
    asks `get_data` with exactly the arguments the early call sent."""
    plan_v1 = _planner_plan(question="kenya debt", schemas=real_schemas, stat_vars="sum")
    plan_v2 = _planner_plan(question="kenya debt, revised", schemas=real_schemas, stat_vars="sum")
    planner = _Planner(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    sends = _EarlySends()
    surface = _CallRecordingSurface(_fixture_surface(real_schemas))
    runner, _ = _plan_runner(
        pg_engine,
        planner,
        tool_surface=surface,
        early_mints=EarlyMints(send=sends, max_concurrent=4, settle_s=3.0),
    )

    await _start(runner)
    await asyncio.sleep(0)
    assert sends.sent == [{"spec_id": SPEC_A, "stat_vars": "sum"}]

    events = await _resume(runner, APPROVE_PAUSE, approve_answer("not-a-real-revision"))
    await asyncio.sleep(0)
    revision_2 = _output_of(events, "planner")["revision"]
    assert len(sends.sent) == 2

    # An answer naming the superseded card offers the same card again.
    await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan_v1)))
    await asyncio.sleep(0)
    assert len(sends.sent) == 2

    await _resume(runner, APPROVE_PAUSE, approve_answer(revision_2))
    assert [args for tool, args in surface.calls if tool == "get_data"] == sends.sent[1:]


@pytest.mark.parametrize(
    ("outcome", "logged"),
    [
        ("declined", "outcome=declined"),
        (RuntimeError("apis unreachable"), "outcome=error:RuntimeError"),
    ],
    ids=["declined", "raised"],
)
async def test_an_early_mint_that_fails_is_logged_and_leaves_the_card_and_plan_alone(
    pg_engine: AsyncEngine,
    real_schemas: list[ToolSchema],
    caplog: pytest.LogCaptureFixture,
    outcome: str | Exception,
    logged: str,
) -> None:
    plan = _planner_plan(schemas=real_schemas)

    async def card(early_mints: EarlyMints | None) -> tuple[Any, Any, str | None]:
        planner = _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}])
        runner, _ = _plan_runner(
            pg_engine,
            planner,
            early_mints=early_mints,
            conversation_id="conv_early" if early_mints is not None else "conv_none",
        )
        events = await _start(runner)
        await asyncio.sleep(0)
        return (
            _output_of(events, "planner"),
            _session_state(runner)["plan"],
            _pending_interrupt(events),
        )

    sends = _EarlySends(outcome)
    with caplog.at_level("INFO", logger="harness.early_mint"):
        offered = await card(EarlyMints(send=sends, max_concurrent=4, settle_s=3.0))

    assert offered == await card(None)
    assert offered[2] == APPROVE_PAUSE
    (line,) = [r.getMessage() for r in caplog.records if r.name == "harness.early_mint"]
    assert f"spec_id={SPEC_A}" in line
    assert f"plan_revision={plan_revision(plan)}" in line
    assert logged in line
    assert "elapsed_s=" in line


async def test_a_call_unsent_at_one_stream_end_logs_one_line_with_its_own_elapsed_time(
    real_schemas: list[ToolSchema], caplog: pytest.LogCaptureFixture
) -> None:
    """A call still pending when the stream ends is logged unsent then, with
    the seconds since its card was offered, and logs nothing when it ends."""
    hold = asyncio.Event()
    sends = _EarlySends(hold=hold)
    now = [100.0]
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=0.01, clock=lambda: now[0])
    early.offer(_planner_plan(schemas=real_schemas))

    with caplog.at_level("INFO", logger="harness.early_mint"):
        now[0] = 102.5
        await early.settle()
        await early.settle()
        now[0] = 109.0
        hold.set()
        await asyncio.gather(*early._tasks)

    (line,) = [r.getMessage() for r in caplog.records if r.name == "harness.early_mint"]
    assert "outcome=unsent_at_stream_end" in line
    assert "elapsed_s=2.50" in line


async def test_a_cards_early_calls_go_out_no_more_than_the_cap_at_once(
    real_schemas: list[ToolSchema],
) -> None:
    """A card with more retrievals than the cap sends the cap's worth, and the
    rest as those end."""
    wide = _with_retrievals(_planner_plan(schemas=real_schemas), 3)
    hold = asyncio.Event()
    sends = _EarlySends(hold=hold)
    early = EarlyMints(send=sends, max_concurrent=2, settle_s=3.0)

    early.offer(wide)
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(sends.sent) == 2

    hold.set()
    await asyncio.gather(*early._tasks)
    assert len(sends.sent) == 3


def _with_retrievals(plan: Any, count: int) -> Any:
    """`plan` with its one retrieval step `count` times over, unvalidated: only
    `EarlyMints` reads it."""
    assert plan.workflow is not None
    (retrieval,) = [step for step in plan.workflow.steps if step.tool == _GET_DATA]
    return plan.model_copy(
        update={"workflow": plan.workflow.model_copy(update={"steps": (retrieval,) * count})}
    )


async def test_offering_a_newer_card_withdraws_the_older_cards_calls_still_waiting_for_a_slot(
    real_schemas: list[ToolSchema], caplog: pytest.LogCaptureFixture
) -> None:
    """The older card's call already out runs on; its calls still queued are
    withdrawn unsent and logged, and the newer card's call goes out."""
    hold = asyncio.Event()
    sends = _EarlySends(hold=hold)
    early = EarlyMints(send=sends, max_concurrent=1, settle_s=3.0)

    with caplog.at_level("INFO", logger="harness.early_mint"):
        early.offer(_with_retrievals(_planner_plan(schemas=real_schemas), 3))
        await asyncio.sleep(0)
        assert len(sends.sent) == 1
        early.offer(_planner_plan(question="kenya debt, revised", schemas=real_schemas))
        hold.set()
        await asyncio.gather(*early._tasks, return_exceptions=True)

    assert len(sends.sent) == 2
    assert sends.cancelled == 0
    lines = [r.getMessage() for r in caplog.records if r.name == "harness.early_mint"]
    assert sum("outcome=withdrawn" in line for line in lines) == 2


async def test_withdrawing_cancels_the_latest_cards_calls_still_waiting_for_a_slot(
    real_schemas: list[ToolSchema],
) -> None:
    hold = asyncio.Event()
    sends = _EarlySends(hold=hold)
    early = EarlyMints(send=sends, max_concurrent=1, settle_s=3.0)
    early.offer(_with_retrievals(_planner_plan(schemas=real_schemas), 3))
    await asyncio.sleep(0)
    assert len(sends.sent) == 1

    early.withdraw()
    hold.set()
    await asyncio.gather(*early._tasks, return_exceptions=True)

    assert len(sends.sent) == 1
    assert sends.cancelled == 0


async def test_evicting_a_conversation_withdraws_its_calls_still_waiting_for_a_slot(
    real_schemas: list[ToolSchema],
) -> None:
    """The oldest conversation's record is dropped once `MAX_CALLERS` newer
    conversations have offered. Its call already out runs on; its queued
    calls never start."""
    hold = asyncio.Event()
    sends = _EarlySends(hold=hold)
    early = EarlyMints(send=sends, max_concurrent=1, settle_s=3.0)
    oldest = _with_retrievals(_planner_plan(schemas=real_schemas), 3)
    other = _planner_plan(schemas=real_schemas)

    current_conversation_id.set("conversation-0")
    early.offer(oldest)
    await asyncio.sleep(0)
    assert len(sends.sent) == 1
    for n in range(1, MAX_CALLERS + 1):
        current_conversation_id.set(f"conversation-{n}")
        early.offer(other)
    hold.set()
    await asyncio.gather(*early._tasks, return_exceptions=True)

    assert len(sends.sent) == 1 + MAX_CALLERS
    assert sends.cancelled == 0


async def test_approving_a_card_withdraws_its_early_calls_and_offering_it_does_not(
    pg_engine: AsyncEngine, real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _planner_plan(schemas=real_schemas)
    early = EarlyMints(send=_EarlySends(), max_concurrent=4, settle_s=3.0)
    withdrawn: list[str] = []
    monkeypatch.setattr(early, "withdraw", lambda: withdrawn.append("withdraw"))
    runner, _ = _plan_runner(
        pg_engine,
        _Planner([{"plan": plan.model_dump(mode="json"), "summary": {}}]),
        tool_surface=_fixture_surface(real_schemas),
        early_mints=early,
    )

    await _start(runner)
    assert withdrawn == []

    await _resume(runner, APPROVE_PAUSE, approve_answer(plan_revision(plan)))
    assert withdrawn == ["withdraw"]


async def test_closing_early_mints_cancels_the_calls_still_running(
    real_schemas: list[ToolSchema],
) -> None:
    sends = _EarlySends(hold=asyncio.Event())
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=3.0)
    early.offer(_planner_plan(schemas=real_schemas))
    await asyncio.sleep(0)
    assert len(sends.sent) == 1

    await early.aclose()

    assert sends.cancelled == 1
    # An offer after the close, as a cut segment's owed work can make at
    # shutdown, starts nothing.
    early.offer(_planner_plan(schemas=real_schemas))
    await asyncio.sleep(0)
    assert len(sends.sent) == 1
